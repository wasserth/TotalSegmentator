"""
Smooth label maps: each model's logits interpolated onto the input grid and composited there.

The default pipeline takes the argmax of every model on the model grid, composites the labels,
and upsamples the result to the input grid with nearest-neighbor sampling
(change_spacing(order=0)), which is why the output looks blocky at the model's voxel size.
With smooth labels each model's logits are interpolated trilinearly onto the input grid and the
argmax is taken there, voxel by voxel, painting into one label map with the default
compositor's rule (a voxel whose label is 0 is left to the models before it). Nothing
K-channel-sized is ever allocated at input resolution: the interpolation and the decision are one
fused kernel from labelfield (Metal on Apple GPUs, Triton on CUDA, torch on anything else).

An input voxel is mapped to the logits through every step the input went through, in order:
change_spacing's forward resample (scipy.ndimage.zoom: the voxel-corner rule), the triple split
along z, nnU-Net's crop to the nonzero region, and nnU-Net's own resample (the voxel-center
rule; usually the identity, since the input already has the model's spacing, but not for a
task whose spacing differs from its plans', such as lung_nodules). With interp="nearest" the
result is the default pipeline's upsampled labels exactly where that resample is the identity,
and before postprocessing: the default postprocesses (-rmb, vertebrae_pp, body) on the model grid
and then upsamples, the smooth path upsamples and then postprocesses on the input grid.
"""
import numpy as np
import torch

try:
    from labelfield.backends import triton_gpu as _triton_gpu
    # labelfield < 0.1.3 keeps a failed triton import as an exception; its traceback reaches every
    # frame live at that import (a whole prediction's arrays and models) for the life of the process
    if isinstance(getattr(_triton_gpu, "_TRITON_IMPORT_ERROR", None), BaseException):
        _triton_gpu._TRITON_IMPORT_ERROR.__traceback__ = None
except ImportError:
    pass


def _labelfield():
    try:
        import labelfield
    except ImportError as e:
        raise ImportError("Smooth label maps need the labelfield package: "
                          "pip install 'labelfield[torch] @ git+https://github.com/mhalle/labelfield.git'") from e
    return labelfield


class SmoothComposite:
    """One label map on the input grid, painted model by model while each model's logits exist.

    input_shape / model_shape: the canonical (nibabel XYZ) shapes of the image before and after
    change_spacing. The label map lives on `device` in nnU-Net's ZYX order until `result()`.
    """

    def __init__(self, input_shape, model_shape, device, interp="linear", chunk_bytes=1 << 30):
        self.lf = _labelfield()
        self.out_shape = tuple(int(s) for s in input_shape[::-1])
        self.model_shape = tuple(int(s) for s in model_shape[::-1])
        self.to_model = self.lf.Mapping.corner(self.out_shape, self.model_shape)
        self.device = torch.device(device)
        self.interp = interp
        self.chunk_bytes = int(chunk_bytes)
        self.out = torch.zeros(self.out_shape, dtype=torch.uint8, device=self.device)

    def to_logits(self, properties, logits_shape, z_offset=0):
        """Input index (ZYX) -> coordinate in the logits of the piece starting at model plane z_offset."""
        Mapping = self.lf.Mapping
        start = [int(b[0]) for b in properties["bbox_used_for_cropping"]]
        cropped = [int(s) for s in properties["shape_after_cropping_and_before_resampling"]]
        shift = Mapping((1.0, 1.0, 1.0), (-(z_offset + start[0]), -start[1], -start[2]))
        return self.to_model >> shift >> Mapping.center(cropped, logits_shape)

    def planes_for(self, keep):
        """The input planes whose nearest model plane is one of `keep` = (first, end)."""
        n_model = self.model_shape[0]
        cz = self.to_model.a[0] * np.arange(self.out_shape[0]) + self.to_model.b[0]
        nearest = np.clip(np.floor(cz + 0.5), 0, n_model - 1)
        first, end = keep if keep is not None else (0, n_model)
        js = np.nonzero((nearest >= first) & (nearest < end))[0]
        return (int(js[0]), int(js[-1]) + 1) if js.size else (0, 0)

    def paint(self, logits, properties, lut=None, z_offset=0, keep=None):
        """Paint one model's logits (K, Z, Y, X) on one piece of the model grid.

        lut: local label -> output label (index = channel); None is the identity.
        z_offset: the model plane the piece starts at; keep: the model planes (first, end) this
        piece is responsible for - the triple split's kept range, so seams match the default.
        """
        K = int(logits.shape[0])
        logits_shape = tuple(int(s) for s in logits.shape[1:])
        table = np.arange(K) if lut is None else np.asarray(lut, dtype=np.int64)
        table = np.pad(table, (0, max(0, K - len(table))))[:K]
        full = self.to_logits(properties, logits_shape, z_offset)
        j0, j1 = self.planes_for(keep)
        if j1 <= j0:
            return
        # Move the logits to the device a slab at a time: a whole part can be several GB.
        a, b = full.a[0], full.b[0]
        n_logit = logits_shape[0]
        plane_bytes = K * logits_shape[1] * logits_shape[2] * logits.element_size()
        planes = max(4, self.chunk_bytes // max(1, plane_bytes))
        step = max(1, int((planes - 4) / a)) if a > 0 else j1 - j0
        for ja in range(j0, j1, step):
            jb = min(ja + step, j1)
            lo = int(np.clip(np.floor(a * ja + b) - 1, 0, n_logit))
            hi = int(np.clip(np.floor(a * (jb - 1) + b) + 3, lo + 1, n_logit))
            if hi <= lo:
                continue
            slab = logits[:, lo:hi]
            if slab.device != self.device:
                slab = slab.to(self.device)
            # out_start keeps each input plane's coordinate a * j + b, computed from its own index
            # (only the integer lo moves into b), so slabs decide exactly as one call would
            mapping = full >> self.lf.Mapping((1.0, 1.0, 1.0), (-lo, 0.0, 0.0))
            self.lf.to_labels(slab, (jb - ja, *self.out_shape[1:]), mapping, interp=self.interp,
                              lut=table, paint=True, transparent="zero", out=self.out[ja:jb],
                              out_start=(ja, 0, 0))

    def result(self):
        """The label map as a nibabel-order (XYZ) uint8 array."""
        return self.out.cpu().numpy().transpose(2, 1, 0)
