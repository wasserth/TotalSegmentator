import numpy as np
import vtk
from vtk.util import numpy_support


def set_input(vtk_object, inp):
    """Set Generic input function which takes into account VTK 5 or 6.

    Parameters
    ----------
    vtk_object: vtk object
    inp: vtkPolyData or vtkImageData or vtkAlgorithmOutput

    Returns
    -------
    vtk_object

    Notes
    -------
    This can be used in the following way::
        from fury.utils import set_input
        poly_mapper = set_input(vtk.vtkPolyDataMapper(), poly_data)

    This function is copied from dipy.viz.utils
    """
    if isinstance(inp, (vtk.vtkPolyData, vtk.vtkImageData)):
        vtk_object.SetInputData(inp)
    elif isinstance(inp, vtk.vtkAlgorithmOutput):
        vtk_object.SetInputConnection(inp)
    vtk_object.Update()
    return vtk_object


def plot_mask(renderer, mask_data, affine, x_current, y_current, orientation="axial",
              smoothing=10, brain_mask=None, color=[1, .27, .18], opacity=1):
    """
    color: default is red
    """
    # 3D Bundle
    mask = mask_data
    mask = mask.transpose(0, 2, 1)
    mask = mask[::-1, :, :]
    if orientation == "sagittal":
        mask = mask.transpose(2, 1, 0)
        mask = mask[::-1, :, :]

    cont_actor = contour_from_roi_smooth(mask, affine=affine,
                                         color=color, opacity=opacity, smoothing=smoothing)
    cont_actor.SetPosition(x_current, y_current, 0)
    return cont_actor


def label(text='Origin', pos=(0, 0, 0), scale=(0.2, 0.2, 0.2),
                  color=(1, 1, 1)):

    atext = vtk.vtkVectorText()
    atext.SetText(text)

    textm = vtk.vtkPolyDataMapper()
    textm.SetInputConnection(atext.GetOutputPort())

    texta = vtk.vtkFollower()
    texta.SetMapper(textm)
    texta.SetScale(scale)

    texta.GetProperty().SetColor(color)
    texta.SetPosition(pos)

    return texta


def contour_from_roi_smooth(data, affine=None, color=np.array([1, 0, 0]), opacity=1, smoothing=0):
    """Generates surface actor from a binary ROI.
    Code from dipy, but added awesome smoothing!

    Parameters
    ----------
    data : array, shape (X, Y, Z)
        An ROI file that will be binarized and displayed.
    affine : array, shape (4, 4)
        Grid to space (usually RAS 1mm) transformation matrix. Default is None.
        If None then the identity matrix is used.
    color : (1, 3) ndarray
        RGB values in [0,1].
    opacity : float
        Opacity of surface between 0 and 1.
    smoothing: int
        Smoothing factor e.g. 10.
    Returns
    -------
    contour_assembly : vtkAssembly
        ROI surface object displayed in space
        coordinates as calculated by the affine parameter.

    """
    major_version = vtk.vtkVersion.GetVTKMajorVersion()

    if data.ndim != 3:
        raise ValueError('Only 3D arrays are currently supported.')
    else:
        nb_components = 1

    data = (data > 0) * 1
    vol = np.interp(data, xp=[data.min(), data.max()], fp=[0, 255])
    vol = vol.astype('uint8')

    im = vtk.vtkImageData()
    if major_version <= 5:
        im.SetScalarTypeToUnsignedChar()
    di, dj, dk = vol.shape[:3]
    im.SetDimensions(di, dj, dk)
    voxsz = (1., 1., 1.)
    # im.SetOrigin(0,0,0)
    im.SetSpacing(voxsz[2], voxsz[0], voxsz[1])
    if major_version <= 5:
        im.AllocateScalars()
        im.SetNumberOfScalarComponents(nb_components)
    else:
        im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, nb_components)

    # copy data
    vol = np.swapaxes(vol, 0, 2)
    vol = np.ascontiguousarray(vol)

    if nb_components == 1:
        vol = vol.ravel()
    else:
        vol = np.reshape(vol, [np.prod(vol.shape[:3]), vol.shape[3]])

    uchar_array = numpy_support.numpy_to_vtk(vol, deep=0)
    im.GetPointData().SetScalars(uchar_array)

    if affine is None:
        affine = np.eye(4)

    # Set the transform (identity if none given)
    transform = vtk.vtkTransform()
    transform_matrix = vtk.vtkMatrix4x4()
    transform_matrix.DeepCopy((
        affine[0][0], affine[0][1], affine[0][2], affine[0][3],
        affine[1][0], affine[1][1], affine[1][2], affine[1][3],
        affine[2][0], affine[2][1], affine[2][2], affine[2][3],
        affine[3][0], affine[3][1], affine[3][2], affine[3][3]))
    transform.SetMatrix(transform_matrix)
    transform.Inverse()

    # Set the reslicing
    image_resliced = vtk.vtkImageReslice()
    set_input(image_resliced, im)
    image_resliced.SetResliceTransform(transform)
    image_resliced.AutoCropOutputOn()

    # Adding this will allow to support anisotropic voxels
    # and also gives the opportunity to slice per voxel coordinates

    rzs = affine[:3, :3]
    zooms = np.sqrt(np.sum(rzs * rzs, axis=0))
    image_resliced.SetOutputSpacing(*zooms)

    image_resliced.SetInterpolationModeToLinear()
    image_resliced.Update()

    # skin_extractor = vtk.vtkContourFilter()
    skin_extractor = vtk.vtkMarchingCubes()
    if major_version <= 5:
        skin_extractor.SetInput(image_resliced.GetOutput())
    else:
        skin_extractor.SetInputData(image_resliced.GetOutput())
    skin_extractor.SetValue(0, 100)

    if smoothing > 0:
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(skin_extractor.GetOutputPort())
        smoother.SetNumberOfIterations(smoothing)
        smoother.SetRelaxationFactor(0.1)
        smoother.SetFeatureAngle(60)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOff()
        smoother.SetConvergence(0)
        smoother.Update()

    skin_normals = vtk.vtkPolyDataNormals()
    if smoothing > 0:
        skin_normals.SetInputConnection(smoother.GetOutputPort())
    else:
        skin_normals.SetInputConnection(skin_extractor.GetOutputPort())
    skin_normals.SetFeatureAngle(60.0)

    skin_mapper = vtk.vtkPolyDataMapper()
    skin_mapper.SetInputConnection(skin_normals.GetOutputPort())
    skin_mapper.ScalarVisibilityOff()

    skin_actor = vtk.vtkActor()
    skin_actor.SetMapper(skin_mapper)
    skin_actor.GetProperty().SetOpacity(opacity)
    skin_actor.GetProperty().SetColor(color[0], color[1], color[2])

    return skin_actor
