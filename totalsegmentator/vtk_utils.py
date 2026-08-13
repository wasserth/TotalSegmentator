"""Backward-compatible imports for rendering helpers.

New code should import from :mod:`totalsegmentator.rendering`.
"""

from totalsegmentator.rendering import (
    contour_from_roi_smooth,
    plot_mask,
    text as render_text,
)


def label(
    text="Origin",
    pos=(0, 0, 0),
    scale=(0.2, 0.2, 0.2),
    color=(1, 1, 1),
):
    """Create a 3D text label using the active FURY backend."""
    return render_text(text, pos, scale, color)
