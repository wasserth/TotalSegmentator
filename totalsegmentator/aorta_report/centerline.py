from __future__ import annotations

from collections import deque
from itertools import combinations

import networkx as nx
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize


class Vertex:
    def __init__(self, point):
        self.point = np.asarray(point)

    def __str__(self):
        return str(self.point)


_NEIGHBORS = np.array(
    [
        (x, y, z)
        for z in (-1, 0, 1)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        if (x, y, z) != (0, 0, 0)
    ]
)


def _build_trees(data):
    """Build the same breadth-first spanning trees used by the source report."""
    unseen = data.astype(bool).copy()
    graphs = []
    shape = np.asarray(unseen.shape)
    while unseen.any():
        start = np.argwhere(unseen)[0]
        start_vertex = Vertex(start)
        graph = nx.Graph()
        graph.add_node(start_vertex)
        unseen[tuple(start)] = False
        queue = deque([start_vertex])
        while queue:
            current = queue.popleft()
            for offset in _NEIGHBORS:
                point = current.point + offset
                if np.any(point < 0) or np.any(point >= shape) or not unseen[tuple(point)]:
                    continue
                vertex = Vertex(point)
                graph.add_edge(current, vertex)
                unseen[tuple(point)] = False
                queue.append(vertex)
        graphs.append(graph)
    return graphs


def get_len_path(path):
    return float(get_cumulative_arc_lengths(path)[-1]) if path else 0.0


def get_distance(a, b, spacing):
    return np.sqrt(np.sum(((np.asarray(a) - np.asarray(b)) * spacing) ** 2))


def get_len_path_mm(path, spacing):
    lengths = get_cumulative_arc_lengths(path, spacing)
    return np.float32(lengths[-1] if len(lengths) else 0.0)


def get_cumulative_arc_lengths(centerline, spacing=None):
    """Return cumulative distances along a centerline, including zero."""
    if not centerline:
        return np.empty(0, dtype=float)
    points = np.asarray([vertex.point for vertex in centerline], dtype=float)
    differences = np.diff(points, axis=0)
    if spacing is not None:
        differences *= np.asarray(spacing, dtype=float)
    segment_lengths = np.linalg.norm(differences, axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def resample_points_by_arc_length(points, positions):
    """Linearly sample points at positions along their cumulative arc length."""
    points = np.asarray(points, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if not len(points) or not len(positions):
        return np.empty((0, 3), dtype=float)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    # Repeated points make np.interp's x coordinates ambiguous. Keeping the
    # first point and the last point at each later distance preserves endpoints.
    unique, first = np.unique(cumulative, return_index=True)
    last = np.searchsorted(cumulative, unique, side="right") - 1
    indexes = np.where(unique == 0, first, last)
    return np.column_stack(
        [np.interp(positions, unique, points[indexes, dim]) for dim in range(3)]
    )


def _paths_from_skeleton(skeleton):
    candidates = []
    for graph in _build_trees(skeleton):
        endpoints = [node for node in graph.nodes if graph.degree(node) == 1]
        if len(endpoints) < 2:
            continue
        longest = max(
            combinations(endpoints, 2),
            key=lambda pair: nx.shortest_path_length(graph, pair[0], pair[1]),
        )
        candidates.append(nx.shortest_path(graph, longest[0], longest[1]))
    return candidates


def _rasterize_centerline(shape, path):
    image = np.zeros(shape)
    limit = np.asarray(shape) - 1
    for vertex in path:
        point = np.clip(np.rint(vertex.point), 0, limit).astype(int)
        image[tuple(point)] = 1
    return image


def _axis_centerline(mask):
    """Straight centerline through a mask whose skeleton has no usable path."""
    points = np.argwhere(mask)
    if len(points) < 2:
        return None
    centered = points.astype(float) - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ axes[0]
    start = points[int(np.argmin(projection))].astype(float)
    end = points[int(np.argmax(projection))].astype(float)
    length = float(np.linalg.norm(end - start))
    if length < 1:
        return None
    steps = max(int(np.ceil(length)), 1)
    return [Vertex(start + idx / steps * (end - start)) for idx in range(steps + 1)]


def get_centerline(data, debug=False):
    if debug:
        print("Skeletonize...")
    mask = np.asarray(data) > 0
    skeleton = skeletonize(mask) if mask.any() else np.zeros(mask.shape, dtype=bool)
    if debug:
        print("Building graph....")
    candidates = _paths_from_skeleton(skeleton) if skeleton.any() else []
    # Lee thinning deletes objects whose thickness is even, leaving an empty
    # skeleton. One extra dilation makes that thickness odd.
    if not candidates and mask.any():
        dilated_skeleton = skeletonize(binary_dilation(mask))
        if dilated_skeleton.any():
            candidates = _paths_from_skeleton(dilated_skeleton)
    if candidates:
        path = max(candidates, key=get_len_path)
        return _rasterize_centerline(mask.shape, path), path
    path = _axis_centerline(mask)
    if path is None:
        raise ValueError("Unable to create a centerline from the aorta mask.")
    return _rasterize_centerline(mask.shape, path), path


def reorder_centerline(centerline):
    if len(centerline) < 2:
        return centerline
    return centerline[::-1] if centerline[0].point[2] > centerline[-1].point[2] else centerline


def get_index_for_point(centerline, point):
    if point is None:
        return None
    for idx, vertex in enumerate(centerline):
        if np.array_equal(vertex.point, point):
            return idx
    return None


def get_mid_of_points(centerline, a, b):
    if not centerline or a is None or b is None:
        return None
    if not (0 <= a < len(centerline) and 0 <= b < len(centerline)):
        return None
    if b < a:
        a, b = b, a
    target = get_len_path(centerline[a:b]) // 2
    length = 0.0
    for idx in range(max(1, a), b):
        length += np.linalg.norm(
            centerline[idx].point - centerline[idx - 1].point
        )
        if length >= target:
            return idx
    return a if a == b else None


def get_next_point_by_distance(centerline, start, distance, spacing):
    if not centerline or start is None or not 0 <= start < len(centerline):
        return None
    if distance == 0:
        return start
    length = 0.0
    if distance > 0:
        indexes = range(start + 1, len(centerline) - 1)
        previous_offset = -1
    else:
        indexes = reversed(range(0, start - 1))
        previous_offset = 1
    for idx in indexes:
        length += get_distance(
            centerline[idx].point,
            centerline[idx + previous_offset].point,
            spacing,
        )
        if length >= abs(distance):
            return idx
    return None


def add_point_to_centerline(image, centerline, point):
    if point is None:
        return image, centerline
    point = np.asarray(point)
    last = centerline[-1].point
    distance = np.linalg.norm(last - point)
    if distance > 1:
        count = int(distance)
        for idx in range(1, count + 1):
            intermediate = last + idx / float(count + 1) * (point - last)
            image[tuple(intermediate.astype(int))] = 1
            centerline.append(Vertex(intermediate))
    image[tuple(point.astype(int))] = 1
    centerline.append(Vertex(point))
    return image, centerline


def extend_centerline(image, centerline, distance):
    if len(centerline) < 2:
        raise ValueError("Not enough vertices to determine direction")
    result = [Vertex(vertex.point) for vertex in centerline]
    result_image = image.copy()
    direction = result[-1].point - result[-2].point
    direction = direction / np.linalg.norm(direction)
    return add_point_to_centerline(
        result_image, result, result[-1].point + distance * direction
    )


def get_closest_point_of_centerline(point, centerline):
    if point is None or not centerline:
        return None
    points = np.asarray([vertex.point for vertex in centerline])
    return tuple(points[np.argmin(np.linalg.norm(points - point, axis=1))])


def get_closest_centerline_section(point, centerline, length_mm, spacing):
    if not centerline:
        return []
    closest = get_index_for_point(centerline, get_closest_point_of_centerline(point, centerline))
    if closest is None:
        return []
    count = int(length_mm / spacing)
    start = max(closest - count // 2, 0)
    end = min(closest + count // 2, len(centerline) - 1)
    if end - start < count - 1:
        if start == 0:
            end = min(count - 1, len(centerline) - 1)
        elif end == len(centerline) - 1:
            start = max(end - count + 1, 0)
    return centerline[start : end + 1]


def get_normal_for_cl_point(centerline, idx, dist=(3, 10)):
    if len(centerline) < 2 or idx is None or not 0 <= idx < len(centerline):
        return np.array([0.0, 0.0, 1.0])
    normals = []
    for distance in range(dist[0], dist[1]):
        start = max(0, idx - distance)
        end = min(len(centerline) - 1, idx + distance)
        normals.append(centerline[start].point - centerline[end].point)
    normal = np.asarray(normals).mean(axis=0)
    if np.linalg.norm(normal) > 0:
        return normal
    for distance in range(1, len(centerline)):
        start = max(0, idx - distance)
        end = min(len(centerline) - 1, idx + distance)
        normal = centerline[start].point - centerline[end].point
        if np.linalg.norm(normal) > 0:
            return normal
    return np.array([0.0, 0.0, 1.0])


def _distance_to_plane(point, normal, plane_point):
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.inf
    return ((np.asarray(point) @ normal) - (plane_point @ normal)) / norm


def get_closest_point_of_centerline_by_plane(point, centerline):
    if point is None or not centerline:
        return None, None
    if len(centerline) == 1:
        return tuple(centerline[0].point), get_normal_for_cl_point(centerline, 0)
    distances = []
    normals = []
    for idx, vertex in enumerate(centerline[:-1]):
        normal = get_normal_for_cl_point(centerline, idx)
        distances.append(abs(_distance_to_plane(point, normal, vertex.point)))
        normals.append(normal)
    idx = int(np.argmin(distances))
    return tuple(centerline[idx].point), normals[idx]


def resample_cl_to_same_distance(centerline, vox_to_mm_affine, dist=20):
    if dist <= 0:
        raise ValueError("dist must be greater than zero")
    points = np.asarray([vertex.point for vertex in centerline], dtype=float)
    if len(points) < 2:
        return [Vertex(point) for point in points]
    points_mm = nib.affines.apply_affine(vox_to_mm_affine, points)
    total_length = np.linalg.norm(np.diff(points_mm, axis=0), axis=1).sum()
    if total_length == 0:
        return [Vertex(points[0])]
    positions = np.linspace(0, total_length, max(2, int(total_length / dist)))
    sampled_mm = resample_points_by_arc_length(points_mm, positions)
    sampled = nib.affines.apply_affine(np.linalg.inv(vox_to_mm_affine), sampled_mm)
    return [Vertex(point) for point in sampled]


def check_cl_distances(centerline):
    print(
        np.asarray(
            [
                np.linalg.norm(centerline[idx].point - centerline[idx - 1].point)
                for idx in range(1, len(centerline))
            ]
        )
    )
