#!/usr/bin/env python3
"""Pure occupancy-grid contract used by world-anchored validation."""

from __future__ import annotations

import numpy as np


def rasterize_convex_polygon(
    polygon_xy: np.ndarray,
    origin_xy: np.ndarray,
    resolution: float,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Return cells touched by an ordered convex polygon in map coordinates."""
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    origin = np.asarray(origin_xy, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError("polygon_xy must contain at least three [x, y] vertices")
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")

    height, width = grid_shape
    padding = resolution / np.sqrt(2.0)
    low = np.min(polygon, axis=0) - padding
    high = np.max(polygon, axis=0) + padding
    col0, row0 = np.floor((low - origin) / resolution).astype(int)
    col1, row1 = np.ceil((high - origin) / resolution).astype(int)
    col0, row0 = max(0, col0), max(0, row0)
    col1, row1 = min(width - 1, col1), min(height - 1, row1)
    mask = np.zeros(grid_shape, dtype=bool)
    if col0 > col1 or row0 > row1:
        return mask

    rows, cols = np.mgrid[row0:row1 + 1, col0:col1 + 1]
    centers = origin + np.stack(
        ((cols + 0.5) * resolution, (rows + 0.5) * resolution), axis=-1
    )
    edges = np.roll(polygon, -1, axis=0) - polygon
    relative = centers[..., None, :] - polygon
    cross = (
        edges[None, None, :, 0] * relative[..., 1]
        - edges[None, None, :, 1] * relative[..., 0]
    )
    inside = np.all(cross >= -1e-12, axis=-1) | np.all(cross <= 1e-12, axis=-1)

    edge_norm_sq = np.sum(edges * edges, axis=1)
    valid_edges = edge_norm_sq > 1e-18
    distances_sq = np.full((*centers.shape[:2], len(edges)), np.inf)
    if np.any(valid_edges):
        projections = np.sum(relative[..., valid_edges, :] * edges[valid_edges], axis=-1)
        projections /= edge_norm_sq[valid_edges]
        projections = np.clip(projections, 0.0, 1.0)
        nearest = (
            polygon[valid_edges]
            + projections[..., None] * edges[valid_edges]
        )
        delta = centers[..., None, :] - nearest
        distances_sq[..., valid_edges] = np.sum(delta * delta, axis=-1)
    touched = inside | (np.min(distances_sq, axis=-1) <= padding * padding)
    mask[row0:row1 + 1, col0:col1 + 1] = touched
    return mask


def combine_static_and_scene_grid(
    source_grid: np.ndarray,
    scene_occupied: np.ndarray,
    augment_scene_geometry: bool,
) -> tuple[np.ndarray, dict[str, int | bool | str]]:
    """Keep the source PGM exact unless scene augmentation is explicit."""
    if source_grid.shape != scene_occupied.shape:
        raise ValueError("source and scene grids must have identical shapes")
    output = source_grid.copy()
    source_occupied = source_grid == 100
    available = np.asarray(scene_occupied, dtype=bool) & ~source_occupied
    if augment_scene_geometry:
        output[np.asarray(scene_occupied, dtype=bool)] = 100
    return output, {
        "static_grid_mode": (
            "source_map_plus_scene_geometry"
            if augment_scene_geometry else "source_map_exact"
        ),
        "scene_geometry_applied_to_grid": bool(augment_scene_geometry),
        "scene_cells_not_in_source_map": int(np.count_nonzero(available)),
        "scene_cells_added": int(np.count_nonzero(output == 100) -
                                 np.count_nonzero(source_occupied)),
    }
