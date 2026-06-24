"""Pass 3 — per-layer bounding boxes, alignment summaries, and loop containment."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_layer_bounding_boxes(layers: dict) -> dict[str, list[float]]:
    """
    Compute a tight bounding box for each layer based on its line endpoints
    and circle/arc centers.

    Returns {layer_name: [min_x, min_y, max_x, max_y]}.
    """
    logger.info("Computing per-layer bounding boxes...")
    bboxes: dict[str, list[float]] = {}

    for layer_name, layer_data in layers.items():
        xs: list[float] = []
        ys: list[float] = []

        for seg in layer_data.get("lines", []):
            xs.extend([seg["start"][0], seg["end"][0]])
            ys.extend([seg["start"][1], seg["end"][1]])

        for ca in layer_data.get("circles_arcs", []):
            r = ca["radius"]
            xs.extend([ca["center"][0] - r, ca["center"][0] + r])
            ys.extend([ca["center"][1] - r, ca["center"][1] + r])

        for t in layer_data.get("text_annotations", []):
            xs.append(t["position"][0])
            ys.append(t["position"][1])

        if xs and ys:
            bboxes[layer_name] = [
                round(min(xs), 2), round(min(ys), 2),
                round(max(xs), 2), round(max(ys), 2),
            ]

    logger.info("Per-layer bounding boxes computed for %d layers.", len(bboxes))
    return bboxes


def compute_alignment_summary(layers: dict) -> dict[str, dict[str, int]]:
    """
    Summarize line alignment vectors per layer.

    Returns {layer_name: {"horizontal": n, "vertical": n, "diagonal": n}}.
    """
    logger.info("Computing alignment vector summary...")
    summary: dict[str, dict[str, int]] = {}

    for layer_name, layer_data in layers.items():
        counts = {"horizontal": 0, "vertical": 0, "diagonal": 0}
        for seg in layer_data.get("lines", []):
            alignment = seg.get("alignment", "diagonal")
            counts[alignment] += 1
        if any(counts.values()):
            summary[layer_name] = counts

    total_h = sum(s["horizontal"] for s in summary.values())
    total_v = sum(s["vertical"] for s in summary.values())
    total_d = sum(s["diagonal"] for s in summary.values())
    logger.info(
        "Alignment summary: horizontal=%d, vertical=%d, diagonal=%d",
        total_h, total_v, total_d,
    )
    return summary


def _point_in_polygon(px: float, py: float, polygon: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def check_containment(
    layers: dict,
    loop_metadata: list[dict],
    nodes: list[list[float]],
) -> list[dict]:
    """
    Determine which text annotations and circles/arcs are contained within
    each detected loop polygon.

    Updates loop_metadata entries in-place with 'contained_text' and
    'contained_symbols' lists. Returns the updated loop_metadata.
    """
    logger.info("Checking containment of text/symbols within %d loops...", len(loop_metadata))

    # Build polygons from loop node coordinates
    loop_polygons = []
    for loop in loop_metadata:
        poly = [nodes[nid] for nid in loop["node_ids"]]
        loop_polygons.append(poly)

    # Gather all text and symbol points across layers
    all_text: list[dict] = []
    all_symbols: list[dict] = []
    for layer_name, layer_data in layers.items():
        for t in layer_data.get("text_annotations", []):
            all_text.append({"layer": layer_name, **t})
        for ca in layer_data.get("circles_arcs", []):
            all_symbols.append({"layer": layer_name, **ca})

    contained_counts = 0
    for idx, (loop, poly) in enumerate(zip(loop_metadata, loop_polygons)):
        contained_text = []
        contained_symbols = []

        for t in all_text:
            if _point_in_polygon(t["position"][0], t["position"][1], poly):
                contained_text.append({"text": t["text"], "position": t["position"], "layer": t["layer"]})

        for s in all_symbols:
            if _point_in_polygon(s["center"][0], s["center"][1], poly):
                contained_symbols.append({"center": s["center"], "radius": s["radius"], "layer": s["layer"]})

        loop["contained_text"] = contained_text
        loop["contained_symbols"] = contained_symbols
        contained_counts += len(contained_text) + len(contained_symbols)

    logger.info(
        "Containment check complete. %d items assigned to loops.", contained_counts
    )
    return loop_metadata


def spatial_descriptors(
    layers: dict, topology: dict, nodes: list[list[float]]
) -> dict:
    """
    Run Pass 3: compute per-layer bounding boxes, alignment summaries,
    and containment of text/symbols within detected loops.
    """
    logger.info("═══ Pass 3: Spatial Descriptors ═══")

    layer_bboxes = compute_layer_bounding_boxes(layers)
    alignment_summary = compute_alignment_summary(layers)

    # Containment updates loop metadata in-place
    loop_metadata = topology.get("loops", [])
    if loop_metadata and nodes:
        check_containment(layers, loop_metadata, nodes)

    return {
        "layer_bounding_boxes": layer_bboxes,
        "alignment_summary": alignment_summary,
    }
