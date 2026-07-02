"""Pass 4 — per-layer statistics and text-to-geometry proximity anchoring."""

from __future__ import annotations

import logging
import math
import re

logger = logging.getLogger(__name__)


def compute_layer_statistics(layers: dict) -> dict[str, dict]:
    """
    Aggregate per-layer statistics: entity counts, total line length,
    and dominant alignment.
    """
    logger.info("Computing layer statistics...")
    stats: dict[str, dict] = {}

    for layer_name, layer_data in layers.items():
        lines = layer_data.get("lines", [])
        circles_arcs = layer_data.get("circles_arcs", [])
        text_anns = layer_data.get("text_annotations", [])

        total_length = 0.0
        alignments = {"horizontal": 0, "vertical": 0, "diagonal": 0}
        for seg in lines:
            dx = seg["end"][0] - seg["start"][0]
            dy = seg["end"][1] - seg["start"][1]
            total_length += math.hypot(dx, dy)
            alignment = seg.get("alignment", "diagonal")
            alignments[alignment] += 1

        dominant = max(alignments, key=alignments.get) if any(alignments.values()) else None

        stats[layer_name] = {
            "line_count": len(lines),
            "circle_arc_count": len(circles_arcs),
            "text_count": len(text_anns),
            "total_line_length": round(total_length, 2),
            "dominant_alignment": dominant,
        }

    logger.info("Layer statistics computed for %d layers.", len(stats))
    return stats


def anchor_text_to_geometry(
    layers: dict, radius: float = 50.0
) -> list[dict]:
    """
    Link floating text annotations to the nearest geometric entity (line
    midpoint or circle/arc center) within a search radius.

    Returns a list of anchoring records:
      {text, text_position, layer, nearest_entity_type, nearest_point, distance}
    """
    logger.info("Anchoring text to nearest geometry (radius=%.1f)...", radius)

    # Collect all geometry reference points with their metadata
    geo_points: list[tuple[float, float, str, str]] = []  # (x, y, layer, type)
    for layer_name, layer_data in layers.items():
        for seg in layer_data.get("lines", []):
            mx = (seg["start"][0] + seg["end"][0]) / 2
            my = (seg["start"][1] + seg["end"][1]) / 2
            geo_points.append((mx, my, layer_name, "line_midpoint"))
        for ca in layer_data.get("circles_arcs", []):
            geo_points.append((ca["center"][0], ca["center"][1], layer_name, "circle_arc_center"))

    anchors: list[dict] = []
    unanchored = 0

    for layer_name, layer_data in layers.items():
        for t in layer_data.get("text_annotations", []):
            tx, ty = t["position"]
            best_dist = float("inf")
            best_point = None
            best_type = None
            best_layer = None

            for gx, gy, glayer, gtype in geo_points:
                d = math.hypot(tx - gx, ty - gy)
                if d < best_dist:
                    best_dist = d
                    best_point = [round(gx, 2), round(gy, 2)]
                    best_type = gtype
                    best_layer = glayer

            if best_dist <= radius:
                anchors.append({
                    "text": t["text"],
                    "text_position": t["position"],
                    "text_layer": layer_name,
                    "nearest_entity_type": best_type,
                    "nearest_point": best_point,
                    "nearest_entity_layer": best_layer,
                    "distance": round(best_dist, 2),
                })
            else:
                unanchored += 1

    logger.info(
        "Text anchoring complete. %d anchored, %d unanchored (beyond radius).",
        len(anchors), unanchored,
    )
    return anchors


def _estimate_anchor_radius(layers: dict) -> float:
    """
    Estimate an appropriate text-to-geometry anchor radius based on the
    actual spacing in the drawing.

    Computes the median nearest-neighbor distance from text positions to
    geometry reference points, then uses 2x median as the radius.
    Falls back to a generous default if insufficient data.
    """
    # Collect geometry reference points
    geo_points: list[tuple[float, float]] = []
    for layer_data in layers.values():
        for seg in layer_data.get("lines", []):
            mx = (seg["start"][0] + seg["end"][0]) / 2
            my = (seg["start"][1] + seg["end"][1]) / 2
            geo_points.append((mx, my))
        for ca in layer_data.get("circles_arcs", []):
            geo_points.append((ca["center"][0], ca["center"][1]))

    if not geo_points:
        return 500.0

    # Sample text positions
    text_positions: list[tuple[float, float]] = []
    for layer_data in layers.values():
        for t in layer_data.get("text_annotations", []):
            text_positions.append((t["position"][0], t["position"][1]))

    if not text_positions:
        return 500.0

    # Compute nearest-neighbor distances (sample up to 100 text entities)
    sample = text_positions[:100]
    nn_dists: list[float] = []
    for tx, ty in sample:
        best = float("inf")
        for gx, gy in geo_points:
            d = math.hypot(tx - gx, ty - gy)
            if d < best:
                best = d
        if best < float("inf"):
            nn_dists.append(best)

    if not nn_dists:
        return 500.0

    nn_dists.sort()
    median_dist = nn_dists[len(nn_dists) // 2]
    radius = max(median_dist * 2.0, 50.0)  # minimum 50, typically 2x median

    logger.info(
        "Adaptive anchor radius: median NN distance=%.1f, using radius=%.1f",
        median_dist, radius,
    )
    return round(radius, 1)


_ROOM_LAYER_KEYWORDS = re.compile(r"TEXT|ANNO|ROOM|NAME|LABEL", re.IGNORECASE)


def assign_rooms(layers: dict) -> dict:
    """Nearest-centroid room assignment using text on room-label layers."""
    # Collect room labels
    room_labels: list[dict] = []
    for layer_name, layer_data in layers.items():
        if not _ROOM_LAYER_KEYWORDS.search(layer_name):
            continue
        for t in layer_data.get("text_annotations", []):
            room_labels.append({"label": t["text"], "position": t["position"], "entities": []})

    if not room_labels:
        return {"rooms": []}

    # Assign every entity to nearest room label. Each entity dict is expected
    # to carry a "handle" key (the CAD entity handle, e.g. "2A3"), matching
    # the convention already used for block entities (see furniture.py). Fall
    # back to the synthetic "id" (e.g. "L_001") assigned by Pass 5's
    # _assign_entity_ids() for callers that run the standalone analyze_dxf()
    # pipeline directly against a DXF file, where no live CAD handle exists.
    for layer_name, layer_data in layers.items():
        for seg in layer_data.get("lines", []):
            mx = (seg["start"][0] + seg["end"][0]) / 2
            my = (seg["start"][1] + seg["end"][1]) / 2
            _assign_nearest(room_labels, mx, my, seg.get("handle", seg.get("id", "?")), "line", layer_name)
        for ca in layer_data.get("circles_arcs", []):
            _assign_nearest(room_labels, ca["center"][0], ca["center"][1], ca.get("handle", ca.get("id", "?")), "circle_arc", layer_name)
        for t in layer_data.get("text_annotations", []):
            _assign_nearest(room_labels, t["position"][0], t["position"][1], t.get("handle", t.get("id", "?")), "text", layer_name)
        for b in layer_data.get("blocks", []):
            _assign_nearest(room_labels, b["position"][0], b["position"][1], b.get("handle", "?"), "block", layer_name)

    return {"rooms": room_labels}


def _assign_nearest(rooms: list[dict], x: float, y: float, entity_handle: str, etype: str, layer: str) -> None:
    best_idx, best_dist = 0, float("inf")
    for i, r in enumerate(rooms):
        d = math.hypot(x - r["position"][0], y - r["position"][1])
        if d < best_dist:
            best_idx, best_dist = i, d
    rooms[best_idx]["entities"].append({"handle": entity_handle, "type": etype, "layer": layer})


def semantic_grouping(layers: dict, proximity_radius: float | None = None) -> dict:
    """
    Run Pass 4: layer aggregation statistics and text proximity anchoring.

    If proximity_radius is None, it is estimated adaptively from the drawing.
    """
    logger.info("═══ Pass 4: Semantic Grouping ═══")

    layer_stats = compute_layer_statistics(layers)

    if proximity_radius is None:
        proximity_radius = _estimate_anchor_radius(layers)

    text_anchors = anchor_text_to_geometry(layers, radius=proximity_radius)
    rooms = assign_rooms(layers)

    result = {
        "layer_statistics": layer_stats,
        "text_anchors": text_anchors,
        "anchor_radius_used": proximity_radius,
    }
    if rooms["rooms"]:
        result["rooms"] = rooms["rooms"]
    return result
