"""Pass 5 — entity ID assignment and hybrid Markdown payload generation for LLM consumption."""

from __future__ import annotations

import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


def _assign_entity_ids(layers: dict, topology: dict) -> dict[str, str]:
    """
    Assign unique IDs to all entities for LLM referencing.

    Returns a lookup of generated IDs used during Markdown generation.
    ID scheme:
      Lines:       L_<counter>
      Circles/Arcs: C_<counter>
      Text:        T_<counter>
      Loops:       LOOP_<counter>
    """
    ids: dict[str, str] = {}
    line_counter = 0
    circle_counter = 0
    text_counter = 0

    for layer_name, layer_data in layers.items():
        for seg in layer_data.get("lines", []):
            line_counter += 1
            seg["id"] = f"L_{line_counter:03d}"
        for ca in layer_data.get("circles_arcs", []):
            circle_counter += 1
            ca["id"] = f"C_{circle_counter:03d}"
        for t in layer_data.get("text_annotations", []):
            text_counter += 1
            t["id"] = f"T_{text_counter:03d}"

    for idx, loop in enumerate(topology.get("loops", []), 1):
        loop["id"] = f"LOOP_{idx:03d}"

    logger.info(
        "Entity IDs assigned: %d lines, %d circles/arcs, %d text, %d loops.",
        line_counter, circle_counter, text_counter,
        len(topology.get("loops", [])),
    )
    return ids


def generate_llm_payload(
    analysis: dict,
    topology: dict,
    spatial: dict,
    semantics: dict,
    drawing_type: str = "generic",
) -> str:
    """
    Generate the hybrid Markdown payload with embedded entity IDs for LLM
    consumption. Output format adapts based on drawing_type.
    """
    logger.info("═══ Pass 5: LLM Payload Generation (type=%s) ═══", drawing_type)

    lines: list[str] = []
    meta = analysis["metadata"]
    paper_info = meta.get("paper_info")

    # Header — adapt title to drawing type
    type_label = {
        "electrical": "ELECTRICAL SCHEMATIC",
        "architectural": "ARCHITECTURAL LAYOUT",
        "mechanical": "MECHANICAL DRAWING",
        "structural": "STRUCTURAL DRAWING",
        "generic": "CAD DRAWING",
    }.get(drawing_type, "CAD DRAWING")

    lines.append(f"# {type_label}: {meta['filename']}")
    lines.append("")

    # Global bounds and paper info
    lines.append("## DRAWING INFORMATION")
    lines.append(f"- Extents: {meta['global_dimensions']['width']:.0f} x {meta['global_dimensions']['height']:.0f} drawing units")
    if paper_info:
        lines.append(f"- Paper Size: {paper_info['paper_size']} (landscape)")
        lines.append(f"- Scale: 1:{paper_info['scale']}")
        lines.append(f"- Real-world extent: {paper_info['expected_extent'][0]}mm x {paper_info['expected_extent'][1]}mm on paper")
    lines.append(f"- Drawing Type: {drawing_type}")
    transform = meta["transform"]
    if transform.get("scale_factor", 1) > 1:
        lines.append(f"- Scale Applied: /{transform['scale_factor']:.0f} (coordinates in paper-space mm)")
    lines.append(f"- Origin Offset: ({transform['offset_x']}, {transform['offset_y']})")
    lines.append("")

    # Detected regions / enclosures
    loops = topology.get("loops", [])
    # Classify loops: "large" vs "small" relative to the biggest loop
    # Small loops are likely device symbols (< 1% of largest area)
    if loops:
        max_area = max(l["area"] for l in loops)
        threshold = max(max_area * 0.01, 1.0)
        large_loops = [l for l in loops if l["area"] >= threshold]
        small_loops = [l for l in loops if l["area"] < threshold]
    else:
        large_loops = []
        small_loops = []

    if large_loops:
        if drawing_type == "electrical":
            lines.append("## ENCLOSURES & PANELS")
        else:
            lines.append("## ROOMS & BOUNDARIES")
        lines.append("")

        for loop in large_loops:
            loop_id = loop.get("id", "LOOP_???")
            area = loop["area"]
            # For architectural drawings at known scale, convert to real-world m2
            # For schematics, just report drawing-unit area (no physical meaning)
            if drawing_type == "architectural" and paper_info and paper_info["scale"] > 1:
                real_area = area * (paper_info["scale"] ** 2) / 1_000_000
                area_str = f"~{real_area:.1f} m2"
            else:
                area_str = f"{area:.0f} sq units"

            lines.append(f"### Region [{loop_id}] ({loop['num_segments']} segments, {area_str})")
            lines.append(f"- Centroid: ({loop['centroid'][0]}, {loop['centroid'][1]})")
            lines.append(f"- BBox: ({loop['bbox'][0]}, {loop['bbox'][1]}) to ({loop['bbox'][2]}, {loop['bbox'][3]})")

            for ct in loop.get("contained_text", []):
                lines.append(f"- Label: \"{ct['text']}\" at ({ct['position'][0]}, {ct['position'][1]})")

            for cs in loop.get("contained_symbols", []):
                lines.append(f"- Symbol: R={cs['radius']} at ({cs['center'][0]}, {cs['center'][1]})")
            lines.append("")

    if small_loops:
        if drawing_type == "electrical":
            lines.append("## DEVICE SYMBOLS")
            lines.append(f"({len(small_loops)} small closed shapes detected — likely device outlines)")
        else:
            lines.append("## SMALL CLOSED SHAPES")
            lines.append(f"({len(small_loops)} detected)")
        lines.append("")

        for loop in small_loops[:20]:
            loop_id = loop.get("id", "LOOP_???")
            lines.append(f"- [{loop_id}] at ({loop['centroid'][0]}, {loop['centroid'][1]}), {loop['num_segments']} sides")
        if len(small_loops) > 20:
            lines.append(f"- ... and {len(small_loops) - 20} more")
        lines.append("")

    # Text annotations — grouped by layer, most important first
    lines.append("## ANNOTATIONS")
    lines.append("")

    layers_data = analysis.get("layers", {})
    layer_stats = semantics.get("layer_statistics", {})

    # Sort layers: text-only layers first (they carry semantic meaning),
    # then by entity count descending
    sorted_layers = sorted(
        layers_data.keys(),
        key=lambda ln: (
            0 if layer_stats.get(ln, {}).get("line_count", 0) == 0 else 1,
            -layer_stats.get(ln, {}).get("text_count", 0),
        ),
    )

    for layer_name in sorted_layers:
        layer_data = layers_data[layer_name]
        layer_text = layer_data.get("text_annotations", [])
        if not layer_text:
            continue

        lines.append(f"### Layer: {layer_name} ({len(layer_text)} labels)")
        for t in layer_text:
            t_id = t.get("id", "?")
            lines.append(f"- \"{t['text']}\" at ({t['position'][0]}, {t['position'][1]}) [{t_id}]")
        lines.append("")

    # Geometry summary per layer (compact — skip raw coordinates for schematic)
    lines.append("## GEOMETRY SUMMARY")
    lines.append("")

    for layer_name in sorted_layers:
        layer_data = layers_data[layer_name]
        stats = layer_stats.get(layer_name, {})
        bbox = spatial.get("layer_bounding_boxes", {}).get(layer_name)

        line_count = stats.get("line_count", 0)
        ca_count = stats.get("circle_arc_count", 0)
        text_count = stats.get("text_count", 0)

        if line_count == 0 and ca_count == 0:
            continue  # text-only layer already covered above

        lines.append(f"### Layer: {layer_name}")
        lines.append(
            f"- {line_count} lines ({stats.get('total_line_length', 0):.0f} units total), "
            f"{ca_count} arcs/circles, {text_count} text"
        )
        lines.append(f"- Dominant direction: {stats.get('dominant_alignment', 'N/A')}")
        if bbox:
            lines.append(f"- BBox: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")

        # For electrical schematics, just show a compact line list
        # For architectural, show more detail
        layer_lines = layer_data.get("lines", [])
        if drawing_type in ("electrical", "generic") and line_count > 30:
            # Compact: just count by alignment
            aligns = defaultdict(int)
            for seg in layer_lines:
                aligns[seg.get("alignment", "diagonal")] += 1
            align_str = ", ".join(f"{k}: {v}" for k, v in sorted(aligns.items()))
            lines.append(f"- Line directions: {align_str}")
        else:
            shown = layer_lines[:30]
            for seg in shown:
                seg_id = seg.get("id", "?")
                length = round(math.hypot(
                    seg["end"][0] - seg["start"][0],
                    seg["end"][1] - seg["start"][1],
                ), 1)
                lines.append(
                    f"  * ({seg['start'][0]},{seg['start'][1]}) to "
                    f"({seg['end'][0]},{seg['end'][1]}) | "
                    f"{seg.get('alignment', 'diagonal')[0].upper()} | L={length} [{seg_id}]"
                )
            if len(layer_lines) > 30:
                lines.append(f"  * ... and {len(layer_lines) - 30} more")

        lines.append("")

    # Topology summary
    lines.append("## TOPOLOGY")
    lines.append(f"- Nodes: {topology['nodes']['count']}")
    lines.append(f"- Edges: {topology['edges']['count']}")
    lines.append(f"- Closed Loops: {len(loops)} ({len(large_loops)} large, {len(small_loops)} small)")
    deg_dist = topology["nodes"].get("degree_distribution", {})
    if deg_dist:
        deg_str = ", ".join(f"deg-{k}: {v}" for k, v in sorted(deg_dist.items()))
        lines.append(f"- Degree Distribution: {deg_str}")
    lines.append("")

    payload = "\n".join(lines)
    logger.info(
        "LLM payload generated. %d lines, ~%d tokens (est. 4 chars/token).",
        len(lines), len(payload) // 4,
    )
    return payload
