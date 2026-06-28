"""Orchestrator — runs all five analysis passes and assembles the final payload."""

from __future__ import annotations

import logging
import os

from .extraction import (
    compute_bounds,
    compute_normalization,
    detect_drawing_type,
    extract_entities,
    load_dxf,
)
from .payload import _assign_entity_ids, generate_llm_payload
from .semantics import semantic_grouping
from .spatial import spatial_descriptors
from .topology import topological_analysis

logger = logging.getLogger(__name__)


def analyze_dxf(file_path: str, **kwargs) -> tuple[dict, str]:
    """
    Run the full geometric analysis pipeline on a DXF file.

    Pass 1: Extract and Normalize
    Pass 2: Topological Analysis
    Pass 3: Spatial Descriptors
    Pass 4: Semantic Grouping
    Pass 5: LLM Payload Generation

    `scale_factor` and `rounding` kwargs override the values auto-detected
    from the drawing's extents and paper size (see `compute_normalization`).

    Returns (analysis_dict, markdown_payload).
    """
    # -- Pass 1 --------------------------------------------------------------
    logger.info("═══ Pass 1: Extract and Normalize ═══")

    doc = load_dxf(file_path)
    msp = doc.modelspace()

    min_x, min_y, max_x, max_y = compute_bounds(msp)
    norm = compute_normalization(min_x, min_y, max_x, max_y)

    scale_factor = kwargs.get("scale_factor", norm["scale_factor"])
    rounding = kwargs.get("rounding", norm["rounding"])
    if scale_factor != norm["scale_factor"]:
        norm["width"] = norm["width"] * norm["scale_factor"] / scale_factor
        norm["height"] = norm["height"] * norm["scale_factor"] / scale_factor
    norm["scale_factor"] = scale_factor
    norm["rounding"] = rounding
    layers = extract_entities(
        msp, norm["center_x"], norm["center_y"],
        scale=scale_factor, rounding=rounding,
    )

    # Detect drawing type from layer names
    drawing_type = detect_drawing_type(layers)
    logger.info("Drawing type detected: %s", drawing_type)

    # -- Pass 2 ----------------------------------------------------------------
    topology = topological_analysis(layers, tolerance=1.0, max_loop_length=20)

    # Assign entity IDs now (Pass 4's room assignment keys entities by id,
    # and the payload in Pass 5 needs them too).
    _assign_entity_ids(layers, topology)

    # -- Pass 3 ----------------------------------------------------------------
    spatial = spatial_descriptors(
        layers, topology, topology["nodes"]["coordinates"]
    )

    # -- Pass 4 (adaptive radius) ------------------------------------------------
    semantics = semantic_grouping(layers, proximity_radius=None)

    # -- Pass 5 ----------------------------------------------------------------

    analysis = {
        "metadata": {
            "filename": os.path.basename(file_path),
            "drawing_type": drawing_type,
            "global_dimensions": {
                "width": round(norm["width"], rounding),
                "height": round(norm["height"], rounding),
                "units": "mm (paper-space)" if scale_factor > 1 else "drawing units",
            },
            "transform": {
                "type": "translate_to_origin" + (f" + scale÷{scale_factor:.0f}" if scale_factor > 1 else ""),
                "offset_x": round(-norm["center_x"], 2),
                "offset_y": round(-norm["center_y"], 2),
                "scale_factor": scale_factor,
                "rounding_precision": rounding,
                "description": (
                    f"Coordinates translated to center at (0,0)"
                    + (f" and divided by {scale_factor:.0f} to paper-space mm" if scale_factor > 1 else "")
                    + f". Rounded to {rounding} decimal places."
                ),
            },
        },
        "layers": dict(layers),
        "topology": topology,
        "spatial": spatial,
        "semantics": semantics,
    }

    # Add paper info to metadata if detected
    if norm.get("paper_info"):
        analysis["metadata"]["paper_info"] = norm["paper_info"]

    markdown_payload = generate_llm_payload(
        analysis, topology, spatial, semantics, drawing_type=drawing_type
    )

    logger.info(
        "Full analysis complete. %d layers, %d nodes, %d loops.",
        len(layers), topology["nodes"]["count"], len(topology["loops"]),
    )
    return analysis, markdown_payload
