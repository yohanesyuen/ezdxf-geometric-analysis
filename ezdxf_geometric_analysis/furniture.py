"""Furniture/fixture detection for architectural plans.

Classifies block instances as furniture or fixtures based on block name
matching against a keyword taxonomy, with a shape-heuristic fallback
for anonymous/dynamic blocks.

Issue #101.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# Furniture category keywords — matched case-insensitively against block names
# Note: order matters — more specific categories should precede generic ones
# to avoid substring false-positives (e.g. "SEAT" matching inside "SOFA-3SEAT").
FURNITURE_TAXONOMY = {
    "sofa": ["SOFA", "COUCH", "LOVESEAT", "SETTEE"],
    "chair": ["CHAIR", "SEAT", "STOOL", "ARMCHAIR"],
    "desk": ["DESK", "WORKSTATION", "WORKSTN"],
    "table": ["TABLE", "DINING", "CONFERENCE", "CONF"],
    "bed": ["BED", "BUNK", "COT", "MATTRESS"],
    "wardrobe": ["WARDROBE", "CLOSET", "CABINET", "ARMOIRE"],
    "sink": ["SINK", "BASIN", "LAVATORY", "LAV"],
    "toilet": ["TOILET", "WC", "WATER CLOSET", "URINAL"],
    "bath": ["BATH", "BATHTUB", "TUB", "SHOWER"],
    "appliance": ["FRIDGE", "REFRIGERATOR", "OVEN", "STOVE", "WASHER", "DRYER", "DISHWASHER", "MICROWAVE"],
    "storage": ["SHELF", "BOOKCASE", "BOOKSHELF", "SHELVING", "RACK"],
    "fixture": ["LIGHT", "LAMP", "FAN", "AC", "HVAC", "RADIATOR"],
}


def classify_block_name(block_name: str) -> str | None:
    """Classify a block name into a furniture category, or None if unrecognized."""
    upper = block_name.upper()
    # Skip anonymous/dynamic blocks
    if upper.startswith("*") or upper.startswith("A$C"):
        return None
    for category, keywords in FURNITURE_TAXONOMY.items():
        for kw in keywords:
            if kw in upper:
                return category
    return None


def detect_furniture(
    block_instances: list[dict],
    drawing_type: str = "generic",
) -> list[dict]:
    """
    Detect furniture/fixtures from block instance data.

    Args:
        block_instances: List of dicts from entity.get_block_instances
            Each: {block_name, handle, layer, insertion_point, scale, rotation}
        drawing_type: Drawing discipline from detect_drawing_type.

    Returns:
        List of detected furniture items:
            {category, block_name, handle, layer, insertion_point, method}
        method is "block_name" for keyword matches.
    """
    if drawing_type not in ("architectural", "generic"):
        return []

    results = []
    for inst in block_instances:
        name = inst.get("block_name", "")
        category = classify_block_name(name)
        if category:
            results.append({
                "category": category,
                "block_name": name,
                "handle": inst.get("handle", ""),
                "layer": inst.get("layer", ""),
                "insertion_point": inst.get("insertion_point"),
                "method": "block_name",
            })

    logger.info("Detected %d furniture/fixture items from %d block instances.",
                len(results), len(block_instances))
    return results


def shape_heuristic_classify(
    bbox_width: float, bbox_height: float
) -> str | None:
    """
    Fallback shape-based classification for anonymous blocks.

    Uses bounding box aspect ratio and size to guess category.
    Dimensions in drawing units (typically mm).
    """
    area = bbox_width * bbox_height
    aspect = max(bbox_width, bbox_height) / max(min(bbox_width, bbox_height), 1e-6)

    # Typical furniture sizes in mm at 1:1 scale
    if 300_000 < area < 3_000_000:  # ~600x500 to ~2000x1500
        if aspect < 1.5:
            return "table"  # roughly square
        elif aspect < 3:
            return "desk"   # rectangular
        else:
            return "sofa"   # elongated
    elif 100_000 < area < 600_000:  # ~300x300 to ~800x800
        if aspect < 1.5:
            return "chair"
        else:
            return "sink"

    return None
