"""
Geometric analysis of a DXF file for LLM consumption.

Pass 1 (extraction)  — load, normalize, and extract entities by type.
Pass 2 (topology)    — node clustering, adjacency graph, closed-loop detection.
Pass 3 (spatial)     — per-layer bounding boxes, alignment vectors, containment.
Pass 4 (semantics)   — layer statistics, text-to-geometry proximity anchoring.
Pass 5 (payload)     — hybrid Markdown generation with embedded entity IDs.

`analyze_dxf()` (in `pipeline.py`) runs all five passes and returns
(analysis_dict, markdown_payload). Each pass lives in its own module so a
change to one concern doesn't require reading the whole pipeline.
"""

from .pipeline import analyze_dxf

__all__ = ["analyze_dxf"]
