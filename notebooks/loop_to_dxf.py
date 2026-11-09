"""Build a standalone DXF document for a detected loop (from analyze_dxf's topology pass).

Usage:
    from ezdxf_geometric_analysis import analyze_dxf
    from loop_to_dxf import loop_to_dxf

    analysis, _ = analyze_dxf("drawing.dxf")
    doc = loop_to_dxf(analysis, "LOOP_003")
    doc.saveas("loop_003.dxf")
"""

import ezdxf


def loop_to_dxf(analysis: dict, loop_id: str, context: bool = True) -> ezdxf.document.Drawing:
    """
    Build a new DXF document containing the boundary of one detected loop.

    `loop_id` is the `LOOP_NNN` id assigned in payload._assign_entity_ids.
    If `context` is True, the original line/circle/text geometry is also
    drawn on its original layers so the loop can be seen in place.

    Returns the ezdxf Drawing — call `.saveas(path)` on it to write a file.
    """
    loops = analysis["topology"]["loops"]
    loop = next((l for l in loops if l.get("id") == loop_id), None)
    if loop is None:
        raise ValueError(f"No loop with id {loop_id!r} in this analysis.")

    coords = analysis["topology"]["nodes"]["coordinates"]
    points = [coords[nid] for nid in loop["node_ids"]]

    doc = ezdxf.new()
    msp = doc.modelspace()

    if context:
        for layer_name, layer_data in analysis["layers"].items():
            doc.layers.new(layer_name) if layer_name not in doc.layers else None
            for seg in layer_data.get("lines", []):
                msp.add_line(seg["start"], seg["end"], dxfattribs={"layer": layer_name})
            for ca in layer_data.get("circles_arcs", []):
                msp.add_circle(ca["center"], ca["radius"], dxfattribs={"layer": layer_name})
            for t in layer_data.get("text_annotations", []):
                msp.add_text(t["text"], dxfattribs={"layer": layer_name, "insert": t["position"]})

    doc.layers.new("LOOP_HIGHLIGHT", dxfattribs={"color": 1})  # red
    msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "LOOP_HIGHLIGHT"})

    return doc
