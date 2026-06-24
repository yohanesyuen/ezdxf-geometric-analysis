"""Tests for defensive handling of missing 'alignment' keys in the semantics/spatial passes."""

from ezdxf_geometric_analysis.semantics import compute_layer_statistics, semantic_grouping
from ezdxf_geometric_analysis.spatial import compute_alignment_summary


def test_no_alignment_key_doesnt_crash():
    """Lines without 'alignment' key should not raise KeyError."""
    layers = {
        "WALLS": {
            "lines": [
                {"start": [0, 0], "end": [100, 0]},      # horizontal
                {"start": [0, 0], "end": [0, 100]},      # vertical
                {"start": [0, 0], "end": [50, 50]},      # diagonal
            ],
            "circles_arcs": [],
            "text_annotations": [],
        }
    }
    # Should not raise
    stats = compute_layer_statistics(layers)
    assert stats["WALLS"]["line_count"] == 3
    assert stats["WALLS"]["dominant_alignment"] == "diagonal"  # all default to diagonal

    align = compute_alignment_summary(layers)
    assert align["WALLS"]["diagonal"] == 3  # all default to diagonal w/o key


def test_with_alignment_key_works():
    """Lines WITH 'alignment' key should work as before."""
    layers = {
        "WALLS": {
            "lines": [
                {"start": [0, 0], "end": [100, 0], "alignment": "horizontal"},
                {"start": [0, 0], "end": [0, 100], "alignment": "vertical"},
                {"start": [0, 0], "end": [50, 50], "alignment": "diagonal"},
            ],
            "circles_arcs": [],
            "text_annotations": [],
        }
    }
    stats = compute_layer_statistics(layers)
    assert stats["WALLS"]["line_count"] == 3
    # No dominant — 1 each
    align = compute_alignment_summary(layers)
    assert align["WALLS"]["horizontal"] == 1
    assert align["WALLS"]["vertical"] == 1
    assert align["WALLS"]["diagonal"] == 1


def test_enrichment_logic():
    """Simulate alignment enrichment applied by a caller before invoking semantic_grouping."""
    geo_data = {
        "layers": {
            "GRID": {
                "lines": [
                    {"start": [0, 0], "end": [1000, 0]},      # horizontal (dy=0)
                    {"start": [0, 0], "end": [0.1, 500]},     # vertical (dx<0.5)
                    {"start": [0, 0], "end": [200, 300]},     # diagonal
                    {"start": [5, 5], "end": [5.3, 5]},       # both dx<0.5 and dy<0.5 → diagonal
                ],
                "circles_arcs": [],
                "text_annotations": [],
            }
        }
    }

    for _layer_data in geo_data.get("layers", {}).values():
        for seg in _layer_data.get("lines", []):
            if "alignment" not in seg:
                dx = abs(seg["end"][0] - seg["start"][0])
                dy = abs(seg["end"][1] - seg["start"][1])
                tol = 0.5
                if dx < tol and dy >= tol:
                    seg["alignment"] = "vertical"
                elif dy < tol and dx >= tol:
                    seg["alignment"] = "horizontal"
                else:
                    seg["alignment"] = "diagonal"

    lines = geo_data["layers"]["GRID"]["lines"]
    assert lines[0]["alignment"] == "horizontal"
    assert lines[1]["alignment"] == "vertical"
    assert lines[2]["alignment"] == "diagonal"
    assert lines[3]["alignment"] == "diagonal"  # both dx and dy < tol

    # Now run through semantics — should not crash
    result = semantic_grouping(geo_data["layers"])
    assert "layer_statistics" in result
    assert result["layer_statistics"]["GRID"]["line_count"] == 4
