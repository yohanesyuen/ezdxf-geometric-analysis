"""Tests for ezdxf_geometric_analysis.furniture — furniture/fixture detection."""

from ezdxf_geometric_analysis.furniture import classify_block_name, detect_furniture, shape_heuristic_classify


class TestClassifyBlockName:
    def test_chair(self):
        assert classify_block_name("CHAIR-OFFICE") == "chair"
        assert classify_block_name("A-FURN-SEAT-01") == "chair"

    def test_desk(self):
        assert classify_block_name("DESK-L-SHAPE") == "desk"
        assert classify_block_name("WORKSTATION-1") == "desk"

    def test_table(self):
        assert classify_block_name("TABLE-ROUND") == "table"
        assert classify_block_name("CONFERENCE-6P") == "table"

    def test_sofa(self):
        assert classify_block_name("SOFA-3SEAT") == "sofa"

    def test_bed(self):
        assert classify_block_name("BED-QUEEN") == "bed"

    def test_toilet(self):
        assert classify_block_name("TOILET-STD") == "toilet"
        assert classify_block_name("WC-01") == "toilet"

    def test_sink(self):
        assert classify_block_name("SINK-KITCHEN") == "sink"

    def test_anonymous_blocks_skipped(self):
        assert classify_block_name("*U123") is None
        assert classify_block_name("A$C272C05F5") is None

    def test_unrecognized(self):
        assert classify_block_name("CUSTOM-WIDGET") is None
        assert classify_block_name("X_R377_3BR_3A2") is None


class TestDetectFurniture:
    def test_basic_detection(self):
        instances = [
            {"block_name": "CHAIR-01", "handle": "A1", "layer": "A-FURN", "insertion_point": [100, 200, 0]},
            {"block_name": "DESK-L", "handle": "A2", "layer": "A-FURN", "insertion_point": [300, 400, 0]},
            {"block_name": "WALL-INT", "handle": "A3", "layer": "A-WALL", "insertion_point": [500, 600, 0]},
        ]
        result = detect_furniture(instances, "architectural")
        assert len(result) == 2
        assert result[0]["category"] == "chair"
        assert result[1]["category"] == "desk"

    def test_non_architectural_returns_empty(self):
        instances = [{"block_name": "CHAIR-01", "handle": "A1", "layer": "E-SYM", "insertion_point": [0, 0, 0]}]
        result = detect_furniture(instances, "electrical")
        assert result == []

    def test_generic_drawing_type_allowed(self):
        instances = [{"block_name": "TABLE-ROUND", "handle": "B1", "layer": "0", "insertion_point": [0, 0, 0]}]
        result = detect_furniture(instances, "generic")
        assert len(result) == 1


class TestShapeHeuristic:
    def test_table_like(self):
        assert shape_heuristic_classify(1000, 800) == "table"

    def test_desk_like(self):
        assert shape_heuristic_classify(1500, 600) == "desk"

    def test_chair_like(self):
        assert shape_heuristic_classify(500, 450) == "chair"

    def test_too_small(self):
        assert shape_heuristic_classify(50, 50) is None

    def test_too_large(self):
        assert shape_heuristic_classify(10000, 10000) is None
