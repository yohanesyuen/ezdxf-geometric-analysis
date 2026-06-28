"""Regression tests for filed GitHub issues.

Issues #3, #4, #5, #6 are open (unfixed) — these tests assert the *correct*
behavior and are marked xfail(strict=True) so they:
  - keep the suite green today (the bug is documented, not a build breaker)
  - fail loudly the moment the underlying bug is actually fixed, signalling
    that the xfail marker should be dropped and the issue closed.

Issue #1 already appears fixed on this branch (see ezdxf_geometric_analysis/
extraction.py:compute_bounds and pipeline.py's pass ordering) — its tests are
plain regression tests guarding the fix against being lost.
"""

from __future__ import annotations

import math
from collections import defaultdict
from types import SimpleNamespace

import ezdxf.bbox
import pytest

from ezdxf_geometric_analysis.extraction import compute_bounds
from ezdxf_geometric_analysis.payload import _assign_entity_ids
from ezdxf_geometric_analysis.semantics import assign_rooms
from ezdxf_geometric_analysis.topology import (
    build_adjacency_graph,
    cluster_nodes,
    compute_loop_metadata,
    detect_loops,
    topological_analysis,
)


def _layers(segments):
    return {"layer0": {"lines": [{"start": list(s), "end": list(e)} for s, e in segments]}}


# ── Issue #1 — LEADER/bbox/room-assignment port from acad-mediator@30f8550 ──
# (already fixed on this branch; these guard against the fix regressing)

class _FakeEntity:
    def __init__(self, dxftype):
        self._dxftype = dxftype
        self.dxf = SimpleNamespace(handle="DEADBEEF")

    def dxftype(self):
        return self._dxftype


def test_compute_bounds_skips_malformed_entity_without_aborting(monkeypatch):
    # A degenerate LEADER (fewer than 2 vertices) makes ezdxf.bbox.extents()
    # raise; that must not abort the bbox pass for the rest of modelspace.
    bad = _FakeEntity("LEADER")
    good = _FakeEntity("LINE")

    def fake_extents(entities):
        if entities[0] is bad:
            raise ValueError("LEADER has fewer than 2 vertices")
        return ezdxf.bbox.BoundingBox([(0, 0, 0), (10, 10, 0)])

    monkeypatch.setattr(ezdxf.bbox, "extents", fake_extents)

    min_x, min_y, max_x, max_y = compute_bounds([bad, good])
    assert (min_x, min_y, max_x, max_y) == (0.0, 0.0, 10.0, 10.0)


def test_compute_bounds_accepts_flat_2d_drawing(monkeypatch):
    # Every purely-2D drawing has z constant 0, so its bbox has size 0 along
    # z — `is_empty` reports True for that (size 0 in *any* axis), even
    # though the box clearly has x/y data. compute_bounds must check
    # `has_data`, not `is_empty`, or every flat drawing would raise as empty.
    flat_bbox = ezdxf.bbox.BoundingBox([(0, 0, 0), (10, 10, 0)])
    assert flat_bbox.has_data is True
    assert flat_bbox.is_empty is True  # confirms the trap is_empty would set

    monkeypatch.setattr(ezdxf.bbox, "extents", lambda entities: flat_bbox)

    min_x, min_y, max_x, max_y = compute_bounds([_FakeEntity("LINE")])
    assert (min_x, min_y, max_x, max_y) == (0.0, 0.0, 10.0, 10.0)


def test_assign_rooms_receives_populated_ids_when_called_after_id_assignment():
    # pipeline.py must call _assign_entity_ids() before Pass 4's
    # assign_rooms(), or every entity falls back to the "?" placeholder id
    # and room-to-entity attribution is useless.
    layers = {
        "A-ROOM-NAME": {
            "lines": [], "circles_arcs": [],
            "text_annotations": [{"text": "OFFICE 101", "position": [0, 0]}],
        },
        "A-WALL": {
            "lines": [{"start": [0, 0], "end": [10, 0]}],
            "circles_arcs": [], "text_annotations": [],
        },
    }
    topology = topological_analysis(layers, tolerance=1.0)
    _assign_entity_ids(layers, topology)  # Pass 2 -> id assignment -> Pass 4

    rooms = assign_rooms(layers)["rooms"]
    wall_entity_ids = {e["id"] for e in rooms[0]["entities"] if e["type"] == "line"}
    assert wall_entity_ids == {"L_001"}


# ── Issue #3 — detect_loops dedups by node set, not edge sequence ──────────────

@pytest.mark.xfail(
    reason="issue #3: detect_loops dedups cycles via frozenset(path) (node "
    "set), so distinct Hamiltonian cycles on the same node set collapse "
    "into one",
    strict=True,
)
def test_distinct_cycles_sharing_the_same_node_set_are_not_collapsed():
    # K4 (4 mutually-connected nodes) has 3 distinct Hamiltonian 4-cycles,
    # all sharing the node set {0,1,2,3}: 0-1-2-3, 0-1-3-2, 0-2-1-3.
    adjacency = [[] for _ in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            adjacency[i].append(j)
            adjacency[j].append(i)

    loops, _truncated = detect_loops(adjacency, max_loop_length=4, max_total_loops=2000)
    four_cycles = [loop for loop in loops if len(loop) == 4]
    assert len(four_cycles) == 3


# ── Issue #4 — cluster_nodes centroid drift (chaining) ─────────────────────────

@pytest.mark.xfail(
    reason="issue #4: cluster_nodes compares each new point against a "
    "drifting running centroid, letting a chain of merges drift endpoints "
    "beyond the configured tolerance",
    strict=True,
)
def test_cluster_nodes_never_chains_points_beyond_tolerance():
    spacing = 0.3
    tolerance = 1.0
    points = [(i * spacing, 0.0) for i in range(60)]
    layers = _layers([(points[i], points[i + 1]) for i in range(len(points) - 1)])

    _nodes, pt_to_id = cluster_nodes(layers, tolerance=tolerance)

    groups: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for p in points:
        key = (round(p[0], 2), round(p[1], 2))
        groups[pt_to_id[key]].append(p)

    for pts in groups.values():
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dist = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                assert dist <= tolerance, (
                    f"points {pts[i]} and {pts[j]} share a node but are "
                    f"{dist:.2f} apart (tolerance={tolerance})"
                )


# ── Issue #5 — sub-centimeter tolerance clamped by hardcoded round(x, 2) ──────

@pytest.mark.xfail(
    reason="issue #5: cluster_nodes' exact-hit fast path hardcodes "
    "round(x, 2), ignoring tolerance below ~0.01",
    strict=True,
)
def test_subcentimeter_tolerance_is_not_clamped_to_centimeter_precision():
    tolerance = 0.005
    layers = _layers([
        ((0.0039, 0.0), (10.0, 0.0)),
        ((-0.0039, 0.0), (20.0, 0.0)),
    ])
    nodes, _pt_to_id = cluster_nodes(layers, tolerance=tolerance)
    assert len(nodes) == 4


# ── Issue #6 — edges' layer/segment_index data is computed but never used ────

@pytest.mark.xfail(
    reason="issue #6: compute_loop_metadata never receives `edges`, so "
    "loops carry no boundary-layer attribution despite "
    "build_adjacency_graph already computing it per edge",
    strict=True,
)
def test_loop_metadata_reports_boundary_layers_from_edges():
    layers = _layers([
        ((0, 0), (10, 0)),
        ((10, 0), (10, 10)),
        ((10, 10), (0, 10)),
        ((0, 10), (0, 0)),
    ])
    nodes, pt_to_id = cluster_nodes(layers, tolerance=1.0)
    adjacency, edges = build_adjacency_graph(layers, pt_to_id, len(nodes))
    loops, _truncated = detect_loops(adjacency, max_loop_length=20)

    loop_data = compute_loop_metadata(loops, nodes, edges)
    assert loop_data[0]["boundary_layers"] == ["layer0"]
