import time

from ezdxf_geometric_analysis.topology import cluster_nodes, detect_loops


def _layers(segments):
    return {"layer0": {"lines": [{"start": list(s), "end": list(e)} for s, e in segments]}}


# ── cluster_nodes ─────────────────────────────────────────────────────────────

def test_merges_endpoints_within_tolerance():
    # Two segments whose adjacent endpoints are 0.3 apart, tolerance 1.0 — should merge to one node.
    layers = _layers([
        ((0, 0), (10, 0)),
        ((10.3, 0.1), (20, 0)),
    ])
    nodes, _ = cluster_nodes(layers, tolerance=1.0)
    assert len(nodes) == 3  # (0,0), merged ~(10,0), (20,0)


def test_keeps_distinct_points_beyond_tolerance():
    layers = _layers([
        ((0, 0), (10, 0)),
        ((15, 0), (20, 0)),  # 5 units from (10,0) — well beyond tolerance
    ])
    nodes, _ = cluster_nodes(layers, tolerance=1.0)
    assert len(nodes) == 4


def test_merges_points_across_adjacent_grid_cell_boundary():
    # tolerance=0.2 -> grid cell size 0.2. P1=(0.15,0) sits in cell 0; P2=(0.25,0) sits in
    # cell 1 — adjacent but distinct cells. They're 0.1 apart (within tolerance), so the
    # 3x3 neighbor scan must still find and merge them. Each point's own segment partner
    # is far away so it can't merge with anything but its intended match.
    layers = _layers([
        ((0.15, 0), (10, 0)),
        ((0.25, 0), (20, 0)),
    ])
    nodes, _ = cluster_nodes(layers, tolerance=0.2)
    assert len(nodes) == 3  # merged (0.15,0)+(0.25,0), plus (10,0), plus (20,0)


def test_handles_large_distinct_point_set_without_quadratic_blowup():
    # 60x60 grid of distinct segments (7200 endpoints), spaced well beyond tolerance.
    # A naive O(n^2) linear scan scales quadratically; the spatial-hash version
    # should finish in well under a second. Generous bound below to avoid CI flakiness.
    segments = [
        ((i * 10, j * 10), (i * 10 + 2, j * 10))
        for i in range(60)
        for j in range(60)
    ]
    layers = _layers(segments)

    start = time.monotonic()
    nodes, _ = cluster_nodes(layers, tolerance=1.0)
    elapsed = time.monotonic() - start

    assert len(nodes) == 7200  # all endpoints distinct
    assert elapsed < 5.0, f"cluster_nodes took {elapsed:.2f}s — spatial hashing regression?"


# ── detect_loops ─────────────────────────────────────────────────────────────

def _square_adjacency():
    adjacency = [[] for _ in range(4)]

    def connect(a, b):
        adjacency[a].append(b)
        adjacency[b].append(a)

    connect(0, 1)
    connect(1, 2)
    connect(2, 3)
    connect(3, 0)
    return adjacency


def test_finds_single_square_loop():
    loops, truncated = detect_loops(_square_adjacency(), max_loop_length=20)
    assert len(loops) == 1
    assert truncated is False


def test_tree_has_no_loops():
    adjacency = [[] for _ in range(4)]
    adjacency[0].append(1); adjacency[1].append(0)
    adjacency[1].append(2); adjacency[2].append(1)
    adjacency[1].append(3); adjacency[3].append(1)

    loops, truncated = detect_loops(adjacency, max_loop_length=20)
    assert loops == []
    assert truncated is False


def test_truncates_at_max_total_loops():
    # K5 (5 mutually-connected nodes) has 37 simple cycles — far more than the cap below.
    adjacency = [[] for _ in range(5)]
    for i in range(5):
        for j in range(i + 1, 5):
            adjacency[i].append(j)
            adjacency[j].append(i)

    loops, truncated = detect_loops(adjacency, max_loop_length=5, max_total_loops=2)
    assert len(loops) == 2
    assert truncated is True
