"""Pass 2 — node clustering, adjacency graph construction, and closed-loop detection."""

from __future__ import annotations

import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


def cluster_nodes(
    layers: dict, tolerance: float = 1.0
) -> tuple[list[list[float]], dict[tuple[float, float], int]]:
    """
    Group line endpoints that fall within *tolerance* into shared topological
    nodes (joints).

    Uses a simple greedy clustering: for each endpoint, find the first existing
    node within tolerance and merge into it; otherwise create a new node.

    Returns:
        nodes    — list of [x, y] cluster centroids, indexed by node_id.
        pt_to_id — mapping from rounded (x, y) tuples to their node_id (for
                   fast lookup when building the adjacency graph).
    """
    logger.info(
        "Clustering line endpoints into topological nodes (tolerance=%.2f)...",
        tolerance,
    )

    nodes: list[list[float]] = []          # node_id → [sum_x, sum_y, count]
    pt_to_id: dict[tuple[float, float], int] = {}

    # Bucket node centroids into a grid keyed by tolerance-sized cells (issue #58), so a new
    # point only checks the 3x3 neighboring cells instead of every existing node.
    cell_size = max(tolerance, 1e-6)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)

    def _cell_of(x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / cell_size), math.floor(y / cell_size))

    def _find_nearby(x: float, y: float, home_cx: int, home_cy: int) -> int | None:
        """Read-only scan of the 3x3 neighboring cells; no mutation, so the grid's
        per-cell lists are never touched while being iterated."""
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for nid in grid.get((home_cx + dx, home_cy + dy), ()):
                    nx, ny, cnt = nodes[nid]
                    cx, cy = nx / cnt, ny / cnt
                    if math.hypot(x - cx, y - cy) <= tolerance:
                        return nid
        return None

    def _get_or_create(x: float, y: float) -> int:
        """Return existing node_id if (x,y) is within tolerance, else create new."""
        # Check exact hit first (fast path for repeated identical points)
        key = (round(x, 2), round(y, 2))
        if key in pt_to_id:
            return pt_to_id[key]

        home_cx, home_cy = _cell_of(x, y)
        matched = _find_nearby(x, y, home_cx, home_cy)
        if matched is not None:
            nx, ny, cnt = nodes[matched]
            old_cell = _cell_of(nx / cnt, ny / cnt)
            nodes[matched][0] += x
            nodes[matched][1] += y
            nodes[matched][2] += 1
            pt_to_id[key] = matched

            new_cell = _cell_of(nodes[matched][0] / nodes[matched][2], nodes[matched][1] / nodes[matched][2])
            if new_cell != old_cell:
                grid[old_cell].remove(matched)
                grid[new_cell].append(matched)
            return matched

        # Create new node
        nid = len(nodes)
        nodes.append([x, y, 1])
        pt_to_id[key] = nid
        grid[(home_cx, home_cy)].append(nid)
        return nid

    # Walk all line segments across all layers
    for layer_name, layer_data in layers.items():
        for seg in layer_data.get("lines", []):
            sx, sy = seg["start"]
            ex, ey = seg["end"]
            _get_or_create(sx, sy)
            _get_or_create(ex, ey)

    # Compute final centroids
    centroids = [
        [round(n[0] / n[2], 2), round(n[1] / n[2], 2)] for n in nodes
    ]

    logger.info("Node clustering complete. %d unique nodes found.", len(centroids))
    return centroids, pt_to_id


def build_adjacency_graph(
    layers: dict, pt_to_id: dict[tuple[float, float], int], num_nodes: int
) -> tuple[list[list[int]], list[dict]]:
    """
    Build an adjacency graph connecting nodes via line segments.

    Returns:
        adjacency — adjacency list: adjacency[node_id] = [connected_node_ids...]
        edges     — list of edge dicts {from, to, layer, segment_index} for
                    later loop attribution.
    """
    logger.info("Building adjacency graph...")
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    edges: list[dict] = []

    for layer_name, layer_data in layers.items():
        for seg_idx, seg in enumerate(layer_data.get("lines", [])):
            sk = (round(seg["start"][0], 2), round(seg["start"][1], 2))
            ek = (round(seg["end"][0], 2), round(seg["end"][1], 2))

            nid_start = pt_to_id.get(sk)
            nid_end = pt_to_id.get(ek)

            if nid_start is None or nid_end is None:
                continue
            if nid_start == nid_end:
                continue  # degenerate zero-length segment

            # Avoid duplicate edges in the adjacency list
            if nid_end not in adjacency[nid_start]:
                adjacency[nid_start].append(nid_end)
            if nid_start not in adjacency[nid_end]:
                adjacency[nid_end].append(nid_start)

            edges.append({
                "from": nid_start,
                "to": nid_end,
                "layer": layer_name,
                "segment_index": seg_idx,
            })

    logger.info(
        "Adjacency graph built. %d nodes, %d edges.", num_nodes, len(edges)
    )
    return adjacency, edges


def detect_loops(
    adjacency: list[list[int]], max_loop_length: int = 20, max_total_loops: int = 2000
) -> tuple[list[list[int]], bool]:
    """
    Detect closed boundary loops in the adjacency graph using DFS cycle finding.

    Only keeps loops with length <= max_loop_length to focus on room-sized
    boundaries and avoid degenerate mega-cycles. Dense rectilinear grids (e.g.
    repeated wall/grid patterns) can still blow up combinatorially even with
    that bound, so detection also bails out once max_total_loops is reached.

    Returns (loops, truncated) — loops is a list of node_id lists forming each
    cycle; truncated is True if detection stopped early due to max_total_loops.
    """
    logger.info(
        "Detecting closed loops (max length=%d, max total=%d)...",
        max_loop_length, max_total_loops,
    )

    num_nodes = len(adjacency)
    loops: list[list[int]] = []
    # Track unique loops by their sorted node set to avoid duplicates
    seen_loops: set[frozenset[int]] = set()
    truncated = False

    def _dfs(start: int) -> None:
        """Find all simple cycles passing through *start* using bounded DFS."""
        nonlocal truncated
        # Stack entries: (current_node, path_so_far, visited_in_path)
        stack: list[tuple[int, list[int], set[int]]] = [
            (start, [start], {start})
        ]

        while stack:
            node, path, visited = stack.pop()

            if len(path) > max_loop_length:
                continue

            for neighbor in adjacency[node]:
                if neighbor == start and len(path) >= 3:
                    # Found a cycle back to start
                    loop_key = frozenset(path)
                    if loop_key not in seen_loops:
                        seen_loops.add(loop_key)
                        loops.append(list(path))
                        if len(loops) >= max_total_loops:
                            truncated = True
                            return
                elif neighbor not in visited and neighbor > start:
                    # Only explore neighbors with id > start to avoid
                    # finding the same cycle from multiple starting nodes
                    new_visited = visited | {neighbor}
                    stack.append((neighbor, path + [neighbor], new_visited))

    for node_id in range(num_nodes):
        if truncated:
            break
        if adjacency[node_id]:  # only start from connected nodes
            _dfs(node_id)

    logger.info("Loop detection complete. %d closed loops found%s.",
                len(loops), " (truncated)" if truncated else "")
    return loops, truncated


def compute_loop_metadata(
    loops: list[list[int]], nodes: list[list[float]]
) -> list[dict]:
    """
    Compute metadata for each detected loop: centroid, approximate area,
    and bounding box.

    Uses the shoelace formula for area calculation.
    """
    logger.info("Computing loop metadata (area, centroid, bbox)...")
    loop_data = []

    for loop_nodes in loops:
        # Get coordinates for the polygon vertices
        coords = [nodes[nid] for nid in loop_nodes]
        if len(coords) < 3:
            continue

        # Shoelace formula for area
        n = len(coords)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
        area = abs(area) / 2.0

        # Centroid
        cx = sum(c[0] for c in coords) / n
        cy = sum(c[1] for c in coords) / n

        # Bounding box
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        bbox = [min(xs), min(ys), max(xs), max(ys)]

        loop_data.append({
            "node_ids": loop_nodes,
            "num_segments": len(loop_nodes),
            "area": round(area, 2),
            "centroid": [round(cx, 2), round(cy, 2)],
            "bbox": [round(v, 2) for v in bbox],
        })

    # Sort by area descending (largest rooms first)
    loop_data.sort(key=lambda d: d["area"], reverse=True)

    logger.info(
        "Loop metadata complete. %d valid loops with area > 0.", len(loop_data)
    )
    return loop_data


def topological_analysis(
    layers: dict, tolerance: float = 1.0, max_loop_length: int = 20, max_total_loops: int = 2000
) -> dict:
    """
    Run the full Pass 2 topological analysis pipeline.

    Steps:
      1. Cluster endpoints into nodes
      2. Build adjacency graph
      3. Detect closed loops
      4. Compute loop metadata (area, centroid, bbox)

    Returns a dict with nodes, edges summary, and loops ready for JSON output.
    """
    logger.info("═══ Pass 2: Topological Analysis ═══")

    nodes, pt_to_id = cluster_nodes(layers, tolerance=tolerance)
    adjacency, edges = build_adjacency_graph(layers, pt_to_id, len(nodes))
    loops, loops_truncated = detect_loops(
        adjacency, max_loop_length=max_loop_length, max_total_loops=max_total_loops
    )
    loop_metadata = compute_loop_metadata(loops, nodes)

    # Compute node degree distribution for summary
    degrees = [len(adj) for adj in adjacency]
    degree_dist = defaultdict(int)
    for d in degrees:
        degree_dist[d] += 1

    logger.info(
        "Topological summary: %d nodes, %d edges, %d loops. "
        "Degree distribution: %s",
        len(nodes), len(edges), len(loop_metadata),
        dict(sorted(degree_dist.items())),
    )

    return {
        "parameters": {
            "clustering_tolerance": tolerance,
            "max_loop_length": max_loop_length,
            "max_total_loops": max_total_loops,
        },
        "nodes": {
            "count": len(nodes),
            "coordinates": nodes,
            "degree_distribution": dict(sorted(degree_dist.items())),
        },
        "edges": {
            "count": len(edges),
        },
        "loops": loop_metadata,
        "loops_truncated": loops_truncated,
    }
