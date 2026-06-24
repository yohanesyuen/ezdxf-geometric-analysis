# ezdxf-geometric-analysis

A five-pass geometric analysis pipeline for DXF drawings built on [ezdxf](https://ezdxf.readthedocs.io/). It turns a DXF file into structured JSON and an LLM-ready Markdown payload describing layers, topology, spatial layout, and semantics.

Extracted from the geometric analysis pipeline used by [AcadMediator](https://github.com/yohanesyuen/acad-mediator) — decoupled here as a standalone, ezdxf-only library with no AutoCAD or mediator dependencies.

## Pipeline

| Pass | Module | Purpose |
|------|--------|---------|
| 1 | `extraction.py` | Load DXF, normalize coordinates, extract entities by type |
| 2 | `topology.py` | Cluster endpoints into nodes, build adjacency graph, detect closed loops |
| 3 | `spatial.py` | Per-layer bounding boxes, alignment summaries, loop containment |
| 4 | `semantics.py` | Layer statistics, text-to-geometry proximity anchoring, room assignment |
| 5 | `payload.py` | Hybrid Markdown generation with embedded entity IDs |

`furniture.py` adds an optional furniture/fixture classifier for architectural block instances (keyword + shape-heuristic matching).

## Install

```bash
pip install -e .
```

## Usage

```bash
analyze-dxf path/to/drawing.dxf
# writes path/to/drawing.json and path/to/drawing.md
```

Or from Python:

```python
from ezdxf_geometric_analysis import analyze_dxf

analysis, markdown = analyze_dxf("drawing.dxf")
```

See `notebooks/analysis.ipynb` for an interactive starting point.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
