"""Pass 1 — load a DXF, normalize coordinates, and extract entities by type."""

from __future__ import annotations

import logging
from collections import defaultdict

import ezdxf
import ezdxf.bbox
import ezdxf.tools.text

logger = logging.getLogger(__name__)


# ── Step 1: Load DXF ──────────────────────────────────────────────────────────


def load_dxf(file_path: str) -> ezdxf.document.Drawing:
    """Load and validate a DXF file, returning the ezdxf Document."""
    logger.info("Loading DXF: %s", file_path)
    try:
        doc = ezdxf.readfile(file_path)
    except IOError as exc:
        raise ValueError(f"Not a DXF file or a generic I/O error: {file_path}") from exc
    except ezdxf.DXFStructureError as exc:
        raise ValueError(f"Invalid or corrupted DXF file: {file_path}") from exc

    logger.info("DXF loaded successfully. Layers: %d", len(doc.layers))
    return doc


# ── Step 2: Compute Global Bounding Box ──────────────────────────────────────


def compute_bounds(msp) -> tuple[float, float, float, float]:
    """
    Compute the global bounding box of all modelspace entities.

    Entities are measured one at a time so a single malformed entity (e.g. a
    LEADER with fewer than 2 vertices, which ezdxf's virtual_entities()
    rejects outright) can't abort the bounding-box pass for the whole file.

    Returns (min_x, min_y, max_x, max_y).
    """
    logger.info("Computing global bounding box...")
    bbox = ezdxf.bbox.BoundingBox()
    skipped = 0

    for entity in msp:
        try:
            entity_bbox = ezdxf.bbox.extents([entity])
        except Exception as exc:
            skipped += 1
            logger.warning(
                "Skipping %s (handle=%s) in bbox computation: %s",
                entity.dxftype(), entity.dxf.handle, exc,
            )
            continue
        if entity_bbox.has_data:
            bbox.extend(entity_bbox)

    if skipped:
        logger.warning(
            "Skipped %d malformed entit%s during bbox computation.",
            skipped, "y" if skipped == 1 else "ies",
        )

    # Note: bbox.is_empty also triggers on a flat (size 0 in one axis) box,
    # which every purely-2D drawing is (z is constant 0) — has_data is the
    # correct "no geometry at all" check here.
    if not bbox.has_data:
        raise ValueError("The DXF file appears to be empty or has no bounding box data.")

    min_x, min_y, _ = bbox.extmin
    max_x, max_y, _ = bbox.extmax
    logger.info(
        "Bounding box: min=(%.2f, %.2f) max=(%.2f, %.2f)",
        min_x, min_y, max_x, max_y,
    )
    return min_x, min_y, max_x, max_y


# ── Step 3: Derive Normalization Transform ────────────────────────────────────

# Standard paper sizes in mm (landscape orientation)
_PAPER_SIZES = {
    "A0": (1189, 841),
    "A1": (841, 594),
    "A2": (594, 420),
    "A3": (420, 297),
    "A4": (297, 210),
}


def detect_paper_size(width: float, height: float) -> dict | None:
    """
    Detect if the drawing extents match a standard paper size at a common scale.

    Common drawing scales: 1:1, 1:5, 1:10, 1:20, 1:25, 1:50, 1:100, 1:200, 1:500

    Returns a dict with paper_size, scale, and confidence if a match is found,
    or None if no reasonable match is detected.
    """
    common_scales = [1, 5, 10, 20, 25, 50, 100, 200, 500]

    # Ensure width >= height for comparison (landscape)
    w, h = max(width, height), min(width, height)

    best_match = None
    best_error = float("inf")

    for paper_name, (pw, ph) in _PAPER_SIZES.items():
        for scale in common_scales:
            # Drawing extent at this scale would be paper_size * scale
            expected_w = pw * scale
            expected_h = ph * scale

            # Allow 15% tolerance for title blocks, margins, bleed
            error_w = abs(w - expected_w) / expected_w
            error_h = abs(h - expected_h) / expected_h
            combined_error = (error_w + error_h) / 2

            if combined_error < best_error and combined_error < 0.15:
                best_error = combined_error
                best_match = {
                    "paper_size": paper_name,
                    "scale": scale,
                    "expected_extent": [expected_w, expected_h],
                    "fit_error": round(combined_error * 100, 1),
                }

    return best_match


def detect_drawing_type(layers: dict) -> str:
    """
    Infer drawing discipline from layer name prefixes.

    Returns one of: 'electrical', 'architectural', 'mechanical',
    'structural', 'generic'.
    """
    prefix_counts = defaultdict(int)
    total_entities = 0

    for layer_name, layer_data in layers.items():
        count = (
            len(layer_data.get("lines", []))
            + len(layer_data.get("circles_arcs", []))
            + len(layer_data.get("text_annotations", []))
        )
        total_entities += count

        # Detect prefix from layer name (common conventions)
        upper = layer_name.upper()
        if upper.startswith("E-") or upper.startswith("E_") or "ELEC" in upper:
            prefix_counts["electrical"] += count
        elif upper.startswith("A-") or upper.startswith("A_") or "ARCH" in upper:
            prefix_counts["architectural"] += count
        elif upper.startswith("M-") or upper.startswith("M_") or "MECH" in upper:
            prefix_counts["mechanical"] += count
        elif upper.startswith("S-") or upper.startswith("S_") or "STRUCT" in upper:
            prefix_counts["structural"] += count

    if not prefix_counts or total_entities == 0:
        return "generic"

    dominant = max(prefix_counts, key=prefix_counts.get)
    ratio = prefix_counts[dominant] / total_entities

    # Need at least 30% of entities on typed layers to declare a type
    if ratio >= 0.3:
        return dominant
    return "generic"


def compute_normalization(
    min_x: float, min_y: float, max_x: float, max_y: float
) -> dict:
    """
    Compute normalization parameters including:
    - Translate to origin (center of bounding box becomes 0,0)
    - Paper size detection and scale inference
    - Coordinate scaling when extents are too large

    If extents match a known paper size at a standard scale, coordinates are
    divided by that scale so the output is in paper-space mm. This keeps
    coordinate values compact (hundreds instead of tens of thousands), which
    saves LLM tokens and improves readability.

    Scaling heuristic:
    - If max extent > 5000 units AND a paper+scale match is found → apply scale
    - Post-scaling, if fractional parts are negligible (< 0.3 on average),
      round to whole numbers (0 decimal places)
    - Otherwise round to 1 decimal place

    Returns a dict with center_x, center_y, width, height, scale_factor,
    rounding_precision, and optional paper_info.
    """
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y

    paper_info = detect_paper_size(width, height)

    # Determine if we should scale down
    scale_factor = 1.0
    max_extent = max(width, height)

    if paper_info and max_extent > 5000:
        scale_factor = paper_info["scale"]
        scaled_width = width / scale_factor
        scaled_height = height / scale_factor
        logger.info(
            "Paper size detected: %s at 1:%d (fit error: %.1f%%). "
            "Extents %.0f x %.0f → scaling to %.1f x %.1f (paper-space mm).",
            paper_info["paper_size"], paper_info["scale"],
            paper_info["fit_error"],
            width, height, scaled_width, scaled_height,
        )
    elif max_extent > 50000 and not paper_info:
        # No paper match but coordinates are enormous — scale to bring into
        # a reasonable range (target max extent ~1000)
        scale_factor = max_extent / 1000.0
        logger.info(
            "Extents very large (%.0f x %.0f) with no paper match. "
            "Applying heuristic scale factor %.1f to bring into ~1000-unit range.",
            width, height, scale_factor,
        )
    else:
        if paper_info:
            logger.info(
                "Paper size detected: %s at 1:%d (fit error: %.1f%%), "
                "but extents (%.0f x %.0f) are manageable — no scaling applied.",
                paper_info["paper_size"], paper_info["scale"],
                paper_info["fit_error"], width, height,
            )
        else:
            logger.info(
                "No standard paper size detected for extents %.2f x %.2f. "
                "Coordinates treated as raw drawing units (no scaling).",
                width, height,
            )

    # Determine rounding precision: sample some coordinates after transform
    # and check if fractional parts are negligible
    rounding = 1  # default: 1 decimal place
    if scale_factor > 1:
        # Test a few sample points
        samples = [
            (min_x - center_x) / scale_factor,
            (max_x - center_x) / scale_factor,
            (min_y - center_y) / scale_factor,
            (max_y - center_y) / scale_factor,
        ]
        avg_frac = sum(abs(s - round(s)) for s in samples) / len(samples)
        if avg_frac < 0.3:
            rounding = 0
            logger.info(
                "Post-scaling coordinates have avg fractional part %.3f — "
                "rounding to whole numbers.", avg_frac,
            )
        else:
            logger.info(
                "Post-scaling coordinates have avg fractional part %.3f — "
                "keeping 1 decimal place.", avg_frac,
            )

    logger.info(
        "Normalization: translate (%.2f, %.2f), scale ÷%.1f, round to %d dp. "
        "Final extents: %.1f x %.1f",
        -center_x, -center_y, scale_factor, rounding,
        width / scale_factor, height / scale_factor,
    )

    result = {
        "center_x": center_x,
        "center_y": center_y,
        "width": width / scale_factor,
        "height": height / scale_factor,
        "scale_factor": scale_factor,
        "rounding": rounding,
    }
    if paper_info:
        result["paper_info"] = paper_info

    return result


# ── Step 4: Extract Entities ──────────────────────────────────────────────────


def _r(value: float, precision: int) -> float:
    """Round a value to the given precision (0 = integer, 1 = one dp, etc.)."""
    return round(value, precision) if precision > 0 else int(round(value))


def extract_lines(entity, center_x: float, center_y: float,
                  scale: float = 1.0, rounding: int = 2) -> dict | None:
    """Extract a LINE entity, applying translation and scaling."""
    sx = _r((entity.dxf.start.x - center_x) / scale, rounding)
    sy = _r((entity.dxf.start.y - center_y) / scale, rounding)
    ex = _r((entity.dxf.end.x - center_x) / scale, rounding)
    ey = _r((entity.dxf.end.y - center_y) / scale, rounding)

    # Alignment tolerance scales with the coordinate magnitude
    tol = 0.1 / scale if scale > 1 else 0.1
    if abs(sx - ex) < tol:
        alignment = "vertical"
    elif abs(sy - ey) < tol:
        alignment = "horizontal"
    else:
        alignment = "diagonal"

    return {"start": [sx, sy], "end": [ex, ey], "alignment": alignment}


def extract_lwpolyline(entity, center_x: float, center_y: float,
                       scale: float = 1.0, rounding: int = 2) -> list[dict]:
    """Extract an LWPOLYLINE into individual line segments, applying translation and scaling."""
    segments = []
    points = entity.get_points()
    tol = 0.1 / scale if scale > 1 else 0.1

    for i in range(len(points) - 1):
        sx = _r((points[i][0] - center_x) / scale, rounding)
        sy = _r((points[i][1] - center_y) / scale, rounding)
        ex = _r((points[i + 1][0] - center_x) / scale, rounding)
        ey = _r((points[i + 1][1] - center_y) / scale, rounding)

        if abs(sx - ex) < tol:
            alignment = "vertical"
        elif abs(sy - ey) < tol:
            alignment = "horizontal"
        else:
            alignment = "diagonal"

        segments.append({"start": [sx, sy], "end": [ex, ey], "alignment": alignment})
    return segments


def extract_circle_or_arc(entity, center_x: float, center_y: float,
                          scale: float = 1.0, rounding: int = 2) -> dict:
    """Extract a CIRCLE or ARC entity, applying translation and scaling."""
    cx = _r((entity.dxf.center.x - center_x) / scale, rounding)
    cy = _r((entity.dxf.center.y - center_y) / scale, rounding)
    radius = _r(entity.dxf.radius / scale, rounding)

    arc_angles = None
    if entity.dxftype() == "ARC":
        arc_angles = [
            round(entity.dxf.start_angle, 1),
            round(entity.dxf.end_angle, 1),
        ]

    return {"center": [cx, cy], "radius": radius, "angles": arc_angles}


def extract_text(entity, center_x: float, center_y: float,
                 scale: float = 1.0, rounding: int = 2) -> dict:
    """Extract a TEXT or MTEXT entity, applying translation and scaling."""
    tx = _r((entity.dxf.insert.x - center_x) / scale, rounding)
    ty = _r((entity.dxf.insert.y - center_y) / scale, rounding)

    if entity.dxftype() == "TEXT":
        content = entity.dxf.text
    else:
        content = entity.text
        # Strip MTEXT formatting codes
        content = ezdxf.tools.text.plain_mtext(content)
        content = content.replace("\n", " ")

    return {"position": [tx, ty], "text": content.strip()}


def extract_entities(msp, center_x: float, center_y: float,
                     scale: float = 1.0, rounding: int = 2) -> dict:
    """
    Iterate all modelspace entities and extract normalized geometry,
    grouped by layer. Explodes INSERT (block reference) entities into
    their constituent primitives. Extracts SOLID as line outlines.

    Returns a defaultdict of layers -> {lines, circles_arcs, text_annotations}.
    """
    logger.info(
        "Extracting and normalizing entities (scale=÷%.1f, rounding=%d dp)...",
        scale, rounding,
    )
    layers = defaultdict(
        lambda: {"lines": [], "circles_arcs": [], "text_annotations": []}
    )

    counts = {"LINE": 0, "LWPOLYLINE": 0, "CIRCLE": 0, "ARC": 0,
              "TEXT": 0, "MTEXT": 0, "INSERT": 0, "SOLID": 0, "skipped": 0}

    def _add_line(layer, sx, sy, ex, ey):
        tol = 0.5
        if abs(sx - ex) < tol:
            alignment = "vertical"
        elif abs(sy - ey) < tol:
            alignment = "horizontal"
        else:
            alignment = "diagonal"
        layers[layer]["lines"].append(
            {"start": [sx, sy], "end": [ex, ey], "alignment": alignment}
        )

    def _process_block_entity(sub_ent, ins_layer, ins_x, ins_y, ins_scale):
        """Process a sub-entity from an exploded block."""
        etype = sub_ent.dxftype()
        layer = ins_layer

        if etype == "LINE":
            sx = _r((sub_ent.dxf.start.x * ins_scale + ins_x - center_x) / scale, rounding)
            sy = _r((sub_ent.dxf.start.y * ins_scale + ins_y - center_y) / scale, rounding)
            ex = _r((sub_ent.dxf.end.x * ins_scale + ins_x - center_x) / scale, rounding)
            ey = _r((sub_ent.dxf.end.y * ins_scale + ins_y - center_y) / scale, rounding)
            _add_line(layer, sx, sy, ex, ey)
            counts["LINE"] += 1

        elif etype == "CIRCLE":
            cx = _r((sub_ent.dxf.center.x * ins_scale + ins_x - center_x) / scale, rounding)
            cy = _r((sub_ent.dxf.center.y * ins_scale + ins_y - center_y) / scale, rounding)
            radius = _r(sub_ent.dxf.radius * ins_scale / scale, rounding)
            layers[layer]["circles_arcs"].append(
                {"center": [cx, cy], "radius": radius, "angles": None}
            )
            counts["CIRCLE"] += 1

        elif etype == "ARC":
            cx = _r((sub_ent.dxf.center.x * ins_scale + ins_x - center_x) / scale, rounding)
            cy = _r((sub_ent.dxf.center.y * ins_scale + ins_y - center_y) / scale, rounding)
            radius = _r(sub_ent.dxf.radius * ins_scale / scale, rounding)
            layers[layer]["circles_arcs"].append(
                {"center": [cx, cy], "radius": radius,
                 "angles": [round(sub_ent.dxf.start_angle, 1),
                            round(sub_ent.dxf.end_angle, 1)]}
            )
            counts["ARC"] += 1

        elif etype in ("TEXT", "MTEXT"):
            tx = _r((sub_ent.dxf.insert.x * ins_scale + ins_x - center_x) / scale, rounding)
            ty = _r((sub_ent.dxf.insert.y * ins_scale + ins_y - center_y) / scale, rounding)
            if etype == "TEXT":
                content = sub_ent.dxf.text
            else:
                content = sub_ent.text
                content = ezdxf.tools.text.plain_mtext(content)
                content = content.replace("\n", " ")
            layers[layer]["text_annotations"].append(
                {"position": [tx, ty], "text": content.strip()}
            )
            counts[etype] += 1

    # Main iteration
    for entity in msp:
        layer = entity.dxf.layer
        etype = entity.dxftype()

        if etype == "INSERT":
            counts["INSERT"] += 1
            ins_pt = entity.dxf.insert
            ins_s = entity.dxf.get("xscale", 1.0)
            try:
                block = entity.doc.blocks.get(entity.dxf.name)
                if block:
                    for sub_ent in block:
                        if sub_ent.dxftype() in ("LINE", "CIRCLE", "ARC", "TEXT", "MTEXT"):
                            _process_block_entity(sub_ent, layer, ins_pt.x, ins_pt.y, ins_s)
            except Exception:
                pass

        elif etype == "SOLID":
            counts["SOLID"] += 1
            pts = []
            for attr in ("vtx0", "vtx1", "vtx2", "vtx3"):
                try:
                    v = getattr(entity.dxf, attr)
                    pts.append((_r((v.x - center_x) / scale, rounding),
                                _r((v.y - center_y) / scale, rounding)))
                except Exception:
                    break
            # Close the solid outline
            for i in range(len(pts)):
                j = (i + 1) % len(pts)
                if pts[i] != pts[j]:
                    _add_line(layer, pts[i][0], pts[i][1], pts[j][0], pts[j][1])

        elif etype == "LINE":
            layers[layer]["lines"].append(
                extract_lines(entity, center_x, center_y, scale, rounding)
            )
            counts["LINE"] += 1

        elif etype == "LWPOLYLINE":
            segs = extract_lwpolyline(entity, center_x, center_y, scale, rounding)
            layers[layer]["lines"].extend(segs)
            counts["LWPOLYLINE"] += 1

        elif etype in ("CIRCLE", "ARC"):
            layers[layer]["circles_arcs"].append(
                extract_circle_or_arc(entity, center_x, center_y, scale, rounding)
            )
            counts[etype] += 1

        elif etype in ("TEXT", "MTEXT"):
            layers[layer]["text_annotations"].append(
                extract_text(entity, center_x, center_y, scale, rounding)
            )
            counts[etype] += 1

        else:
            counts["skipped"] += 1

    logger.info(
        "Extraction complete. LINE=%d, LWPOLYLINE=%d, CIRCLE=%d, ARC=%d, "
        "TEXT=%d, MTEXT=%d, INSERT=%d (exploded), SOLID=%d, skipped=%d",
        counts["LINE"], counts["LWPOLYLINE"], counts["CIRCLE"], counts["ARC"],
        counts["TEXT"], counts["MTEXT"], counts["INSERT"], counts["SOLID"],
        counts["skipped"],
    )

    # ── Post-process: straighten nearly-vertical/horizontal lines ─────────────
    # In schematic drawings, lines are meant to be orthogonal. Small X/Y offsets
    # are mouse drawing errors. Snap lines that are within a tolerance of being
    # truly vertical or horizontal.
    straightened = 0
    for layer_data in layers.values():
        for seg in layer_data.get("lines", []):
            dx = abs(seg["end"][0] - seg["start"][0])
            dy = abs(seg["end"][1] - seg["start"][1])
            if dy > 3 and dx > 0 and dx / dy < 0.15:
                # Nearly vertical — snap X to midpoint
                mid_x = _r((seg["start"][0] + seg["end"][0]) / 2, rounding)
                seg["start"][0] = mid_x
                seg["end"][0] = mid_x
                seg["alignment"] = "vertical"
                straightened += 1
            elif dx > 3 and dy > 0 and dy / dx < 0.15:
                # Nearly horizontal — snap Y to midpoint
                mid_y = _r((seg["start"][1] + seg["end"][1]) / 2, rounding)
                seg["start"][1] = mid_y
                seg["end"][1] = mid_y
                seg["alignment"] = "horizontal"
                straightened += 1

    if straightened > 0:
        logger.info("Straightened %d nearly-orthogonal lines.", straightened)

    # ── Post-process: equalize spacing for aligned parallel lines ─────────────
    # Group vertical lines by approximate X position, then regularize spacing
    vert_lines = []
    for layer_data in layers.values():
        for seg in layer_data.get("lines", []):
            if seg["alignment"] == "vertical":
                vert_lines.append(seg)

    # Find groups of vertical lines at similar Y range (same "row" of MCBs)
    # Group by Y-band (similar start/end Y)
    if vert_lines:
        # Cluster by Y midpoint band
        y_bands: dict[int, list] = defaultdict(list)
        for seg in vert_lines:
            y_mid = int(round((seg["start"][1] + seg["end"][1]) / 2))
            # Quantize to 20-unit bands
            band = y_mid // 20
            y_bands[band].append(seg)

        for band, band_lines in y_bands.items():
            if len(band_lines) < 3:
                continue
            # Sort by X
            band_lines.sort(key=lambda s: s["start"][0])
            xs = [s["start"][0] for s in band_lines]
            # Compute spacings
            spacings = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
            if not spacings:
                continue
            # Check if spacings are similar (std < 30% of mean)
            avg_space = sum(spacings) / len(spacings)
            if avg_space < 1:
                continue
            std_space = (sum((s - avg_space) ** 2 for s in spacings) / len(spacings)) ** 0.5
            if std_space / avg_space < 0.3 and std_space > 0.5:
                # Regularize: redistribute with equal spacing
                x_start = xs[0]
                for i, seg in enumerate(band_lines):
                    new_x = _r(x_start + i * avg_space, rounding)
                    seg["start"][0] = new_x
                    seg["end"][0] = new_x
                logger.info(
                    "Equalized %d vertical lines in Y-band %d (spacing=%.1f).",
                    len(band_lines), band * 20, avg_space,
                )

    return layers
