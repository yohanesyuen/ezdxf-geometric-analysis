"""CLI entry point for geometric DXF analysis (see pipeline.py for the 5-pass pipeline).

Writes JSON to <input_path>.json and Markdown to <input_path>.md.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from .pipeline import analyze_dxf

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) != 2:
        print("Usage: analyze-dxf <path-to-dxf>")
        sys.exit(1)

    dxf_path = sys.argv[1]
    result, markdown = analyze_dxf(dxf_path)

    json_path = Path(dxf_path).with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("JSON output written to: %s", json_path)

    md_path = Path(dxf_path).with_suffix(".md")
    md_path.write_text(markdown, encoding="utf-8")
    logger.info("Markdown payload written to: %s", md_path)


if __name__ == "__main__":
    main()
