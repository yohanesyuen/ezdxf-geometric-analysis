"""Infer a JSON Schema from a JSON file's structure, using genson."""

import json
import sys

from genson import SchemaBuilder


def _collapse_uniform_objects(schema):
    """Collapse objects with many identically-shaped properties (e.g. a dict
    keyed by layer name/id) into a single `additionalProperties` schema."""
    if not isinstance(schema, dict):
        return schema

    if schema.get("type") == "object" and "properties" in schema:
        props = {k: _collapse_uniform_objects(v) for k, v in schema["properties"].items()}
        values = list(props.values())
        if len(values) > 3 and all(v == values[0] for v in values[1:]):
            return {"type": "object", "additionalProperties": values[0]}
        return {**schema, "properties": props}

    if schema.get("type") == "array" and "items" in schema:
        return {**schema, "items": _collapse_uniform_objects(schema["items"])}

    return schema


def infer_schema(data) -> dict:
    builder = SchemaBuilder()
    builder.add_object(data)
    schema = builder.to_schema()
    schema.pop("$schema", None)
    return _collapse_uniform_objects(schema)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data.json"
    with open(path) as f:
        data = json.load(f)
    schema = infer_schema(data)
    print(json.dumps(schema, indent=2))
