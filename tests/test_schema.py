import pytest

from app.schema import (
    DocumentSchema,
    FieldDefinition,
    ValidationError,
    parse_schema,
    schema_errors,
    validate_document,
)


def make_schema():
    return DocumentSchema(
        name="article",
        version=2,
        fields=[
            FieldDefinition(name="slug", type="text", required=True, unique=True, primary_key=True, max_length=64),
            FieldDefinition(name="title", type="text", required=True, min_length=3),
            FieldDefinition(name="body", type="text", required=True),
            FieldDefinition(name="word_count", type="integer", minimum=0),
            FieldDefinition(name="score", type="number", minimum=0.0, maximum=1.0, default=0.5),
            FieldDefinition(name="published", type="boolean", default=False),
            FieldDefinition(name="tags", type="array", items=FieldDefinition(name="item", type="text", min_length=1)),
            FieldDefinition(name="author", type="object", fields=[FieldDefinition(name="name", type="text", required=True)]),
            FieldDefinition(name="created_at", type="datetime"),
        ],
    )


def test_schema_requires_primary_key():
    with pytest.raises(ValueError):
        DocumentSchema(fields=[FieldDefinition(name="title", type="text")])


def test_schema_rejects_duplicate_and_bad_names():
    with pytest.raises(ValueError):
        DocumentSchema(
            fields=[
                FieldDefinition(name="id", type="text", primary_key=True),
                FieldDefinition(name="id", type="text"),
            ]
        )
    with pytest.raises(ValueError):
        DocumentSchema(fields=[FieldDefinition(name="1bad!", type="text", primary_key=True)])


def test_schema_rejects_unknown_type_and_bad_constraints():
    with pytest.raises(ValueError):
        FieldDefinition(name="x", type="blob")
    with pytest.raises(ValueError):
        FieldDefinition(name="x", type="text", min_length=10, max_length=5)
    with pytest.raises(ValueError):
        FieldDefinition(name="x", type="integer", minimum=10, maximum=5)
    with pytest.raises(ValueError):
        FieldDefinition(name="x", type="text", items=FieldDefinition(name="i", type="text"))


def test_valid_document_is_normalised_with_defaults():
    schema = make_schema()
    doc = validate_document(schema, {"slug": "hello", "title": "Hello", "body": "some body text"})
    assert doc["score"] == 0.5
    assert doc["published"] is False


def test_missing_required_field_reports_detailed_error():
    schema = make_schema()
    errors = schema_errors(schema, {"title": "abc"})
    assert errors
    codes = {e["code"] for e in errors}
    assert "missing_field" in codes
    offending = next(e for e in errors if e["code"] == "missing_field")
    assert offending["field"] == "slug"
    assert offending["expected_type"] == "text"


def test_type_mismatch_and_bounds_errors():
    schema = make_schema()
    errors = schema_errors(
        schema,
        {
            "slug": "ok",
            "title": "ab",
            "body": 123,
            "word_count": -5,
            "score": 2.0,
            "tags": ["good", 3],
        },
    )
    codes = {e["code"] for e in errors}
    assert {"type_mismatch", "min_length_violation", "minimum_violation", "maximum_violation"} <= codes
    assert any(e["field"] == "tags[1]" for e in errors)


def test_unknown_field_reported():
    schema = make_schema()
    errors = schema_errors(schema, {"slug": "x", "title": "abc", "body": "b", "bogus": 1})
    assert any(e["code"] == "unknown_field" and e["field"] == "bogus" for e in errors)


def test_null_handling():
    schema = DocumentSchema(
        fields=[
            FieldDefinition(name="id", type="text", primary_key=True),
            FieldDefinition(name="note", type="text", nullable=True),
            FieldDefinition(name="title", type="text", required=True),
        ]
    )
    assert validate_document(schema, {"id": "1", "title": "t", "note": None})["note"] is None
    errors = schema_errors(schema, {"id": "1", "title": "t", "note": None, "bogus": "x"})
    assert errors


def test_datetime_normalisation_and_pattern_and_enum():
    schema = DocumentSchema(
        fields=[
            FieldDefinition(name="id", type="text", primary_key=True),
            FieldDefinition(name="created_at", type="datetime", required=True),
            FieldDefinition(name="kind", type="text", enum=["post", "note"]),
            FieldDefinition(name="code", type="text", pattern=r"^[A-Z]{3}\d{2}$"),
        ]
    )
    doc = validate_document(schema, {"id": "1", "created_at": "2024-01-02T03:04:05Z", "kind": "post", "code": "ABC12"})
    assert doc["created_at"].endswith("+00:00")
    errors = schema_errors(schema, {"id": "1", "created_at": "not-a-date", "kind": "blog", "code": "abc"})
    codes = {e["code"] for e in errors}
    assert {"invalid_datetime", "enum_violation", "pattern_violation"} <= codes


def test_nested_object_validation():
    schema = make_schema()
    errors = schema_errors(schema, {"slug": "s", "title": "abc", "body": "b", "author": {"extra": 1}})
    codes = {e["code"] for e in errors}
    assert "unknown_field" in codes
    assert any(e["field"] == "author.name" and e["code"] == "missing_field" for e in errors)
    errors = schema_errors(schema, {"slug": "s", "title": "abc", "body": "b", "author": {"name": 5}})
    assert any(e["field"] == "author.name" and e["code"] == "type_mismatch" for e in errors)


def test_parse_schema_from_payload_and_errors():
    schema = parse_schema(
        {
            "name": "notes",
            "fields": [
                {"name": "id", "type": "text", "primary_key": True, "required": True},
                {"name": "text", "type": "text", "required": True},
            ],
        }
    )
    assert schema.primary_key_field.name == "id"
    with pytest.raises(ValidationError):
        parse_schema({"fields": [{"name": "x", "type": "mystery"}]})
    with pytest.raises(ValidationError):
        parse_schema("not an object")


def test_to_dict_roundtrip():
    schema = make_schema()
    payload = schema.to_dict()
    assert payload["primary_key"] == "slug"
    assert payload["unique_fields"] == ["slug"]
    rebuilt = parse_schema(payload)
    assert {f.name for f in rebuilt.fields} == {f.name for f in schema.fields}
