"""Strict schema definition & validation layer for RaidenLocalFullText documents.

The schema layer defines the document fields, their types, constraints and
validation rules, and produces detailed, machine-readable error reports.

Field types
-----------
- ``text``      : str (non-empty after trimming, honouring min/max length)
- ``integer``   : int  (bools rejected; optional min/max bounds)
- ``number``    : int | float (bools rejected; optional min/max bounds)
- ``boolean``   : bool
- ``array``     : list of items validated against ``items`` type
- ``object``    : dict validated against nested ``fields``
- ``datetime``  : ISO-8601 string normalised to UTC ISO-8601

Constraints
-----------
- ``required``   : field must be present and not null
- ``nullable``   : field may be null (only meaningful when required)
- ``unique``     : value must be unique across indexed documents (enforced by
  the storage layer via a secondary index)
- ``min_length`` / ``max_length`` : string length bounds
- ``minimum`` / ``maximum``       : numeric bounds
- ``enum``                        : allowed values
- ``pattern``                     : regex the string must match
- ``primary_key``                 : marks the identity field of a document
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, field_validator


SUPPORTED_TYPES = {"text", "integer", "number", "boolean", "array", "object", "datetime"}

_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_RESERVED_NAMES = {"id", "created_at", "updated_at", "_doc_id", "_score"}


class ValidationError(Exception):
    """Raised when a schema definition itself is invalid."""


class FieldValidationError(Exception):
    """Raised when a document fails validation against the schema."""

    def __init__(self, errors: List[Dict[str, Any]]):
        super().__init__(f"document failed validation: {len(errors)} error(s)")
        self.errors = errors

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kind": "validation_error",
            "count": len(self.errors),
            "errors": self.errors,
        }


class FieldDefinition(BaseModel):
    name: str
    type: str
    required: bool = False
    nullable: bool = False
    unique: bool = False
    primary_key: bool = False
    default: Optional[Any] = None
    min_length: Optional[int] = Field(None, ge=0)
    max_length: Optional[int] = Field(None, ge=0)
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    enum: Optional[List[Any]] = None
    pattern: Optional[str] = None
    items: Optional["FieldDefinition"] = None
    fields: Optional[List["FieldDefinition"]] = None

    @field_validator("type")
    @classmethod
    def _type_is_supported(cls, value: str) -> str:
        if value not in SUPPORTED_TYPES:
            raise ValueError(f"unsupported field type {value!r}")
        return value

    @field_validator("pattern")
    @classmethod
    def _pattern_compiles(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        try:
            re.compile(value)
        except re.error as exc:  # pragma: no cover - defensive
            raise ValueError(f"invalid regex pattern: {exc}") from exc
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.min_length is not None and self.max_length is not None and self.min_length > self.max_length:
            raise ValueError("min_length must be <= max_length")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum must be <= maximum")
        if self.enum is not None and len(set(map(str, self.enum))) != len(self.enum):
            raise ValueError("enum values must be unique")
        if self.items is not None and self.type != "array":
            raise ValueError("items is only valid for array fields")
        if self.fields is not None and self.type != "object":
            raise ValueError("fields is only valid for object fields")

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"name": self.name, "type": self.type}
        for key in (
            "required",
            "nullable",
            "unique",
            "primary_key",
            "default",
            "min_length",
            "max_length",
            "minimum",
            "maximum",
            "enum",
            "pattern",
        ):
            value = getattr(self, key)
            if value is not None and value is not False:
                payload[key] = value
        if self.items is not None:
            payload["items"] = self.items.to_dict()
        if self.fields is not None:
            payload["fields"] = [f.to_dict() for f in self.fields]
        return payload


class DocumentSchema(BaseModel):
    """Strict definition of the documents accepted by the indexer."""

    name: str = "document"
    version: int = Field(1, ge=1)
    fields: List[FieldDefinition]

    @field_validator("fields")
    @classmethod
    def _fields_are_valid(cls, fields: List[FieldDefinition]) -> List[FieldDefinition]:
        if not fields:
            raise ValueError("schema must define at least one field")
        seen = set()
        pk_count = 0
        for field in fields:
            if not _FIELD_NAME_RE.match(field.name):
                raise ValueError(f"invalid field name {field.name!r}")
            if field.name in seen:
                raise ValueError(f"duplicate field name {field.name!r}")
            seen.add(field.name)
            if field.primary_key:
                pk_count += 1
        if pk_count != 1:
            raise ValueError("schema must define exactly one primary_key field")
        return fields

    @field_validator("name")
    @classmethod
    def _name_is_valid(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("schema name must not be empty")
        return value.strip()

    def field(self, name: str) -> Optional[FieldDefinition]:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    @property
    def primary_key_field(self) -> FieldDefinition:
        for f in self.fields:
            if f.primary_key:
                return f
        raise ValidationError("schema has no primary key")  # guarded by validator

    def unique_field_names(self) -> List[str]:
        return [f.name for f in self.fields if f.unique]

    def validate(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ``document`` and return a normalised copy.

        Raises :class:`FieldValidationError` with a detailed error list.
        """
        if not isinstance(document, dict):
            raise FieldValidationError(
                [{"field": None, "code": "invalid_document", "message": "document must be a JSON object"}]
            )

        errors: List[Dict[str, Any]] = []
        normalised: Dict[str, Any] = {}

        for key in document:
            if self.field(key) is None:
                errors.append(
                    {
                        "field": key,
                        "code": "unknown_field",
                        "message": f"field {key!r} is not defined in schema {self.name!r}",
                        "allowed_fields": sorted(f.name for f in self.fields),
                    }
                )

        for field in self.fields:
            present = field.name in document
            value = document.get(field.name)

            if not present:
                if field.required:
                    errors.append(
                        {
                            "field": field.name,
                            "code": "missing_field",
                            "message": f"field {field.name!r} is required",
                            "expected_type": field.type,
                        }
                    )
                elif field.default is not None:
                    normalised[field.name] = field.default
                continue

            if value is None:
                if field.nullable:
                    normalised[field.name] = None
                else:
                    errors.append(
                        {
                            "field": field.name,
                            "code": "null_not_allowed",
                            "message": f"field {field.name!r} must not be null",
                            "expected_type": field.type,
                        }
                    )
                continue

            normalised[field.name] = self._validate_value(field, value, errors)

        if errors:
            raise FieldValidationError(errors)
        return normalised

    def _validate_value(self, field: FieldDefinition, value: Any, errors: List[Dict[str, Any]]) -> Any:
        path = field.name

        if field.type == "text":
            if not isinstance(value, str):
                errors.append(
                    {
                        "field": path,
                        "code": "type_mismatch",
                        "message": f"expected text (string), got {type(value).__name__}",
                        "expected_type": "text",
                    }
                )
                return value
            if field.enum is not None and value not in field.enum:
                errors.append(
                    {
                        "field": path,
                        "code": "enum_violation",
                        "message": f"value {value!r} is not one of {field.enum}",
                        "allowed": field.enum,
                    }
                )
                return value
            if field.pattern is not None and not re.search(field.pattern, value):
                errors.append(
                    {
                        "field": path,
                        "code": "pattern_violation",
                        "message": f"value {value!r} does not match pattern {field.pattern!r}",
                        "pattern": field.pattern,
                    }
                )
                return value
            if field.min_length is not None and len(value) < field.min_length:
                errors.append(
                    {
                        "field": path,
                        "code": "min_length_violation",
                        "message": f"length {len(value)} is below minimum {field.min_length}",
                        "min_length": field.min_length,
                    }
                )
            if field.max_length is not None and len(value) > field.max_length:
                errors.append(
                    {
                        "field": path,
                        "code": "max_length_violation",
                        "message": f"length {len(value)} exceeds maximum {field.max_length}",
                        "max_length": field.max_length,
                    }
                )
            return value

        if field.type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(
                    {
                        "field": path,
                        "code": "type_mismatch",
                        "message": f"expected integer, got {type(value).__name__}",
                        "expected_type": "integer",
                    }
                )
                return value
            if field.minimum is not None and value < field.minimum:
                errors.append(
                    {
                        "field": path,
                        "code": "minimum_violation",
                        "message": f"value {value} is below minimum {field.minimum}",
                        "minimum": field.minimum,
                    }
                )
            if field.maximum is not None and value > field.maximum:
                errors.append(
                    {
                        "field": path,
                        "code": "maximum_violation",
                        "message": f"value {value} exceeds maximum {field.maximum}",
                        "maximum": field.maximum,
                    }
                )
            return value

        if field.type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(
                    {
                        "field": path,
                        "code": "type_mismatch",
                        "message": f"expected number, got {type(value).__name__}",
                        "expected_type": "number",
                    }
                )
                return value
            if field.minimum is not None and value < field.minimum:
                errors.append(
                    {
                        "field": path,
                        "code": "minimum_violation",
                        "message": f"value {value} is below minimum {field.minimum}",
                        "minimum": field.minimum,
                    }
                )
            if field.maximum is not None and value > field.maximum:
                errors.append(
                    {
                        "field": path,
                        "code": "maximum_violation",
                        "message": f"value {value} exceeds maximum {field.maximum}",
                        "maximum": field.maximum,
                    }
                )
            return value

        if field.type == "boolean":
            if not isinstance(value, bool):
                errors.append(
                    {
                        "field": path,
                        "code": "type_mismatch",
                        "message": f"expected boolean, got {type(value).__name__}",
                        "expected_type": "boolean",
                    }
                )
            return value

        if field.type == "datetime":
            if not isinstance(value, str):
                errors.append(
                    {
                        "field": path,
                        "code": "type_mismatch",
                        "message": f"expected ISO-8601 datetime string, got {type(value).__name__}",
                        "expected_type": "datetime",
                    }
                )
                return value
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                errors.append(
                    {
                        "field": path,
                        "code": "invalid_datetime",
                        "message": f"value {value!r} is not a valid ISO-8601 datetime",
                        "expected_type": "datetime",
                    }
                )
                return value
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()

        if field.type == "array":
            if not isinstance(value, list):
                errors.append(
                    {
                        "field": path,
                        "code": "type_mismatch",
                        "message": f"expected array, got {type(value).__name__}",
                        "expected_type": "array",
                    }
                )
                return value
            if field.min_length is not None and len(value) < field.min_length:
                errors.append(
                    {
                        "field": path,
                        "code": "min_length_violation",
                        "message": f"array length {len(value)} is below minimum {field.min_length}",
                        "min_length": field.min_length,
                    }
                )
            if field.max_length is not None and len(value) > field.max_length:
                errors.append(
                    {
                        "field": path,
                        "code": "max_length_violation",
                        "message": f"array length {len(value)} exceeds maximum {field.max_length}",
                        "max_length": field.max_length,
                    }
                )
            if field.items is None:
                return value
            cleaned: List[Any] = []
            for index, item in enumerate(value):
                if item is None and field.items.nullable:
                    cleaned.append(None)
                    continue
                sub_errors: List[Dict[str, Any]] = []
                cleaned.append(self._validate_value(field.items, item, sub_errors))
                for err in sub_errors:
                    errors.append(
                        {
                            "field": f"{path}[{index}]",
                            "code": err["code"],
                            "message": err["message"],
                            "expected_type": err.get("expected_type", field.items.type),
                        }
                    )
            return cleaned

        if field.type == "object":
            if not isinstance(value, dict):
                errors.append(
                    {
                        "field": path,
                        "code": "type_mismatch",
                        "message": f"expected object, got {type(value).__name__}",
                        "expected_type": "object",
                    }
                )
                return value
            if field.fields is None:
                return value
            nested_errors: List[Dict[str, Any]] = []
            nested_field_map = {f.name: f for f in field.fields}
            nested_prefix = f"{path}."
            for key in value:
                if key not in nested_field_map:
                    nested_errors.append(
                        {
                            "field": key,
                            "code": "unknown_field",
                            "message": f"field {key!r} is not defined in object {path!r}",
                            "allowed_fields": sorted(nested_field_map),
                        }
                    )
            for nested in field.fields:
                if nested.name in value:
                    nested_value = value[nested.name]
                    if nested_value is None:
                        if not nested.nullable:
                            nested_errors.append(
                                {
                                    "field": nested.name,
                                    "code": "null_not_allowed",
                                    "message": f"field {nested.name!r} must not be null",
                                    "expected_type": nested.type,
                                }
                            )
                    else:
                        self._validate_value(nested, nested_value, nested_errors)
                elif nested.required:
                    nested_errors.append(
                        {
                            "field": nested.name,
                            "code": "missing_field",
                            "message": f"field {nested.name!r} is required",
                            "expected_type": nested.type,
                        }
                    )
            errors.extend(
                {
                    "field": nested_prefix + err["field"],
                    "code": err["code"],
                    "message": err["message"],
                    **{k: v for k, v in err.items() if k not in {"field", "code", "message"}},
                }
                for err in nested_errors
            )
            return value

        # Unreachable: type validated on schema definition.
        errors.append({"field": path, "code": "unsupported_type", "message": f"unsupported type {field.type!r}"})
        return value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "fields": [f.to_dict() for f in self.fields],
            "primary_key": self.primary_key_field.name,
            "unique_fields": self.unique_field_names(),
        }


def parse_schema(payload: Dict[str, Any]) -> DocumentSchema:
    """Build a :class:`DocumentSchema` from a JSON payload with clear errors."""
    if not isinstance(payload, dict):
        raise ValidationError("schema payload must be a JSON object")
    fields_payload = payload.get("fields")
    if fields_payload is None:
        raise ValidationError("schema payload must contain a 'fields' array")
    if not isinstance(fields_payload, list):
        raise ValidationError("schema 'fields' must be an array")
    try:
        return DocumentSchema(
            name=payload.get("name", "document"),
            version=payload.get("version", 1),
            fields=[FieldDefinition(**f) for f in fields_payload],
        )
    except Exception as exc:  # pydantic / ValueError
        raise ValidationError(str(exc)) from exc


def validate_document(schema: DocumentSchema, document: Dict[str, Any]) -> Dict[str, Any]:
    return schema.validate(document)


def schema_errors(schema: DocumentSchema, document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of validation errors for ``document`` (empty when valid)."""
    try:
        schema.validate(document)
        return []
    except FieldValidationError as exc:
        return exc.errors


def iter_field_names(schema: DocumentSchema) -> Iterable[str]:
    return (f.name for f in schema.fields)
