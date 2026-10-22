"""Small private station-control contract; requests take effect at airing boundaries."""
from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
import rfc8785

from .contract import HASH_PATTERN, SLUG_PATTERN

CONTROL_LIMIT = 65_536
CATALOG_LIMIT = 1_048_576
CATALOG_ENTRIES = 1000


class ControlError(ValueError):
    pass


class ReleaseRef(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    release_id: Annotated[str, Field(pattern=SLUG_PATTERN)]
    package_sha256: Annotated[str, Field(pattern=HASH_PATTERN)]


class Rundown(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1]
    revision: Annotated[int, Field(gt=0, le=9_007_199_254_740_991)]
    mode: Literal["programmes", "fallback"]
    programme_releases: Annotated[list[ReleaseRef], Field(max_length=128)]
    fallback_release: ReleaseRef | None

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be integer 1")
        return value

    @model_validator(mode="after")
    def coherent_mode(self):
        ids = [entry.release_id for entry in self.programme_releases]
        if len(ids) != len(set(ids)):
            raise ValueError("programme releases must be unique")
        if self.mode == "programmes" and not self.programme_releases:
            raise ValueError("programmes mode requires at least one programme")
        if self.mode == "fallback" and self.fallback_release is None:
            raise ValueError("fallback mode requires a selected ident")
        if self.fallback_release and self.fallback_release.release_id in ids:
            raise ValueError("the fallback ident cannot also be a programme")
        return self


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ControlError("duplicate JSON key")
        result[key] = value
    return result


def parse_rundown(raw: bytes) -> Rundown:
    if len(raw) > CONTROL_LIMIT:
        raise ControlError("station control exceeds 64 KiB")
    try:
        data = json.loads(raw, object_pairs_hook=_unique_object)
        return Rundown.model_validate(data)
    except (ValueError, TypeError, ValidationError) as exc:
        raise ControlError("station control structure or version is invalid") from exc


def config_fingerprint(config: Rundown) -> str:
    return hashlib.sha256(rfc8785.dumps(config.model_dump(mode="json"))).hexdigest()
