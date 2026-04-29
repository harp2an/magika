# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Type definitions for the Magika file type detection library."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class MagikaStatus(str, Enum):
    """Status codes for Magika detection results."""

    OK = "ok"
    # The file is empty; no meaningful prediction can be made.
    EMPTY = "empty"
    # The file is too small to make a confident prediction.
    TOO_SMALL = "too_small"
    # The model returned a low-confidence prediction; a generic type is used.
    LOW_CONFIDENCE = "low_confidence"
    # An error occurred during detection.
    ERROR = "error"


@dataclass
class ContentTypeInfo:
    """Metadata about a detected content type."""

    # Short label identifying the content type (e.g., "python", "pdf").
    label: str
    # Human-readable description of the content type.
    description: str
    # Standard MIME type string (e.g., "text/x-python", "application/pdf").
    mime_type: str
    # Common file extensions associated with this content type.
    extensions: list[str] = field(default_factory=list)
    # Whether this content type is considered a text-based format.
    is_text: bool = False

    def __str__(self) -> str:
        return self.label


@dataclass
class MagikaResult:
    """Result of a Magika file type detection operation."""

    # The file path that was analyzed, if applicable.
    path: Optional[Path]
    # Detection status indicating how the result was determined.
    status: MagikaStatus
    # Information about the detected content type.
    output: ContentTypeInfo
    # Model confidence score in the range [0.0, 1.0].
    # May be None if detection did not use the ML model.
    score: Optional[float] = None

    @property
    def ok(self) -> bool:
        """Return True if detection completed without errors."""
        return self.status == MagikaStatus.OK

    def __str__(self) -> str:
        path_str = str(self.path) if self.path is not None else "<bytes>"
        return (
            f"MagikaResult(path={path_str!r}, status={self.status.value}, "
            f"label={self.output.label!r}, score={self.score})"
        )


@dataclass
class ModelFeatures:
    """Features extracted from file content for model inference."""

    # Raw bytes sampled from the beginning of the file.
    beg: bytes
    # Raw bytes sampled from the middle of the file.
    mid: bytes
    # Raw bytes sampled from the end of the file.
    end: bytes

    @property
    def total_size(self) -> int:
        """Total number of bytes across all feature segments."""
        return len(self.beg) + len(self.mid) + len(self.end)
