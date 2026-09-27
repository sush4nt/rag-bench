"""Resolve ``[n]`` citation markers in a generated answer to retrieved passages."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ragbench.common.protocol import Citation, RetrieveResult

# Matches "[1]", "[2, 3]" and "[2,3]"; adjacent markers like "[2][3]" match separately.
_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_citations(
    answer: str, contexts: Sequence[RetrieveResult]
) -> tuple[list[Citation], list[int]]:
    """Return ``(valid_citations, invalid_markers)`` in order of first appearance.

    A marker is invalid when it points outside ``1..len(contexts)`` — i.e. the
    model cited a passage it was never shown.
    """
    citations: list[Citation] = []
    invalid: list[int] = []
    seen: set[int] = set()
    for match in _MARKER_RE.finditer(answer):
        for raw in match.group(1).split(","):
            marker = int(raw)
            if marker in seen:
                continue
            seen.add(marker)
            if 1 <= marker <= len(contexts):
                citations.append(
                    Citation(
                        marker=marker,
                        doc_id=contexts[marker - 1].doc_id,
                        context_index=marker - 1,
                    )
                )
            else:
                invalid.append(marker)
    return citations, invalid
