# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, TypedDict

# Re-exported from native module. See src/dataset/optimize.rs for implementation.
from .lance import Compaction as Compaction
from .lance import CompactionMetrics as CompactionMetrics
from .lance import CompactionPlan as CompactionPlan
from .lance import CompactionTask as CompactionTask
from .lance import RewriteResult as RewriteResult

# from .lance import CompactionPlan as CompactionPlan


@dataclass(frozen=True)
class CompactionPlannerConfig:
    """Configuration for selecting a compaction planner."""

    planner: str
    parameters: dict[str, Any]

    @staticmethod
    def default() -> "CompactionPlannerConfig":
        return CompactionPlannerConfig(planner="default", parameters={})

    @staticmethod
    def bounded(
        *,
        max_compaction_rows: int | None = None,
        max_compaction_bytes: int | None = None,
    ) -> "CompactionPlannerConfig":
        parameters: dict[str, Any] = {}
        if max_compaction_rows is not None:
            parameters["max_compaction_rows"] = max_compaction_rows
        if max_compaction_bytes is not None:
            parameters["max_compaction_bytes"] = max_compaction_bytes
        return CompactionPlannerConfig(planner="bounded", parameters=parameters)


def _resolve_compaction_planner(
    planner: str | None,
    *,
    max_compaction_rows: int | None,
    max_compaction_bytes: int | None,
) -> str | None:
    """
    Resolve the compaction planner to use.

    Parameters
    ----------
    planner : str, optional
        The compaction planner to use. If None, the default planner will be used.

    Returns
    -------
    str or None
        The compaction planner to use, or None if no planner is selected.
    """
    has_limit = max_compaction_rows is not None or max_compaction_bytes is not None
    if planner is None:
        return "bounded" if has_limit else None

    normalized = planner.strip().lower()
    if normalized == "bounded" and not has_limit:
        raise ValueError(
            "planner='bounded' requires at least one of "
            "max_compaction_rows or max_compaction_bytes."
        )
    if normalized == "default" and has_limit:
        raise ValueError(
            "planner='default' cannot be combined with "
            "max_compaction_rows or max_compaction_bytes."
        )
    return normalized


class CompactionOptions(TypedDict):
    """Options for compaction."""

    target_rows_per_fragment: Optional[int]
    """
    The target number of rows per fragment. This is the number of rows
    that will be in each fragment after compaction. (default: 1024*1024)
    """
    max_rows_per_group: Optional[int]
    """
    Max number of rows per group. This does not affect which fragments
    need compaction, but does affect how they are re-written if selected.
    (default: 1024)
    """
    max_bytes_per_file: Optional[int]
    """
    Max number of bytes in a single file.  This does not affect which
    fragments need compaction, but does affect how they are re-written if
    selected.  If this value is too small you may end up with fragments
    that are smaller than `target_rows_per_fragment`.

    The default will use the default from ``write_dataset``.
    """
    materialize_deletions: Optional[bool]
    """
    Whether to compact fragments with soft deleted rows so they are no
    longer present in the file. (default: True)
    """
    materialize_deletions_threadhold: Optional[float]
    """
    The fraction of original rows that are soft deleted in a fragment
    before the fragment is a candidate for compaction.
    (default: 0.1 = 10%)
    """
    planner: Optional[str]
    """
    The compaction planner to use. Supported values include ``"default"``
    and ``"bounded"``.
    """
    max_compaction_rows: Optional[int]
    """
    When using the bounded planner, stop planning additional compaction once the
    total number of input rows planned exceeds this value.
    """
    max_compaction_bytes: Optional[int]
    """
    When using the bounded planner, stop planning additional compaction once the
    total number of input bytes planned exceeds this value.
    """
    num_threads: Optional[int]
    """
    The number of threads to use when performing compaction. If not
    specified, defaults to the number of cores on the machine.
    """
    batch_size: Optional[int]
    """
    The batch size to use when scanning input fragments.  You may want
    to reduce this if you are running out of memory during compaction.

    The default will use the same default from ``scanner``.
    """
    compaction_mode: Optional[
        Literal["reencode", "try_binary_copy", "force_binary_copy"]
    ]
    """
    The compaction mode to use. Valid values:

    - ``"reencode"``: Decode and re-encode data (default).
    - ``"try_binary_copy"``: Try binary copy if fragments are compatible,
      fall back to reencode otherwise.
    - ``"force_binary_copy"``: Use binary copy or fail if fragments are
      not compatible.
    """
    binary_copy_read_batch_bytes: Optional[int]
    """
    The batch size in bytes for reading during binary copy operations.
    Controls how much data is read at once when performing binary copy.
    (default: 16MB)
    """
