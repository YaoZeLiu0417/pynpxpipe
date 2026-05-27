"""Optional auto-merge stage using SpikeInterface SLAy auto-merge.

Default OFF (config.merge.enabled = False). When enabled, merges similar
units to reduce over-splitting. Creates a new SortingAnalyzer in
03_merged/{probe_id}/ — original sorted output is preserved.

Must run BEFORE curate so that Bombcell classification operates on the
final (merged) unit set.

No UI dependencies.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import spikeinterface.core as si

from pynpxpipe.core.errors import MergeError
from pynpxpipe.stages.base import BaseStage

if TYPE_CHECKING:
    from pynpxpipe.core.session import Session


class MergeStage(BaseStage):
    """Optional auto-merge stage using SpikeInterface SLAy auto-merge.

    Default OFF (config.merge.enabled = False). When enabled, merges
    similar units to reduce over-splitting. Creates a new SortingAnalyzer
    in 03_merged/{probe_id}/ — original sorted output is preserved.

    Must run BEFORE curate so that Bombcell classification operates
    on the final (merged) unit set.

    Raises:
        MergeError: If sorted analyzer cannot be loaded.
    """

    STAGE_NAME = "merge"

    def __init__(
        self,
        session: Session,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> None:
        """Initialize the merge stage.

        Args:
            session: Active pipeline session with sorting results available.
            progress_callback: Optional GUI/progress callback; None in CLI mode.
        """
        super().__init__(session, progress_callback)

    def run(self) -> None:
        """Run auto-merge for all probes, or skip if disabled."""
        if not self.session.config.merge.enabled:
            self._report_progress("Merge skipped (disabled)", 1.0)
            return

        if self._is_complete():
            self._report_progress("Merge already complete", 1.0)
            return

        self._report_progress("Starting merge", 0.0)
        self._setup_spikeinterface_jobs()

        n_probes = len(self.session.probes)
        for i, probe in enumerate(self.session.probes):
            probe_id = probe.probe_id
            try:
                self._merge_probe(probe_id)
            except Exception as exc:
                self._write_failed_checkpoint(exc, probe_id=probe_id)
                raise
            self._report_progress(f"Merged {probe_id}", (i + 1) / n_probes)

        self._write_checkpoint({"probe_ids": [p.probe_id for p in self.session.probes]})
        self._report_progress("Merge complete", 1.0)

    def _merge_probe(self, probe_id: str) -> None:
        """Auto-merge one probe's sorted units.

        Loads the sorted SortingAnalyzer, ensures required extensions are
        computed, runs SLAy merge-group detection, and saves the merged result to a new
        binary_folder. The original sorted output is not modified.

        Args:
            probe_id: Probe identifier (e.g. "imec0").

        Raises:
            MergeError: If sorted analyzer cannot be loaded.
        """
        if self._is_complete(probe_id=probe_id):
            return

        sorted_path = self.session.output_dir / "02_sorted" / probe_id

        try:
            analyzer = si.load(sorted_path)
        except Exception as exc:
            raise MergeError(f"Failed to load sorted analyzer for {probe_id}: {exc}") from exc

        n_before = len(analyzer.sorting.get_unit_ids())

        # Ensure core extensions for SLAy auto-merge. SpikeInterface can
        # compute additional preset-specific extensions internally.
        if not analyzer.has_extension("templates"):
            analyzer.compute("random_spikes")
            analyzer.compute("waveforms")
            analyzer.compute("templates")
        if not analyzer.has_extension("template_similarity"):
            analyzer.compute("template_similarity")

        from spikeinterface.curation import MergeUnitsSorting, compute_merge_unit_groups

        merge_config = self.session.config.merge
        steps_params = merge_config.steps_params()
        merge_groups = compute_merge_unit_groups(
            analyzer,
            preset=merge_config.preset,
            resolve_graph=merge_config.resolve_graph,
            steps_params=steps_params,
            extra_outputs=False,
        )
        merge_groups = _normalize_merge_groups(merge_groups)
        new_unit_ids = [group[0] for group in merge_groups]
        merged_sorting = (
            MergeUnitsSorting(
                analyzer.sorting,
                merge_groups,
                new_unit_ids=new_unit_ids,
            )
            if merge_groups
            else analyzer.sorting
        )

        merged_dir = self.session.output_dir / "03_merged" / probe_id
        merged_analyzer = si.create_sorting_analyzer(
            merged_sorting,
            analyzer.recording,
            format="binary_folder",
            folder=merged_dir,
            sparse=True,
        )

        n_after = len(merged_sorting.get_unit_ids())

        # Write merge_log.json
        merges = [
            {
                "merged_ids": [_json_safe_unit_id(unit_id) for unit_id in group],
                "new_id": _json_safe_unit_id(group[0]),
            }
            for group in merge_groups
        ]
        merge_log = {
            "preset": merge_config.preset,
            "resolve_graph": merge_config.resolve_graph,
            "steps_params": steps_params,
            "merges": merges,
            "n_units_before": n_before,
            "n_units_after": n_after,
        }
        merged_dir.mkdir(parents=True, exist_ok=True)
        (merged_dir / "merge_log.json").write_text(
            json.dumps(merge_log, indent=2), encoding="utf-8"
        )

        self.logger.info(
            "Merge result",
            probe_id=probe_id,
            n_before=n_before,
            n_after=n_after,
            n_merges=len(merges),
        )

        self._write_checkpoint(
            {
                "probe_id": probe_id,
                "n_units_before": n_before,
                "n_units_after": n_after,
                "n_merges": len(merges),
            },
            probe_id=probe_id,
        )

        del analyzer, merged_analyzer
        gc.collect()


def _normalize_merge_groups(merge_groups: list | tuple) -> list[tuple]:
    """Return only valid multi-unit merge groups as tuples."""
    normalized: list[tuple] = []
    for group in merge_groups:
        group_tuple = tuple(group)
        if len(group_tuple) > 1:
            normalized.append(group_tuple)
    return normalized


def _json_safe_unit_id(unit_id: object) -> int | str:
    try:
        return int(unit_id)
    except (TypeError, ValueError):
        return str(unit_id)
