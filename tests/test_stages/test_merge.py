"""Tests for stages/merge.py -- SLAy auto-merge stage."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pynpxpipe.core.config import MergeConfig, PipelineConfig
from pynpxpipe.core.session import ProbeInfo, Session, SessionManager, SubjectConfig
from pynpxpipe.stages.merge import MergeStage


def _make_subject() -> SubjectConfig:
    return SubjectConfig(
        subject_id="test",
        description="desc",
        species="Macaca mulatta",
        sex="M",
        age="P3Y",
        weight="10kg",
    )


def _make_probe(probe_id: str, base: Path) -> ProbeInfo:
    return ProbeInfo(
        probe_id=probe_id,
        ap_bin=base / f"{probe_id}.ap.bin",
        ap_meta=base / f"{probe_id}.ap.meta",
        lf_bin=None,
        lf_meta=None,
        sample_rate=30000.0,
        n_channels=384,
        serial_number="SN_TEST",
        probe_type="NP1010",
        target_area="V4",
    )


@pytest.fixture
def session(tmp_path: Path) -> Session:
    session_dir = tmp_path / "session_g0"
    session_dir.mkdir()
    bhv_file = tmp_path / "test.bhv2"
    bhv_file.write_bytes(b"\x00" * 30)
    output_dir = tmp_path / "output"
    s = SessionManager.create(
        session_dir,
        bhv_file,
        _make_subject(),
        output_dir,
        experiment="nsd1w",
        probe_plan={"imec0": "V4"},
        date="240101",
    )
    s.probes = [_make_probe("imec0", tmp_path)]
    s.config = PipelineConfig(merge=MergeConfig(enabled=True))
    return s


def _make_analyzer(unit_ids: list[int]) -> MagicMock:
    sorting = MagicMock()
    sorting.get_unit_ids.return_value = unit_ids
    analyzer = MagicMock()
    analyzer.sorting = sorting
    analyzer.recording = MagicMock()
    analyzer.has_extension.return_value = True
    return analyzer


def test_skip_when_disabled(session: Session) -> None:
    """MergeStage returns immediately when config.merge.enabled is false."""
    session.config.merge.enabled = False

    with patch("pynpxpipe.stages.merge.si.load") as load:
        MergeStage(session).run()

    load.assert_not_called()


def test_merge_probe_uses_spikeinterface_slay_preset(session: Session) -> None:
    """Task 3 requires the SpikeInterface SLAy preset, not SI's default preset."""
    analyzer = _make_analyzer([1, 2, 3])
    merged_sorting = MagicMock()
    merged_sorting.get_unit_ids.return_value = [3, 1]

    with (
        patch("pynpxpipe.stages.merge.si.load", return_value=analyzer),
        patch("pynpxpipe.stages.merge.si.create_sorting_analyzer") as create_analyzer,
        patch(
            "spikeinterface.curation.compute_merge_unit_groups",
            return_value=[(1, 2)],
        ) as compute_merge_unit_groups,
        patch(
            "spikeinterface.curation.MergeUnitsSorting",
            return_value=merged_sorting,
        ) as merge_units_sorting,
    ):
        MergeStage(session)._merge_probe("imec0")

    compute_merge_unit_groups.assert_called_once_with(
        analyzer,
        preset="slay",
        resolve_graph=True,
        extra_outputs=False,
    )
    merge_units_sorting.assert_called_once_with(
        analyzer.sorting,
        [(1, 2)],
        new_unit_ids=[1],
    )
    create_analyzer.assert_called_once()
    assert create_analyzer.call_args.kwargs["folder"] == session.output_dir / "03_merged" / "imec0"
    assert create_analyzer.call_args.args[0] is merged_sorting


def test_merge_log_records_slay_groups(session: Session) -> None:
    """merge_log.json records enough history to audit or roll back SLAy merges."""
    analyzer = _make_analyzer([1, 2, 3])
    merged_sorting = MagicMock()
    merged_sorting.get_unit_ids.return_value = [3, 1]

    with (
        patch("pynpxpipe.stages.merge.si.load", return_value=analyzer),
        patch("pynpxpipe.stages.merge.si.create_sorting_analyzer"),
        patch("spikeinterface.curation.compute_merge_unit_groups", return_value=[(1, 2)]),
        patch("spikeinterface.curation.MergeUnitsSorting", return_value=merged_sorting),
    ):
        MergeStage(session)._merge_probe("imec0")

    merge_log = json.loads(
        (session.output_dir / "03_merged" / "imec0" / "merge_log.json").read_text(encoding="utf-8")
    )
    assert merge_log["preset"] == "slay"
    assert merge_log["merges"] == [{"merged_ids": [1, 2], "new_id": 1}]
    assert merge_log["n_units_before"] == 3
    assert merge_log["n_units_after"] == 2
