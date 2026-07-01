"""Tests for stages/merge.py -- SLAy auto-merge stage."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pynpxpipe.core.config import MergeConfig, PipelineConfig
from pynpxpipe.core.errors import MergeError
from pynpxpipe.core.session import ProbeInfo, Session, SessionManager, SubjectConfig
from pynpxpipe.stages.merge import MergeStage


def _default_steps_params() -> dict[str, dict[str, float | str]]:
    return MergeConfig().steps_params()


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
def merge_session(tmp_path: Path) -> Session:
    session_dir = tmp_path / "session_g0"
    session_dir.mkdir()
    bhv_file = tmp_path / "test.bhv2"
    bhv_file.write_bytes(b"\x00" * 30)
    output_dir = tmp_path / "output"
    session = SessionManager.create(
        session_dir,
        bhv_file,
        _make_subject(),
        output_dir,
        experiment="nsd1w",
        probe_plan={"imec0": "V4"},
        date="240101",
    )
    session.probes = [_make_probe("imec0", tmp_path)]
    session.config = PipelineConfig(merge=MergeConfig(enabled=True))
    return session


def _make_analyzer(unit_ids: list[int], *, has_extensions: bool = True) -> MagicMock:
    sorting = MagicMock()
    sorting.get_unit_ids.return_value = unit_ids
    analyzer = MagicMock()
    analyzer.sorting = sorting
    analyzer.recording = MagicMock()
    analyzer.has_extension.return_value = has_extensions
    return analyzer


def test_disabled_merge_skips_without_processing(merge_session: Session) -> None:
    merge_session.config.merge.enabled = False

    with patch.object(MergeStage, "_merge_probe") as mock_merge:
        MergeStage(merge_session).run()

    mock_merge.assert_not_called()


def test_load_failure_raises_merge_error(merge_session: Session) -> None:
    with (
        patch("pynpxpipe.stages.merge.si.load", side_effect=RuntimeError("missing sorted")),
        pytest.raises(MergeError, match="imec0"),
    ):
        MergeStage(merge_session)._merge_probe("imec0")


def test_unexpected_probe_error_wrapped_as_merge_error(merge_session: Session) -> None:
    with (
        patch.object(
            MergeStage,
            "_merge_probe",
            side_effect=RuntimeError("auto_merge failed"),
        ),
        pytest.raises(MergeError, match="Failed to merge imec0"),
    ):
        MergeStage(merge_session).run()

    cp = merge_session.output_dir / "checkpoints" / "merge_imec0.json"
    data = json.loads(cp.read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    assert "Failed to merge imec0" in data["error"]


def test_merge_probe_uses_spikeinterface_slay_preset(merge_session: Session) -> None:
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
        MergeStage(merge_session)._merge_probe("imec0")

    compute_merge_unit_groups.assert_called_once_with(
        analyzer,
        preset="slay",
        resolve_graph=True,
        steps_params=_default_steps_params(),
        extra_outputs=False,
    )
    merge_units_sorting.assert_called_once_with(
        analyzer.sorting,
        [(1, 2)],
        new_unit_ids=[1],
    )
    create_analyzer.assert_called_once()
    assert create_analyzer.call_args.kwargs["folder"] == (
        merge_session.output_dir / "03_merged" / "imec0"
    )
    assert create_analyzer.call_args.args[0] is merged_sorting


def test_merge_probe_passes_configured_slay_parameters(merge_session: Session) -> None:
    merge_session.config.merge.resolve_graph = False
    merge_session.config.merge.template_similarity.similarity_method = "cosine"
    merge_session.config.merge.template_similarity.template_diff_thresh = 0.4
    merge_session.config.merge.slay.k1 = 0.4
    merge_session.config.merge.slay.k2 = 1.5
    merge_session.config.merge.slay.slay_threshold = 0.65
    expected_steps = merge_session.config.merge.steps_params()

    analyzer = _make_analyzer([1, 2, 3])

    with (
        patch("pynpxpipe.stages.merge.si.load", return_value=analyzer),
        patch("pynpxpipe.stages.merge.si.create_sorting_analyzer"),
        patch(
            "spikeinterface.curation.compute_merge_unit_groups",
            return_value=[],
        ) as compute_merge_unit_groups,
    ):
        MergeStage(merge_session)._merge_probe("imec0")

    compute_merge_unit_groups.assert_called_once_with(
        analyzer,
        preset="slay",
        resolve_graph=False,
        steps_params=expected_steps,
        extra_outputs=False,
    )


def test_merge_log_records_slay_groups_and_checkpoint(merge_session: Session) -> None:
    analyzer = _make_analyzer([1, 2, 3], has_extensions=False)
    merged_sorting = MagicMock()
    merged_sorting.get_unit_ids.return_value = [3, 1]

    with (
        patch("pynpxpipe.stages.merge.si.load", return_value=analyzer),
        patch("pynpxpipe.stages.merge.si.create_sorting_analyzer"),
        patch("spikeinterface.curation.compute_merge_unit_groups", return_value=[(1, 2)]),
        patch("spikeinterface.curation.MergeUnitsSorting", return_value=merged_sorting),
    ):
        MergeStage(merge_session)._merge_probe("imec0")

    analyzer.compute.assert_any_call("random_spikes")
    analyzer.compute.assert_any_call("waveforms")
    analyzer.compute.assert_any_call("templates")
    analyzer.compute.assert_any_call("template_similarity")

    merged_dir = merge_session.output_dir / "03_merged" / "imec0"
    merge_log = json.loads((merged_dir / "merge_log.json").read_text(encoding="utf-8"))
    assert merge_log["preset"] == "slay"
    assert merge_log["resolve_graph"] is True
    assert merge_log["steps_params"] == _default_steps_params()
    assert merge_log["merges"] == [{"merged_ids": [1, 2], "new_id": 1}]
    assert merge_log["n_units_before"] == 3
    assert merge_log["n_units_after"] == 2

    cp = merge_session.output_dir / "checkpoints" / "merge_imec0.json"
    data = json.loads(cp.read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert data["n_merges"] == 1
