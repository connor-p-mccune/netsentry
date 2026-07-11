"""Zeek conn.log adapter: parsing (TSV + JSON lines), mapping arithmetic, e2e score."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from netsentry.data import schema
from netsentry.integrations.zeek import (
    ZEEK_META_COLUMNS,
    ZeekReadError,
    mapped_feature_count,
    read_conn_log,
    zeek_record_to_cic,
    zeek_to_cic,
)

_TSV = "\t".join

_CONN_LOG = "\n".join(
    [
        "#separator \\x09",
        "#set_separator\t,",
        "#empty_field\t(empty)",
        "#unset_field\t-",
        "#path\tconn",
        _TSV(
            [
                "#fields",
                "ts",
                "uid",
                "id.orig_h",
                "id.orig_p",
                "id.resp_h",
                "id.resp_p",
                "proto",
                "duration",
                "orig_bytes",
                "resp_bytes",
                "history",
                "orig_pkts",
                "resp_pkts",
            ]
        ),
        _TSV(
            [
                "1600000000.5",
                "CxT1",
                "10.0.0.5",
                "49152",
                "192.0.2.10",
                "80",
                "tcp",
                "2.0",
                "1000",
                "5000",
                "ShADadFf",
                "10",
                "20",
            ]
        ),
        _TSV(  # a scanner-ish record with unset duration/bytes
            [
                "1600000001.0",
                "CxT2",
                "10.0.0.6",
                "49153",
                "192.0.2.11",
                "22",
                "tcp",
                "-",
                "-",
                "-",
                "S",
                "1",
                "0",
            ]
        ),
        "#close\t2020-09-13-00-00-02",
    ]
)


def test_tsv_parse_respects_fields_and_unset(tmp_path: Path) -> None:
    path = tmp_path / "conn.log"
    path.write_text(_CONN_LOG, encoding="utf-8")
    records = read_conn_log(path)
    assert len(records) == 2
    assert records[0]["uid"] == "CxT1"
    assert records[0]["id.resp_p"] == "80"
    assert "duration" not in records[1]  # '-' unset fields are dropped


def test_json_lines_parse(tmp_path: Path) -> None:
    path = tmp_path / "conn.log"
    path.write_text(
        '{"uid": "Cj1", "id.orig_h": "10.0.0.1", "id.resp_p": 443, "proto": "tcp", '
        '"duration": 1.5, "orig_pkts": 3, "resp_pkts": 4, "orig_bytes": 30, "resp_bytes": 40}\n',
        encoding="utf-8",
    )
    (record,) = read_conn_log(path)
    assert record["id.resp_p"] == "443"
    row = zeek_record_to_cic(record)
    assert row["Flow Duration"] == pytest.approx(1_500_000)  # seconds -> microseconds


def test_not_a_zeek_log_raises(tmp_path: Path) -> None:
    path = tmp_path / "conn.log"
    path.write_text("ts,uid\n1,2\n", encoding="utf-8")
    with pytest.raises(ZeekReadError):
        read_conn_log(path)


def test_mapping_arithmetic_hand_checked() -> None:
    record = {
        "id.resp_p": "80",
        "duration": "2.0",
        "orig_pkts": "10",
        "resp_pkts": "20",
        "orig_bytes": "1000",
        "resp_bytes": "5000",
        "history": "ShADadFf",
    }
    row = zeek_record_to_cic(record)
    assert row["Flow Duration"] == pytest.approx(2_000_000)
    assert row["Flow Bytes/s"] == pytest.approx(3000.0)  # 6000 bytes / 2 s
    assert row["Flow Packets/s"] == pytest.approx(15.0)
    assert row["Fwd Packet Length Mean"] == pytest.approx(100.0)
    assert row["Bwd Packet Length Mean"] == pytest.approx(250.0)
    assert row["Down/Up Ratio"] == pytest.approx(2.0)
    assert row["SYN Flag Count"] == 1.0  # 'S' in history (lower bound)
    assert row["FIN Flag Count"] == 2.0  # 'F' + 'f'


def test_zero_duration_rates_are_nan_not_inf() -> None:
    # Matches cleaning's Inf -> NaN policy: the pipeline imputes, never sees inf.
    row = zeek_record_to_cic({"duration": "0.0", "orig_pkts": "1", "orig_bytes": "40"})
    assert "Flow Bytes/s" not in row  # non-finite values are left for imputation
    assert row["Total Fwd Packets"] == 1.0


def test_zeek_to_cic_covers_the_full_schema_with_meta() -> None:
    records = [
        {
            "uid": "C1",
            "id.orig_h": "10.0.0.5",
            "id.orig_p": "49152",
            "id.resp_h": "192.0.2.10",
            "id.resp_p": "80",
            "proto": "tcp",
            "duration": "2.0",
            "orig_pkts": "10",
            "resp_pkts": "20",
            "orig_bytes": "1000",
            "resp_bytes": "5000",
        }
    ]
    features, meta = zeek_to_cic(records)
    assert list(features.columns) == list(schema.FEATURE_COLUMNS)
    assert list(meta.columns) == ZEEK_META_COLUMNS
    assert meta.loc[0, "Zeek UID"] == "C1"
    # Unmapped detail features stay missing for the pipeline to impute.
    assert math.isnan(features.loc[0, "Flow IAT Mean"])
    assert np.isfinite(features.loc[0, "Flow Bytes/s"])
    mapped = mapped_feature_count(features)
    assert 15 <= mapped <= 30  # the volume/rate/shape subset, not the whole schema


@pytest.mark.slow
def test_score_zeek_log_end_to_end(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    from netsentry.config import load_settings
    from netsentry.data.split import make_splits
    from netsentry.integrations.zeek import score_zeek_log
    from netsentry.serving.batch import OUTPUT_COLUMNS
    from netsentry.serving.bundle import build_serving_bundle

    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.models_dir = tmp_path / "models"
    settings.mlflow.enabled = False
    settings.supervised.n_estimators = 40
    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)
    build_serving_bundle(settings)

    log = tmp_path / "conn.log"
    log.write_text(_CONN_LOG, encoding="utf-8")
    out = tmp_path / "scored.csv"
    stats = score_zeek_log(settings, log, out, flows_out=tmp_path / "flows.csv")
    assert stats["connections"] == 2
    result = pd.read_csv(out)
    assert len(result) == 2
    assert list(result.columns) == ZEEK_META_COLUMNS + OUTPUT_COLUMNS
    assert result["attack_probability"].between(0, 1).all()
