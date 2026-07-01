# NetSentry — Data Quality Report

**Verdict: PASS (no failures)** — 0 warning(s).

Structural problems (missing columns, unknown labels, empty data) fail; quality problems (missingness, duplicates, degenerate balance) warn.

| check | status | detail |
|---|---|---|
| non_empty | PASS | 60,000 rows |
| required_features | PASS | all 77 feature columns present |
| label_vocabulary | PASS | 13 known labels |
| numeric_features | PASS | all present features numeric |
| missing_values | PASS | max missing/inf fraction 0.020 (Flow Bytes/s) |
| duplicates | PASS | 0.0% exact duplicate rows |
| class_balance | PASS | attack fraction 0.220 |
