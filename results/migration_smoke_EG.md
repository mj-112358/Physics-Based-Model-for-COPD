# Migration Smoke Test, Cycles E/G

- VERDICT: MIGRATION-CLEAN
- status: PASS

| cycle | raw_source | loaded_rows | expected_raw_rows | parse_errors | participants_with_truth | distinct_from_f |
|---|---|---|---|---|---|---|
| E | new-extracted | 34857 | 34857 | 26 | 6319 | True |
| G | new-extracted | 37661 | 37661 | 60 | 6652 | True |
