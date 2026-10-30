# Real-Curve Gate Status

Status: BLOCKED pending real NHANES SPXRAW raw-curve files.

Do not run Gate 2 or Gate 3 yet. The current raw-curve files are stubs or corrupt-sized placeholders, not usable real curves:

| file | observed size | status |
|---|---:|---|
| `data/SPXRAW_E.rds` | 44 B | stub |
| `data/SPXRAW_F.rds` | 44 B | stub |
| `data/SPXRAW_G.rds` | 44 B | stub |
| `data/SPXRAW_G.xpt` | 20 KB | corrupt/error-page-sized stub |
| `data/SPXRAW_G.csv` | 3 B | stub |

Next action when real curves land: run the staged prompt in `results/gate2_3_staged_verdict_prompt.md`.

Do not optimize the TSNPE CNN encoder before Gate 3. The remaining project verdict is whether OMBE fits real NHANES curves, especially obstructed curves.
