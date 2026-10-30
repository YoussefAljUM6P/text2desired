# Project A on Mip-NeRF 360 (6 scenes, 27 queries) — 2026-10-22

Model per scene: gsplat MCMC, data factor 1, 30k iterations, **no depth prior**. Script: `scripts/run_multiscene.sh`.
Visual check of every query: `contact_sheet.jpg` (left: detection + SAM mask in the reference view, right: desired image).
Grounding and view quality judged by eye (no annotation yet).

| scene | query | grounding | desired image | ρ | support | note |
|---|---|---|---|---|---|---|
| bicycle | bench | ok | ok (bicycle in front) | 0.60 | 0.89 | |
| bicycle | bicycle | ok | **fail** (blurry, object missing) | 0.13 | 0.00 | view outside training coverage |
| bonsai | bicycle | **wrong** (bonsai, conf 0.27) | fail | 0.25 | 0.84 | |
| bonsai | bonsai | ok | ok | 0.90 | 0.97 | |
| bonsai | cardboard box | **wrong** (bonsai) | (boxes visible by chance) | 0.30 | 0.65 | |
| counter | egg carton | ok | ok | 0.14 | 0.74 | |
| counter | metal bowl | doubtful (blue bowl) | partial | 0.29 | 0.74 | |
| counter | oven mitt | ok | ok | 0.07 | 0.97 | |
| counter | potted plant | ok | ok | 0.51 | 0.54 | |
| garden | round table | ok | ok | 0.80 | 0.74 | |
| garden | vase | ok | ok | 0.10 | 0.01 | good despite low support |
| kitchen | chair | ok | **fail** (blurry, cut) | 0.43 | 0.87 | |
| kitchen | toy bulldozer | ok | ok | 0.26 | 0.94 | |
| room | bookshelf | ok | partial (blurry edge) | 0.11 | 0.89 | |
| room | coffee table | **no detection** | — | — | — | |
| room | door | ok | ok | 0.24 | 0.82 | |
| room | lamp | ok | ok | 0.44 | 1.00 | |
| room | piano | ok | ok | 0.66 | 0.78 | |
| room | piano bench | **wrong** (sofa) | fail | 0.45 | 0.00 | |
| room | plant | ok | ok | 0.45 | 0.56 | |
| room | radiator | **wrong** (speaker) | fail | 0.38 | 0.88 | |
| room | screen | ok | ok | 0.44 | 0.84 | |
| room | side table | ok | ok | 0.61 | 0.96 | |
| room | sofa | ok | ok | 0.25 | 0.97 | |
| room | speaker | ok | **fail** (dark, blurry) | 0.34 | 0.04 | |
| room | television | ok | **fail** (garbage render) | 0.44 | 0.04 | |
| room | window | ok | **fail** (transparent) | 0.44 | 0.23 | known failure |

**Totals:** grounding correct 21/27 (+1 doubtful); usable desired image 15/27 (+2 partial).
**Failures:** 5 grounding (wrong object or none), 5 view failures with correct grounding.
**Pattern:** 4 of the 5 view failures have training-camera support ≤ 0.23 (bicycle 0.00, speaker 0.04, television 0.04,
window 0.23). The search only enforces min_support for the planar normal, not for the chosen candidate.
Servo: converged to the generated goal on every run (final pose error ~0), except piano bench.
