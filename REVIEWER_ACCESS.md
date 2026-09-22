# Reviewer Access Mirror

This repository is a GitHub mirror of the reproducibility package deposited in Mendeley Data for the manuscript:

**Network-aware control of inverter-based air-conditioner fleets for simultaneous frequency and voltage support**

Formal dataset record (resolves to the latest public version):

https://doi.org/10.17632/kdbtxv66n9

The repository contains the Python/OpenDSS simulation framework, calibration scripts, consistency/trackability-check scripts, representative experiment traces, generated result summaries, figures, tables, and regression tests needed to reproduce the reported computational results. See `README_RESEARCH_DATA_EPSR.txt` for the recommended reviewer workflow.

After downloading and extracting the Mendeley package, run its portable public-artifact check from the extracted package root with `python scripts/package_preflight.py --public-package .`. Manuscript assembly and submission-packaging tools are intentionally excluded because they are not needed to reproduce the public computational results.
