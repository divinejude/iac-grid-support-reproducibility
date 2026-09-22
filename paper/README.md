# Manuscript Asset Package

This folder contains the scripts, figures, and LaTeX tables used to generate manuscript assets from the experiment outputs.

Files:

- `generate_paper_assets.py`: regenerates figures and LaTeX tables from the latest experiment CSV files.
- `figures/`: publication figures in PDF and PNG format.
- `tables/`: LaTeX tables generated from calibration, experiment, and consistency/trackability-check outputs.

Regenerate assets:

```powershell
python paper\generate_paper_assets.py
```
