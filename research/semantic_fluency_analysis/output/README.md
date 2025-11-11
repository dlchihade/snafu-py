# Output directory

This folder contains generated figures and supporting artifacts.

## Layout
- figures/
  - bars/: bar charts (mean cosine similarity ± 95% CI)
  - tables/: table renderings
  - correlograms/
    - svf_top30/: correlograms using SVF-frequency top-30 words
    - svf_top30_z/: z-scored versions
    - svf_top30_combo/: consolidated multi-panel versions
    - zipf_top30/: correlograms using Zipf-frequency top-30 words
    - zipf_top30_z/: z-scored versions
    - zipf_top30_combo/: consolidated multi-panel versions
    - words/: word lists used to label correlograms (CSV/TXT)
- scripts/: symlinks to scripts used to generate figures
- NATURE_REAL_metrics.csv: aggregated metrics per participant used by the bar/table figures

## Re-generate key figures
From the repository root (/Users/diettachihade/snafu-py):

- Exploitation vs. Exploration bar chart:
  python3 research/semantic_fluency_analysis/create_exploit_explore_bar.py

- Exploitation vs. Exploration table (one-liner example embedded in that script above).

- Model comparison correlograms: see consolidated figures in figures/correlograms/*_combo/.
  Word lists used are in figures/correlograms/words/.

## Notes
- Colors use colorblind-friendly palettes (no yellows).
- All PDFs/PNGs are archived by theme to reduce clutter.
