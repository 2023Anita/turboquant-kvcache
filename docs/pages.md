# GitHub Pages

The repository includes a static GitHub Pages deployment workflow.

## What gets deployed

The site is built from:

- `scripts/build_pages.py`
- `assets/`
- the generated `report.html`

The output is written to the local `site/` directory and deployed through `.github/workflows/pages.yml`.

## Local build

```bash
PYTHONPATH=src python3 scripts/build_pages.py
```

## Published site

- <https://2023anita.github.io/turboquant-kvcache/>
