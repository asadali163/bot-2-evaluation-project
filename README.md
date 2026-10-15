# BOT-2 Shape Scoring Dashboard

A Streamlit dashboard that scores children's copies of the eight BOT-2 Fine Motor Integration shapes (circle, square,
overlapping circles, wavy line, triangle, diamond, star, overlapping pencils). It recognises the item of an uploaded
drawing automatically and scores it facet by facet (basic shape, closure, edges, orientation, overlap, overall size),
showing the reason for every decision.

Each item is scored with the final method of the accompanying project: rule-based measurements with OpenCV, learned
facet models, or (for the overlapping pencils) a fine-tuned EfficientNet-B0.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Deploy on Streamlit Community Cloud

Create an app from this repository with the main file `dashboard/app.py` and Python 3.13 (Advanced settings).

## Structure

- `dashboard/app.py`: the Streamlit app
- `src/`: preprocessing, measurements and the scorer of every item (`src/scoring/`), and the shape detector
- `models/`: the final rule limits and trained models of every item and of the shape detector

This repository contains no data. Uploaded drawings are processed in memory and in a temporary file that is deleted
immediately.
