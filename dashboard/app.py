"""BOT-2 shape scoring dashboard.

Upload one or more scanned drawings: the item is recognised automatically and the drawing is scored with the final
method of that item (rules and learned models from models/<shape>/), facet by facet.

Run locally from the project folder:
    streamlit run dashboard/app.py
Uploaded images are only processed in memory and in a temporary file that is deleted straight away.
"""
import json
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scoring import load_scorer  # noqa: E402
from src.scoring.base import MODELS_DIR  # noqa: E402
from src.shape_detection import SHAPE_NAMES, ShapeDetector  # noqa: E402

st.set_page_config(page_title="BOT-2 Shape Scoring", layout="wide")

# Final method and cross-validated results of every item (from the notebooks).
METHODS = {
    "circle": ("Gradient boosting on 116 shape features", 0.67, 0.916),
    "square": ("Rules + learned closure model", 0.82, 0.964),
    "triangle": ("Rules (triangle fit)", 0.70, 0.926),
    "diamond": ("Rules + basic-shape check + learned closure model", 0.67, 0.888),
    "star": ("Rules (five tips and five inner corners)", 0.67, 0.637),
    "wave": ("Rules (centre line split into two arcs)", 0.68, 0.871),
    "overlapped_circle": ("Rules + learned closure and overlap + basic-shape check", 0.54, 0.770),
    "overlapped_pencils": ("Fine-tuned EfficientNet-B0 (10 networks averaged)", 0.55, 0.637),
}
EXPERT_SCORE = re.compile(r"-(\d)(?:\((\d+)\))?$")  # file names of the dataset: img<id>-<code>-<score>(<facets>)

st.markdown("""
<style>
:root { --ink: #1f2430; --muted: #6b7280; --line: #e6e8ee; --pass: #14804a; --pass-bg: #e9f7ef;
        --fail: #c0392b; --fail-bg: #fdecea; --accent: #2a5bd7; --card: #ffffff; }
.block-container { padding-top: 2rem; max-width: 1200px; }
h1, h2, h3 { color: var(--ink); letter-spacing: -0.01em; }
.hero { padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, #eef3ff 0%, #f7f9ff 60%, #ffffff 100%); border: 1px solid var(--line); }
.hero h1 { margin: 0 0 .3rem 0; font-size: 1.9rem; }
.hero p { margin: 0; color: var(--muted); font-size: 1rem; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1.1rem 1.2rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, .04); }
.score { font-size: 3.2rem; font-weight: 700; color: var(--ink); line-height: 1; }
.score small { font-size: 1.4rem; color: var(--muted); font-weight: 500; }
.label { text-transform: uppercase; font-size: .72rem; letter-spacing: .08em; color: var(--muted); margin-bottom: .3rem; }
.shape { font-size: 1.25rem; font-weight: 600; color: var(--ink); }
.pill { display: inline-block; padding: .15rem .6rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
        background: #eef1f6; color: #3d4452; margin-right: .3rem; }
.facet { display: flex; gap: .8rem; align-items: flex-start; padding: .7rem .2rem; border-bottom: 1px solid var(--line); }
.facet:last-child { border-bottom: none; }
.badge { flex: 0 0 auto; width: 4.2rem; text-align: center; padding: .2rem 0; border-radius: 8px; font-weight: 700;
         font-size: .8rem; }
.badge.pass { background: var(--pass-bg); color: var(--pass); }
.badge.fail { background: var(--fail-bg); color: var(--fail); }
.facet-name { font-weight: 600; color: var(--ink); }
.facet-detail { color: var(--muted); font-size: .88rem; margin-top: .1rem; }
.facet-by { color: #8a90a0; font-size: .74rem; margin-top: .15rem; }
.note { color: var(--muted); font-size: .85rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the shape detector …")
def get_detector():
    return ShapeDetector(MODELS_DIR / "shape_detector")


@st.cache_resource(show_spinner="Loading the scoring models …")
def get_scorer(shape):
    return load_scorer(shape)


def read_upload(upload):
    """Decode the uploaded bytes into a BGR image, and write them to a temporary file for the scorers."""
    data = upload.getvalue()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return data, image


def score_upload(data, shape):
    suffix = Path(st.session_state.get("current_name", "x.png")).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:  # deleted when closed
        handle.write(data)
        handle.flush()
        return get_scorer(shape).score(handle.name)


def facet_rows(result):
    html = []
    for facet in result.facets:
        state = "pass" if facet.passed else "fail"
        html.append(f"""<div class="facet"><div class="badge {state}">{'✓ 1' if facet.passed else '✗ 0'}</div>
            <div><div class="facet-name">{facet.name}</div><div class="facet-detail">{facet.detail}</div>
            <div class="facet-by">decided by: {facet.decided_by}</div></div></div>""")
    return "".join(html)


# ---------------------------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown("Each drawing is cleaned, the BOT-2 item is recognised, and the drawing is scored facet by facet "
                "with the final method of that item.")
    detector_config = json.loads((MODELS_DIR / "shape_detector" / "config.json").read_text())
    st.markdown(f"**Shape detector:** {detector_config['cross_validated_accuracy']:.1%} correct in cross-validation.")
    st.markdown("**Methods and cross-validated results**")
    st.dataframe(pd.DataFrame([{"Item": SHAPE_NAMES[k], "QWK": f"{q:.2f}", "Acc.": f"{a:.0%}"}
                               for k, (m, q, a) in METHODS.items()]),
                 hide_index=True, width="stretch")
    st.caption("Drawings from the training data score optimistically here; the table shows results on drawings the "
               "models had not seen. Everything runs on this computer; uploaded images are not stored.")

# ---------------------------------------------------------------------------------------------------------- main
st.markdown("""<div class="hero"><h1>BOT-2 Shape Scoring</h1>
<p>Upload a scanned or photographed drawing. The item is recognised automatically and scored facet by facet.</p></div>""",
            unsafe_allow_html=True)

uploads = st.file_uploader("Drawings (PNG or JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True,
                           label_visibility="collapsed")
if not uploads:
    st.info("Drag one or more drawings here to score them.")
    st.stop()

names = [u.name for u in uploads]
selected = st.selectbox("Drawing", names, index=0) if len(uploads) > 1 else names[0]
upload = uploads[names.index(selected)]
st.session_state["current_name"] = upload.name
data, image = read_upload(upload)
if image is None:
    st.error("This file could not be read as an image.")
    st.stop()

with st.spinner("Recognising the item …"):
    detected, probabilities = get_detector().predict(image)
confidence = probabilities[detected]
options = list(SHAPE_NAMES)
choice = st.radio("Item", options, index=options.index(detected), horizontal=True,
                  format_func=lambda k: SHAPE_NAMES[k] + (" (detected)" if k == detected else ""),
                  key=f"item_{selected}")
if confidence < detector_config["ask_user_to_check_if_confidence_below"]:
    st.warning(f"The detector is not sure ({confidence:.0%} {SHAPE_NAMES[detected]}). Please check the item above.")

try:
    with st.spinner(f"Scoring the {SHAPE_NAMES[choice].lower()} …"):
        result = score_upload(data, choice)
except Exception as error:  # an unusual image the measurements cannot handle
    st.error(f"This drawing could not be scored as a {SHAPE_NAMES[choice].lower()} ({type(error).__name__}).")
    st.stop()

left, right = st.columns([1, 1.25], gap="large")
with left:
    expert = EXPERT_SCORE.search(Path(upload.name).stem)
    expert_html = (f'<span class="pill">expert score in file name: {expert.group(1)}</span>' if expert else "")
    st.markdown(f"""<div class="card"><div class="label">Score</div>
        <div class="score">{result.total}<small> / {result.max_score}</small></div>
        <div style="margin-top:.8rem" class="label">Item</div><div class="shape">{SHAPE_NAMES[choice]}</div>
        <div style="margin-top:.6rem"><span class="pill">detector: {confidence:.0%} {SHAPE_NAMES[detected]}</span>
        {expert_html}</div>
        <div style="margin-top:.8rem" class="note">{result.method}</div></div>""", unsafe_allow_html=True)
    for note in result.notes:
        st.caption(note)
    st.write("")
    picture_left, picture_right = st.columns(2)
    picture_left.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded drawing", width="stretch")
    picture_right.image(255 - result.mask * 255, caption="Cleaned ink that was measured", width="stretch",
                        clamp=True)
with right:
    st.markdown(f'<div class="card"><div class="label">Facets</div>{facet_rows(result)}</div>', unsafe_allow_html=True)
    with st.expander("Detector probabilities"):
        st.dataframe(pd.DataFrame({"Item": [SHAPE_NAMES[k] for k in probabilities],
                                   "Probability": [f"{p:.1%}" for p in probabilities.values()]}),
                     hide_index=True, width="stretch")

if len(uploads) > 1 and st.button("Score all uploaded drawings"):
    rows, progress = [], st.progress(0.0)
    for i, u in enumerate(uploads):
        st.session_state["current_name"] = u.name
        d, im = read_upload(u)
        try:
            shape, probs = get_detector().predict(im)
            r = score_upload(d, shape)
            rows.append({"file": u.name, "item": SHAPE_NAMES[shape], "detector confidence": round(probs[shape], 3),
                         "score": r.total, "max": r.max_score,
                         **{f.name: int(f.passed) for f in r.facets}})
        except Exception as error:
            rows.append({"file": u.name, "item": "could not be scored", "score": None, "error": type(error).__name__})
        progress.progress((i + 1) / len(uploads))
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download the scores (CSV)", table.to_csv(index=False).encode(), "scores.csv", "text/csv")
