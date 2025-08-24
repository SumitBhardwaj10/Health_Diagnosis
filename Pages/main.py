import streamlit as st
import pandas as pd
from keras.src.layers import maximum
from tensorflow.keras.models import load_model
import joblib
import time
import json
import random
import numpy as np

st.set_page_config(
    page_title="Health Diagnosis — About",
    page_icon="🩺",
    layout="wide"
)


st.markdown("""
<style>
        .pill {
            display:inline-flex; align-items:center; gap:.5rem;
            padding:.35rem .7rem; border-radius:999px;
            background:linear-gradient(135deg, #EEF2FF, #F5F3FF);
            border:1px solid rgba(99,102,241,.25); color:#111827; font-weight:600;
        }
        .card {
            border-radius:1rem; padding:1rem; height:100%;
            background:linear-gradient(180deg, #FFFFFF, #FAFAFA);
            border:1px solid #eee; box-shadow:0 4px 16px rgba(0,0,0,.05);
        }
        .card h3 {
            margin:0 0 .25rem 0; font-size:1.1rem;
        }
        .subtle {
            color:#6B7280; font-size:.9rem;
        }
        .divider {
            height:1px; background:rgba(0,0,0,.06); margin:.75rem 0 1rem 0;
        }
        .chip {
            display:inline-flex; align-items:center; gap:.4rem;
            padding:.25rem .6rem; margin:.15rem .15rem 0 0;
            border-radius:999px; border:1px solid rgba(0,0,0,.08);
            background:rgba(0,0,0,.03); font-size:.85rem;
        }
        .bignum {
            font-size:2rem; font-weight:800; line-height:1; letter-spacing:-.02em;
        }
        </style>
<style>
    .chip {
        display:inline-flex; align-items:center; gap:.4rem;
        padding:.35rem .6rem; margin:.25rem .25rem 0 0;
        border-radius:999px; border:1px solid rgba(0,0,0,.1);
        background:rgba(0,0,0,.04); font-size:0.9rem;
    }
    .chip-label { white-space:nowrap; }
    </style>
<style>
    .info-container {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 20px;
        font-size: 16px;
        color: #333;
        margin-bottom: 15px;
    }
    </style>
<style>
.small { font-size:0.9rem; opacity:0.8 }
.kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
       background:#1111; border:1px solid #5555; padding:2px 6px; border-radius:6px; }
.banner {
  padding: 20px 22px; border-radius: 16px;
  background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(56,189,248,0.12));
  border: 1px solid rgba(255,255,255,0.16);
}
.pill { padding:4px 10px; border-radius:999px; background:#00000010; margin-right:8px; }
hr { border: none; height: 1px; background: linear-gradient(90deg, #0000, #0003, #0000); }
</style>
""", unsafe_allow_html=True)

def initializer():
    if "start" not in st.session_state: st.session_state.start = 1
    if "selected" not in st.session_state: st.session_state.selected = []
    if "top_diseases" not in st.session_state: st.session_state.top_diseases=None
    if "score" not in st.session_state: st.session_state.score=None
    if "doctor" not in st.session_state:st.session_state.doctor=None
    if "current" not in st.session_state:st.session_state.current=None

initializer()

c1, c2 = st.columns([0.75, 0.25])
with c1:
    st.title("🩺 Health Diagnosis-Predictor")
    st.markdown(
        "<div class='banner'><b>Ready To Go 🔥🔥</b> .</div>",
        unsafe_allow_html=True
    )
with c2:
    st.metric("Build Using", "ANN", "relu activation function")
    st.metric("Average Accuracy Of Model", "85%")

st.markdown("<div class='info-container'></div>",unsafe_allow_html=True)

def load_data():
    model_path="Model/predictor.keras"
    diseases_path="Model/encoder_class.pkl"
    symptoms_path="Model/symbtoms.pkl"
    file_path="Model/Doctos.json"

    with st.spinner("Loading data..."):
        model=load_model(model_path)
        diseases=list(joblib.load(diseases_path,"wb"))
        symptoms=list(joblib.load(symptoms_path,"wb"))
        symptoms.remove("diseases")
        with open(file_path, "r", encoding="utf-8") as f:
            disease_doctors = json.load(f)
        time.sleep(2)

    return model,diseases,symptoms,disease_doctors


@st.cache_data()
def make_vector(diseases,symptoms,features,count):
    sym2idx = {s: i for i, s in enumerate(symptoms)}
    idx = [sym2idx[f] for f in features if f in sym2idx]
    vector = np.zeros(len(symptoms), dtype=np.float32)
    if idx:
        vector[idx] = 1.0
    vector = vector.reshape(1, -1)
    preds = Model.predict(vector)
    preds = np.asarray(preds).reshape(-1)
    k = min(count, preds.size)
    top_idx = np.argpartition(preds, -k)[-k:]
    top_idx = top_idx[np.argsort(preds[top_idx])[::-1]]
    diseases = np.asarray(diseases)
    top_diseases = diseases[top_idx]
    top_scores = preds[top_idx]
    return top_diseases, top_scores

def make_ui(top_diseases, top_scores, doctors_data):
    st.caption("These are the most likely conditions based on your inputs.")
    scores = np.array(top_scores, dtype=float).reshape(-1)
    diseases = np.array(list(top_diseases), dtype=str).reshape(-1)
    doctors = []
    for name in diseases:
        doc = None
        if isinstance(doctors_data, dict) and name in doctors_data:
            inner = doctors_data.get(name, {})
            doc = inner.get("DoctorToVisit") or inner.get("doctor")
        doctors.append(doc if doc else "Unable to suggest")

    if scores.max() > 1.0 or scores.min() < 0.0:
        ex = np.exp(scores - scores.max())
        probs = ex / ex.sum()
    else:
        probs = scores
    df = pd.DataFrame({
        "Rank": np.arange(1, len(diseases) + 1),
        "Disease": diseases,
        "Doctor": doctors,
        "Score": scores,
        "Confidence": (probs * 100).round(2)
    })

    left, mid, right = st.columns([1.2, 1, 1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Most likely**")
        st.markdown(f"<div class='bignum'>{df.loc[0, 'Disease']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='subtle'>Confidence: {df.loc[0, 'Confidence']}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.progress(float(probs[0]))
        st.markdown("</div>", unsafe_allow_html=True)

    with mid:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Average confidence**")
        st.markdown(f"<div class='bignum'>{df['Confidence'].mean():.1f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='subtle'>Across top {len(df)}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Spread (max − min)**")
        st.markdown(f"<div class='bignum'>{(df['Confidence'].max() - df['Confidence'].min()):.1f}%</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='subtle'>Max {df['Confidence'].max():.1f}% / Min {df['Confidence'].min():.1f}%</div>",
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📊 Ranked results")
    cols = st.columns(3)
    for i, row in df.iterrows():
        with cols[i % 3]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**#{int(row['Rank'])} — {row['Disease']}**")
            st.markdown(f"Doctor to visit : {row['Doctor']}")
            st.markdown(f"<span class='subtle'>Confidence: {row['Confidence']}%</span>", unsafe_allow_html=True)
            st.progress(float(probs[i]))

            st.markdown("<div class='chip'>Likelihood</div><div class='chip'>AI model</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.start is not None:
    Model,Diseases,Symbtoms,st.session_state.doctors=load_data()

    c1,c2=st.columns([1,1])
    with c1:
        st.header("❄️ Diseases Informations ")
        st.metric("Total Number Of Disease In This Model",len(Diseases))
        st.caption("Some of the diseases info")
        visual=random.sample(Diseases,5)
        visual=pd.DataFrame(visual,columns=["Diseases"],index=["D1","D2","D3","D4","D5"])
        st.table(visual)
    with c2:
        st.header("👾 Symbtoms Informations ")
        st.metric("Total Number Of Symbtoms Available",len(Symbtoms))
        st.caption("Some of the symbtoms info")
        visual=random.sample(Symbtoms,5)
        visual=pd.DataFrame(visual,columns=["Symbtoms"],index=["S1","S2","S3","S4","S5"])
        st.table(visual)

    left,right=st.columns([1,1])
    with left:
        st.header("🤹🏻‍♂️Select Symbtoms")
        st.markdown("### Search & select")
        st.session_state.current = st.multiselect(
            "Start typing to search…",
            options=Symbtoms,
            default=st.session_state.selected,
            placeholder="e.g., cross-eyed,skin rash",
        )
        if st.session_state.current  != st.session_state.selected:
            st.session_state.selected = st.session_state.current
        st.write("Final selection:", st.session_state.selected)

        st.sidebar.subheader("🔎 Predictor Zone")
        count = st.sidebar.slider("Select Top Diseases Count", 2, 5, 3,1)

        if st.button("✅Predict Diseases"):
            if len(st.session_state.current )<2:
                st.error("Select atleast 2 Symbtoms")
            else:
                st.session_state.top_diseases,st.session_state.score=make_vector(Diseases,Symbtoms,st.session_state.current ,count)
                sidebar_result=pd.DataFrame(st.session_state.top_diseases,columns=["Diseases"])
                sidebar_result["Score"]=st.session_state.score
                sidebar_result["Score"]=sidebar_result["Score"].apply(lambda x: str(round(x*100,1))+"%")
                st.sidebar.markdown("### 🗂️ Table view")
                st.sidebar.table(sidebar_result)
                csv=sidebar_result.to_csv(index=False).encode("utf-8")
                st.sidebar.download_button("⬇️ Download results (CSV)", data=csv, file_name="top_diseases.csv", mime="text/csv")
                st.balloons()
                st.success("🙆🏻‍♂️Thank You for using this app")
    with right:
        st.header("🏹Diseases May Present ")
        if st.session_state.top_diseases is not None:
            make_ui(st.session_state.top_diseases,st.session_state.score,st.session_state.doctors)


