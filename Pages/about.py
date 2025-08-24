import streamlit as st
import pandas as pd
import time
from datetime import datetime

st.set_page_config(
    page_title="Health Diagnosis — About",
    page_icon="🩺",
    layout="wide"
)


st.markdown("""
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


SYMPTOMS = [
    "fever","cough","sore throat","runny nose","body ache","fatigue",
    "headache","nausea","vomiting","burning urination","shortness of breath",
    "chest pain","dizziness","diarrhea","abdominal pain"
]

SPECIALTY = {
    "Influenza": "General Physician",
    "Allergic Rhinitis": "Allergist",
    "UTI": "Urologist",
    "Migraine": "Neurologist",
    "GERD": "Gastroenterologist",
    "Pneumonia": "Pulmonologist",
}

URGENCY_RULES = [
    ({"chest pain","shortness of breath"}, "🚨 Possible cardiac/respiratory emergency."),
    ({"severe headache","neck stiffness"}, "🚨 Possible meningitis pattern."),
    ({"burning urination","fever"}, "⚠️ Possible complicated UTI."),
]


def score_conditions(symptoms: set):
    candidates = {
        "Influenza": {"fever","cough","sore throat","body ache","fatigue","runny nose"},
        "Allergic Rhinitis": {"runny nose","sore throat","cough"},
        "UTI": {"burning urination","fever","abdominal pain"},
        "Migraine": {"headache","nausea","vomiting","dizziness"},
        "GERD": {"abdominal pain","nausea","vomiting"},
        "Pneumonia": {"fever","cough","shortness of breath","chest pain"},
    }
    out = []
    for disease, feats in candidates.items():
        overlap = len(symptoms & feats)
        if overlap == 0:
            continue

        conf = min(0.15 + 0.18*overlap, 0.95)
        out.append((disease, round(conf, 2), feats & symptoms))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:3]

def detect_urgency(symptoms: set):
    hits = []
    lower = {s.lower() for s in symptoms}
    for need, msg in URGENCY_RULES:
        if set(map(str.lower, need)).issubset(lower):
            hits.append(msg)
    return hits


c1, c2 = st.columns([0.75, 0.25])
with c1:
    st.title("🩺 Health Diagnosis")
    st.markdown(
        "<div class='banner'><b>About this project:</b> Enter symptoms → get the "
        "top 3 likely causes, why the model thinks so, and which doctor to visit. "
        "This page explains the model in an interactive way.</div>",
        unsafe_allow_html=True
    )
with c2:
    st.metric("Build status", "MVP", "v0.1")
    st.metric("Last updated", datetime.now().strftime("%d %b %Y"))

st.write("")


st.subheader("✨ What makes it useful")
colA, colB, colC, colD = st.columns(4)
with colA:
    st.markdown("**Top-3 predictions**")
    st.caption("Ranked with confidence and short explanations.")
with colB:
    st.markdown("**Doctor recommendation**")
    st.caption("Maps likely cause → specialty.")
with colC:
    st.markdown("**Red-flag checks**")
    st.caption("Rule-based warnings for urgent patterns.")
with colD:
    st.markdown("**Privacy-first**")
    st.caption("No data leaves your browser in the demo.")

st.write("")


st.subheader("🎛️ Try the explainer (demo)")
left, right = st.columns([0.58, 0.42])

with left:
    st.markdown("Select or type symptoms (comma separated).")
    chosen = st.multiselect("Symptoms", options=SYMPTOMS, default=["fever","cough"])
    free = st.text_input("Add more (comma separated)", placeholder="e.g., sore throat, body ache")
    # parse free text
    extra = [s.strip() for s in free.split(",") if s.strip()]
    all_syms = set([*chosen, *extra])

    go = st.button("Explain how the model would reason →", type="primary", use_container_width=True)

    if go:
        with st.spinner("Scoring symptom pattern…"):
            time.sleep(0.6)
        preds = score_conditions(all_syms)
        alerts = detect_urgency(all_syms)

        if alerts:
            st.error("  \n".join(alerts))

        if not preds:
            st.info("Not enough signal. Try adding 2–3 more symptoms.")
        else:
            for i, (cond, prob, why) in enumerate(preds, start=1):
                with st.container(border=True):
                    st.markdown(f"### #{i} — **{cond}**")
                    st.progress(int(prob*100))
                    st.caption(f"Confidence: **{int(prob*100)}%**")
                    st.markdown("**Why this appears:** " + ", ".join(sorted(why)) if why else "_feature overlap_")
                    st.markdown(f"**Doctor to visit:** `{SPECIALTY.get(cond, 'General Physician')}`")

with right:
    st.markdown("##### 📊 Example dataset glimpse")
    df = pd.DataFrame({
        "Condition":["Influenza","UTI","Migraine","GERD","Allergic Rhinitis","Pneumonia"],
        "Samples":[320,210,180,150,120,90]
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

st.write("")
st.divider()


st.subheader("🧠 How the model works")
t1, t2 = st.tabs(["Pipeline", "Safety & Limitations"])

with t1:
    st.markdown("""
**Pipeline (MVP):**
1. **Preprocess:** Tokenize comma-separated symptoms, normalize to a controlled vocabulary.
2. **Vectorize:** Multi-hot (CountVectorizer with `binary=True`).
3. **Classifier:** Logistic Regression / XGBoost (one-vs-rest) with calibrated probabilities.
4. **Explainability:** Highlight overlapping symptoms that contribute to each prediction.
5. **Triage rules:** Simple safety checks for urgent combinations (e.g., **chest pain + shortness of breath**).

**Why Top-3?**  
Healthcare is uncertain; top-3 gives users optionality and reduces over-confidence.
    """)

with t2:
    st.markdown("""
- **Not medical advice:** This is an educational demo and must not replace a clinician.
- **Data bias:** Public datasets may over-represent common conditions and under-represent rare ones.
- **Low-support labels:** Classes with very few samples are filtered or merged.
- **Privacy:** Avoid sending personal data to external services without consent.
    """)

st.write("")
st.divider()


st.subheader("❓ FAQ")
with st.expander("What data will you use?"):
    st.write("Kaggle/HuggingFace symptom→disease datasets, optionally augmented with synthetic patient records.")
with st.expander("How do you recommend a doctor?"):
    st.write("Each predicted condition maps to a specialty (e.g., UTI → Urologist). We include a safe fallback to General Physician.")
with st.expander("Can it detect emergencies?"):
    st.write("Red-flag rules fire if risky combinations are present. This is conservative and not a diagnosis.")
with st.expander("How do I integrate this page?"):
    st.markdown("""
- **Single page:** `streamlit run about.py`  
- **Multipage app:** put this as `pages/1_📖_About.py` and your main predictor as `Home.py`.
    """)

st.write("")
st.caption("© {} Health Diagnosis — Educational demo. Built with Streamlit.".format(datetime.now().year))
