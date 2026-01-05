import re

import joblib
import pandas as pd
import unicodedata

import streamlit as st

def preprocess_slovak_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text).lower()

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    final_text_list = text.split()

    return " ".join(final_text_list)


st.set_page_config(page_title="Column Normalizer", page_icon="📊")


@st.cache_resource
def load_my_model(file):
    return joblib.load(file)


model = load_my_model("model.pkl")
model_preprocessed = load_my_model("model_preprocessed.pkl")

st.title("📊 Column Normalizer")
st.markdown("Enter a column name to see its standardized mapping.")

user_input = st.text_input("Column Name", placeholder="e.g., cena bytu")

if user_input:
    probs = model_preprocessed.predict_proba([user_input])[0]
    best_idx = probs.argmax()
    prediction = model.classes_[best_idx]
    confidence = probs[best_idx]

    st.success(f"Predicted Category: **{prediction}**")
    st.progress(float(confidence))
    st.write(f"Confidence: {confidence:.2%}")

st.title("📊 Column Normalizer Preprocessed")
st.markdown("Enter a column name to see its standardized mapping.")

user_input_clear = st.text_input("Column Name (preprocessed model)",
                                 placeholder="e.g., cena bytu")

if user_input_clear:
    user_input_preprocessed = preprocess_slovak_text(user_input_clear)
    probs = model_preprocessed.predict_proba([user_input_preprocessed])[0]
    best_idx = probs.argmax()
    prediction = model_preprocessed.classes_[best_idx]
    confidence = probs[best_idx]

    st.success(f"Predicted Category: **{prediction}**")
    st.progress(float(confidence))
    st.write(f"Confidence: {confidence:.2%}")


st.subheader("Test results")

dummy_data = pd.DataFrame(
    {
        "Normal": ["73.16 %", "93.16 %"],
        "Preprocessed": ["75.66 %", "93.65 %"],
    },
    index=["Top-1 accuracy", "Top-3 accuracy"]

)

st.table(dummy_data)

st.subheader("Model description")

model_description = """Model: Použili sme logistickú regresiu kvôli rýchlosti a stabilite. Vstupné dáta sme zmenili pomocou TF-IDF a následne ich zabalili do Pipeline.

Kalibrácia: Použili sme kalibráciu (sigmoid metóda a 3-násobná cross-validácia), aby výsledok lepšie zodpovedal realite pri "noisy" dátach a určení top-k presnosti.

Trénovanie: Trénujeme na čistých aj noisy dátach pre väčšiu robustnosť a testujeme na čistých dátach, aby hodnotenie ostalo férové.

Metriky: Pozeráme na správnosť modelu pri top-1 accuracy a zároveň aj top-3 accuracy."""
st.write(model_description)
