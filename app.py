import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Aussie Rain (без пайплайна)", page_icon="🌦️")

# --- 1) Завантаження бандла з артефактами ---
@st.cache_resource
def load_bundle(path="models/aussie_rain.joblib"):
    obj = joblib.load(path)
    if not isinstance(obj, dict):
        raise ValueError("Очікую dict з ключами: model, imputer_num, imputer_cat, scaler, ohe, num_features, cat_features.")
    need = ["model","imputer_num","imputer_cat","scaler","ohe","num_features","cat_features"]
    for k in need:
        if k not in obj:
            raise ValueError(f"У joblib бракує ключа: {k}")
    return obj

bundle = load_bundle()

model = bundle["model"]
imp_num = bundle["imputer_num"]
imp_cat = bundle["imputer_cat"]
scaler  = bundle["scaler"]
ohe     = bundle["ohe"]
NUM_FEATS = list(bundle["num_features"])
CAT_FEATS = list(bundle["cat_features"])
FEAT_ORDER = bundle.get("feature_order")  # опційно

st.title("🌦️ Чи піде дощ завтра?")
st.caption("RandomForest + ручний препроцесинг: імпутація → масштабування → OHE")

# --- 2) Проста форма вводу (як у твоєму Iris-прикладі) ---
st.header("Ввід даних")
c1, c2 = st.columns(2)

with c1:
    MinTemp = st.number_input("MinTemp", value=10.0, step=0.1)
    MaxTemp = st.number_input("MaxTemp", value=20.0, step=0.1)
    Rainfall = st.number_input("Rainfall", value=0.0, step=0.1)
    Evaporation = st.number_input("Evaporation", value=5.0, step=0.1)
    Sunshine = st.number_input("Sunshine", value=7.0, step=0.1)
    WindGustSpeed = st.number_input("WindGustSpeed", value=40.0, step=1.0)
    WindSpeed9am = st.number_input("WindSpeed9am", value=10.0, step=1.0)
    WindSpeed3pm = st.number_input("WindSpeed3pm", value=15.0, step=1.0)

with c2:
    Humidity9am = st.number_input("Humidity9am", value=70.0, step=1.0)
    Humidity3pm = st.number_input("Humidity3pm", value=50.0, step=1.0)
    Pressure9am = st.number_input("Pressure9am", value=1015.0, step=0.1)
    Pressure3pm = st.number_input("Pressure3pm", value=1012.0, step=0.1)
    Cloud9am = st.number_input("Cloud9am (0–9)", value=4.0, step=1.0)
    Cloud3pm = st.number_input("Cloud3pm (0–9)", value=4.0, step=1.0)
    Temp9am = st.number_input("Temp9am", value=16.0, step=0.1)
    Temp3pm = st.number_input("Temp3pm", value=18.0, step=0.1)

st.subheader("Категоріальні")
Location = st.text_input("Location", value="Sydney")
WindGustDir = st.text_input("WindGustDir", value="N")
WindDir9am = st.text_input("WindDir9am", value="N")
WindDir3pm = st.text_input("WindDir3pm", value="N")
RainToday = st.selectbox("RainToday", ["No","Yes"], index=0)

# --- 3) Кнопка прогнозу ---
def preprocess_row(row_df: pd.DataFrame) -> pd.DataFrame:
    """Імпутація → масштабування → OHE → конкат → вирівнювання порядку."""
    # Імпутація
    if NUM_FEATS:
        row_df[NUM_FEATS] = imp_num.transform(row_df[NUM_FEATS])
    if CAT_FEATS:
        row_df[CAT_FEATS] = imp_cat.transform(row_df[CAT_FEATS])

    # Масштабування числових
    if NUM_FEATS:
        row_df[NUM_FEATS] = scaler.transform(row_df[NUM_FEATS])

    # One-Hot для категорійних
    if CAT_FEATS:
        cat_mat = ohe.transform(row_df[CAT_FEATS])
        if hasattr(cat_mat, "toarray"):
            cat_mat = cat_mat.toarray()
        # назви ohe-колонок
        if hasattr(ohe, "get_feature_names_out"):
            cat_cols = list(ohe.get_feature_names_out(CAT_FEATS))
        else:
            # запасний варіант
            cat_cols = [f"{c}_{i}" for c in CAT_FEATS for i in range(cat_mat.shape[1])]
        cat_df = pd.DataFrame(cat_mat, columns=cat_cols, index=row_df.index)
    else:
        cat_df = pd.DataFrame(index=row_df.index)

    X_num = row_df[NUM_FEATS] if NUM_FEATS else pd.DataFrame(index=row_df.index)
    X = pd.concat([X_num, cat_df], axis=1)

    # Вирівнювання порядку (якщо збережено)
    if FEAT_ORDER:
        for col in FEAT_ORDER:
            if col not in X.columns:
                X[col] = 0.0  # нових категорій не було під час тренування
        X = X[FEAT_ORDER]
    return X

if st.button("🔮 Прогнозувати тип погоди"):
    # Один рядок з іменами колонок точнісінько як на тренуванні
    row = {
        "MinTemp": MinTemp, "MaxTemp": MaxTemp, "Rainfall": Rainfall, "Evaporation": Evaporation, "Sunshine": Sunshine,
        "WindGustSpeed": WindGustSpeed, "WindSpeed9am": WindSpeed9am, "WindSpeed3pm": WindSpeed3pm,
        "Humidity9am": Humidity9am, "Humidity3pm": Humidity3pm,
        "Pressure9am": Pressure9am, "Pressure3pm": Pressure3pm,
        "Cloud9am": Cloud9am, "Cloud3pm": Cloud3pm,
        "Temp9am": Temp9am, "Temp3pm": Temp3pm,
        "Location": Location, "WindGustDir": WindGustDir, "WindDir9am": WindDir9am, "WindDir3pm": WindDir3pm,
        "RainToday": RainToday
    }
    X_row = pd.DataFrame([row])

    try:
        X_ready = preprocess_row(X_row.copy())
        proba = model.predict_proba(X_ready)[0, 1]
        pred = int(proba >= 0.5)
        st.success(f"RainTomorrow: **{'Yes' if pred else 'No'}**")
        st.metric("Ймовірність дощу", f"{proba*100:.1f}%")
    except Exception as e:
        st.error("Помилка під час препроцесингу/інференсу. Перевір відповідність артефактів.")
        st.exception(e)
