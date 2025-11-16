import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Aussie Rain", page_icon="🌦️")

@st.cache_resource
def load_bundle(path="models/aussie_rain.joblib"):
    b = joblib.load(path)
    need = ["model", "imputer", "scaler", "encoder",
            "numeric_cols", "categorical_cols", "input_cols"]
    for k in need:
        if k not in b:
            raise ValueError(f"У joblib бракує ключа: {k}")
    return b

bundle = load_bundle()
model   = bundle["model"]
imputer = bundle["imputer"]      # <= імпутер ТІЛЬКИ для числових ознак
scaler  = bundle["scaler"]
encoder = bundle["encoder"]
NUM     = list(bundle["numeric_cols"])
CAT     = list(bundle["categorical_cols"])
INPUT_COLS = list(bundle["input_cols"])   # просто для інформації, далі не критично

st.title("🌦️ Чи піде дощ завтра?")
st.caption("Імпутація (NUM) → масштабування (NUM) → OHE (CAT) → модель")

# ----- Форма вводу -----
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
WindDir9am  = st.text_input("WindDir9am", value="N")
WindDir3pm  = st.text_input("WindDir3pm", value="N")
RainToday   = st.selectbox("RainToday", ["No", "Yes"], index=0)

# ----- ВАЖЛИВА ЧАСТИНА: ПРЕПРОЦЕСИНГ -----
def preprocess_row(df: pd.DataFrame) -> np.ndarray:
    """
    1) беремо окремо NUM і CAT;
    2) імпутуємо ТІЛЬКИ NUM (як було на тренуванні);
    3) масштабуємо NUM;
    4) OHE для CAT;
    5) конкатенуємо в одну матрицю X.
    """
    # 1. розділяємо
    df_num = df[NUM]
    df_cat = df[CAT]

    # 2. імпутація числових
    df_num_imp = pd.DataFrame(
        imputer.transform(df_num),
        columns=NUM,
        index=df.index,
    )

    # 3. масштабування числових
    df_num_scaled = pd.DataFrame(
        scaler.transform(df_num_imp),
        columns=NUM,
        index=df.index,
    )

    # 4. OHE для категоріальних
    X_cat = encoder.transform(df_cat)
    if hasattr(X_cat, "toarray"):
        X_cat = X_cat.toarray()

    # 5. збірка кінцевого X
    X = np.hstack([df_num_scaled.values, X_cat])
    return X

# ----- Кнопка прогнозу -----
if st.button("🔮 Прогнозувати"):
    row = {
        "MinTemp": MinTemp, "MaxTemp": MaxTemp, "Rainfall": Rainfall,
        "Evaporation": Evaporation, "Sunshine": Sunshine,
        "WindGustSpeed": WindGustSpeed, "WindSpeed9am": WindSpeed9am,
        "WindSpeed3pm": WindSpeed3pm, "Humidity9am": Humidity9am,
        "Humidity3pm": Humidity3pm, "Pressure9am": Pressure9am,
        "Pressure3pm": Pressure3pm, "Cloud9am": Cloud9am,
        "Cloud3pm": Cloud3pm, "Temp9am": Temp9am, "Temp3pm": Temp3pm,
        "Location": Location, "WindGustDir": WindGustDir,
        "WindDir9am": WindDir9am, "WindDir3pm": WindDir3pm,
        "RainToday": RainToday
    }

    X_in = pd.DataFrame([row])

    try:
        X_ready = preprocess_row(X_in)
        proba = float(model.predict_proba(X_ready)[0, 1])
        pred = int(proba >= 0.5)
        st.success(f"RainTomorrow: **{'Yes' if pred else 'No'}**")
        st.metric("Ймовірність дощу", f"{proba*100:.1f}%")
    except Exception as e:
        st.error("Помилка під час препроцесингу/інференсу.")
        st.exception(e)
