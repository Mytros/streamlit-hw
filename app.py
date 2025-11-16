import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Aussie Rain", page_icon="🌦️")

# ---------- 1. Завантаження моделі та препроцесингу ----------
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
imputer = bundle["imputer"]      # імп’ютер для числових
scaler  = bundle["scaler"]
encoder = bundle["encoder"]
NUM     = list(bundle["numeric_cols"])
CAT     = list(bundle["categorical_cols"])

# ---------- 2. Завантаження датасету для мін/макс/категорій ----------
@st.cache_data
def load_data(path="data/weatherAUS.csv"):
    return pd.read_csv(path)

try:
    df = load_data()
except Exception as e:
    st.error("Не вдалося прочитати data/weatherAUS.csv. "
             "Перевір шлях до файлу (data/weatherAUS.csv).")
    st.exception(e)
    st.stop()

# словнички: статистики для числових та унікальні значення для категоріальних
num_stats = {
    col: (
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].median())
    )
    for col in NUM
}

cat_values = {
    col: sorted(df[col].dropna().unique().tolist())
    for col in CAT
}

st.title("🌦️ Чи піде дощ завтра?")
st.caption("Імпутація (NUM) → масштабування (NUM) → OHE (CAT) → модель")

# ---------- 3. Форма вводу на основі датасету ----------
st.header("Ввід даних з датасету")

cols = st.columns(2)

numeric_inputs = {}
for i, col in enumerate(NUM):
    mn, mx, med = num_stats[col]
    # невеликий крок для слайдера
    step = (mx - mn) / 100 if mx > mn else 0.1
    with cols[i % 2]:
        numeric_inputs[col] = st.slider(
            col,
            min_value=mn,
            max_value=mx,
            value=med,
            step=step
        )

categorical_inputs = {}
for i, col in enumerate(CAT):
    options = [str(o) for o in cat_values[col]]
    # якщо є "No" — ставимо її дефолтною
    default_idx = 0
    if "No" in options:
        default_idx = options.index("No")
    with cols[i % 2]:
        categorical_inputs[col] = st.selectbox(col, options=options, index=default_idx)

# ---------- 4. Препроцесинг (такий самий, як раніше) ----------
def preprocess_row(df_in: pd.DataFrame) -> np.ndarray:
    """
    1) NUM і CAT окремо;
    2) імпутація тільки NUM;
    3) масштабування NUM;
    4) OHE CAT;
    5) конкатенація.
    """
    df_num = df_in[NUM]
    df_cat = df_in[CAT]

    # імпутація числових
    df_num_imp = pd.DataFrame(
        imputer.transform(df_num),
        columns=NUM,
        index=df_in.index,
    )

    # масштабування числових
    df_num_scaled = pd.DataFrame(
        scaler.transform(df_num_imp),
        columns=NUM,
        index=df_in.index,
    )

    # OHE для категоріальних
    X_cat = encoder.transform(df_cat)
    if hasattr(X_cat, "toarray"):
        X_cat = X_cat.toarray()

    X = np.hstack([df_num_scaled.values, X_cat])
    return X

# ---------- 5. Прогноз ----------
if st.button("🔮 Прогнозувати"):
    # один рядок з усіма фічами
    row = {**numeric_inputs, **categorical_inputs}
    X_in = pd.DataFrame([row])

    try:
        X_ready = preprocess_row(X_in)
        proba = float(model.predict_proba(X_ready)[0, 1])
        pred = int(proba >= 0.5)

        st.success(f"RainTomorrow: **{'Yes' if pred else 'No'}**")
        st.metric("Ймовірність дощу", f"{proba*100:.1f}%")

        with st.expander("Введені значення"):
            st.json(row)

    except Exception as e:
        st.error("Помилка під час препроцесингу/інференсу.")
        st.exception(e)
