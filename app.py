import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------------
# Streamlit page configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Aussie Rain Predictor", page_icon="🌦️")

# -----------------------------------------------------------------------------
# Load trained model bundle (model, imputer, scaler, encoder, feature lists)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_bundle(path="models/aussie_rain.joblib"):
    """
    Load the saved model bundle that contains:
    - model
    - imputer   (numeric)
    - scaler    (numeric)
    - encoder   (categorical, OneHotEncoder)
    - numeric_cols
    - categorical_cols
    - input_cols
    """
    bundle = joblib.load(path)

    required_keys = [
        "model", "imputer", "scaler", "encoder",
        "numeric_cols", "categorical_cols", "input_cols"
    ]

    for key in required_keys:
        if key not in bundle:
            raise ValueError(f"Your joblib file is missing key: {key}")

    return bundle


bundle = load_bundle()
model   = bundle["model"]
imputer = bundle["imputer"]      # numeric imputer
scaler  = bundle["scaler"]       # numeric scaler
encoder = bundle["encoder"]      # OneHotEncoder
NUM     = list(bundle["numeric_cols"])
CAT     = list(bundle["categorical_cols"])

# -----------------------------------------------------------------------------
# Load dataset to extract min/max for numeric sliders and categories for selectboxes
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(path="data/weatherAUS.csv"):
    return pd.read_csv(path)

try:
    df = load_data()
except Exception as e:
    st.error("Could not load data/weatherAUS.csv. Please check the path.")
    st.exception(e)
    st.stop()

# Pre-calc numeric min/max/median
num_stats = {
    col: (
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].median())
    )
    for col in NUM
}

# Pre-calc categorical unique values
cat_values = {
    col: sorted(df[col].dropna().unique().tolist())
    for col in CAT
}

# -----------------------------------------------------------------------------
# Title / description
# -----------------------------------------------------------------------------
st.title("🌦️ Aussie Rain Predictor")
st.caption("Numeric: min–max sliders • Categorical: dropdown lists • Full preprocessing pipeline")

st.header("Input Weather Data")

# -----------------------------------------------------------------------------
# Build UI dynamically based on dataset statistics
# -----------------------------------------------------------------------------
col_left, col_right = st.columns(2)

numeric_inputs = {}
for i, col in enumerate(NUM):
    min_val, max_val, med_val = num_stats[col]
    step = (max_val - min_val) / 100 if max_val > min_val else 0.1

    with (col_left if i % 2 == 0 else col_right):
        numeric_inputs[col] = st.slider(
            label=col,
            min_value=min_val,
            max_value=max_val,
            value=med_val,
            step=step
        )

categorical_inputs = {}
for i, col in enumerate(CAT):
    options = [str(o) for o in cat_values[col]]
    default_index = options.index("No") if "No" in options else 0

    with (col_left if i % 2 == 0 else col_right):
        categorical_inputs[col] = st.selectbox(
            label=col,
            options=options,
            index=default_index
        )

# -----------------------------------------------------------------------------
# Preprocessing function (numeric imputation → scaling → OHE)
# -----------------------------------------------------------------------------
def preprocess_row(df_in: pd.DataFrame) -> np.ndarray:
    """
    Preprocess input row using training-time transformations:
    1. Extract numeric and categorical columns
    2. Impute numeric values
    3. Scale numeric values
    4. One-hot encode categorical values
    5. Concatenate into final feature vector for the model
    """
    df_num = df_in[NUM]
    df_cat = df_in[CAT]

    # 1) Impute numeric columns
    df_num_imputed = pd.DataFrame(
        imputer.transform(df_num),
        columns=NUM,
        index=df_in.index
    )

    # 2) Scale numeric columns
    df_num_scaled = pd.DataFrame(
        scaler.transform(df_num_imputed),
        columns=NUM,
        index=df_in.index
    )

    # 3) One-hot encode categorical columns
    X_cat = encoder.transform(df_cat)
    if hasattr(X_cat, "toarray"):
        X_cat = X_cat.toarray()  # convert sparse to dense if needed

    # 4) Final matrix
    X = np.hstack([df_num_scaled.values, X_cat])
    return X

# -----------------------------------------------------------------------------
# Predict button
# -----------------------------------------------------------------------------
if st.button("🔮 Predict RainTomorrow"):
    row = {**numeric_inputs, **categorical_inputs}
    X_in = pd.DataFrame([row])

    try:
        X_ready = preprocess_row(X_in)
        prob = float(model.predict_proba(X_ready)[0, 1])
        pred = "Yes" if prob >= 0.5 else "No"

        st.success(f"RainTomorrow prediction: **{pred}**")
        st.metric("Rain probability", f"{prob*100:.1f}%")

        with st.expander("Input details"):
            st.json(row)

    except Exception as e:
        st.error("Error during preprocessing or model prediction.")
        st.exception(e)
