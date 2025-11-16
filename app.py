import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Streamlit page configuration
st.set_page_config(page_title="Aussie Rain Predictor", page_icon="🌦️")

# Load trained model bundle (model, imputer, scaler, encoder, feature lists)
@st.cache_resource
def load_bundle(path: str = "models/aussie_rain.joblib"):
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

# Load dataset to extract min/max for numeric sliders and categories for selects
@st.cache_data
def load_data(path: str = "data/weatherAUS.csv"):
    return pd.read_csv(path)

try:
    df = load_data()
except Exception as e:
    st.error("Could not load data/weatherAUS.csv. Please check the path.")
    st.exception(e)
    st.stop()

# Numeric stats: min, max, median for each numeric feature
num_stats = {
    col: (
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].median())
    )
    for col in NUM
}

# Unique categorical values for each categorical feature
cat_values = {
    col: sorted(df[col].dropna().unique().tolist())
    for col in CAT
}

# Title / description
st.title("🌦️ Aussie Rain Predictor")
st.caption(
    "This application predicts whether it will rain tomorrow in Australia "
    "based on daily weather observations from the Australian Bureau of Meteorology."
)

with st.expander("Feature Description"):
    st.markdown("""
**Numeric Features**
- **MinTemp / MaxTemp** — Minimum and maximum temperature of the day (°C).
- **Rainfall** — Total rainfall measured (mm).
- **Evaporation** — Amount of water evaporated (mm).
- **Sunshine** — Hours of bright sunshine.
- **WindGustSpeed / WindSpeed9am / WindSpeed3pm** — Wind speeds (km/h).
- **Humidity9am / Humidity3pm** — Relative humidity (%).
- **Pressure9am / Pressure3pm** — Atmospheric pressure (hPa).
- **Cloud9am / Cloud3pm** — Cloud coverage (0–9 scale).
- **Temp9am / Temp3pm** — Temperature at 9am / 3pm (°C).

**Categorical Features**
- **Location** — Weather station location in Australia.
- **WindGustDir** — Direction of the strongest wind gust.
- **WindDir9am / WindDir3pm** — Wind direction at 9am / 3pm.
- **RainToday** — Whether it rained today ("Yes" / "No").

**Wind Direction Codes**
- **N** — North  
- **NNE** — North–North–East  
- **NE** — North–East  
- **ENE** — East–North–East  
- **E** — East  
- **ESE** — East–South–East  
- **SE** — South–East  
- **SSE** — South–South–East  
- **S** — South  
- **SSW** — South–South–West  
- **SW** — South–West  
- **WSW** — West–South–West  
- **W** — West  
- **WNW** — West–North–West  
- **NW** — North–West  
- **NNW** — North–North–West  

_All features come from the official WeatherAUS dataset._
    """)

# Randomization & Reset logic
if "inputs_initialized" not in st.session_state:
    st.session_state.inputs_initialized = True

    # Initialize numeric with medians
    for col in NUM:
        _, _, med = num_stats[col]
        st.session_state[col] = float(med)

    # Initialize categorical with default values 
    for col in CAT:
        options = cat_values[col]
        default_value = "No" if "No" in options else options[0]
        st.session_state[col] = default_value

# Buttons row
btn_col1, btn_col2 = st.columns(2)

# Randomize Inputs
if btn_col1.button("🎲 Randomize Inputs"):
    # Random numeric within min–max
    for col in NUM:
        mn, mx, _ = num_stats[col]
        st.session_state[col] = float(np.random.uniform(mn, mx))
    # Random categorical from known values
    for col in CAT:
        st.session_state[col] = str(np.random.choice(cat_values[col]))

# Reset Inputs
if btn_col2.button("🔄 Reset Inputs"):
    # Reset numeric to medians
    for col in NUM:
        _, _, med = num_stats[col]
        st.session_state[col] = float(med)
    # Reset categorical to default
    for col in CAT:
        options = cat_values[col]
        default_value = "No" if "No" in options else options[0]
        st.session_state[col] = default_value


# Input Form

st.header("Input Weather Data")

col_left, col_right = st.columns(2)

numeric_inputs = {}
for i, col in enumerate(NUM):
    min_val, max_val, _ = num_stats[col]
    step = (max_val - min_val) / 100 if max_val > min_val else 0.1

    with (col_left if i % 2 == 0 else col_right):
        numeric_inputs[col] = st.slider(
            label=col,
            min_value=min_val,
            max_value=max_val,
            value=st.session_state[col],
            step=step,
            key=col  # each feature uses its own key in session_state
        )

categorical_inputs = {}
for i, col in enumerate(CAT):
    options = [str(o) for o in cat_values[col]]

    # Current value stored in session_state[col]
    current_value = str(st.session_state[col])
    # Fallback in case current_value is not found in options
    index = options.index(current_value) if current_value in options else 0

    with (col_left if i % 2 == 0 else col_right):
        categorical_inputs[col] = st.selectbox(
            label=col,
            options=options,
            index=index,
            key=col
        )

# Preprocessing function (numeric imputation - scaling - OHE)
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

    # Impute numeric columns
    df_num_imputed = pd.DataFrame(
        imputer.transform(df_num),
        columns=NUM,
        index=df_in.index
    )

    # Scale numeric columns
    df_num_scaled = pd.DataFrame(
        scaler.transform(df_num_imputed),
        columns=NUM,
        index=df_in.index
    )

    # OHE categorical columns
    X_cat = encoder.transform(df_cat)
    if hasattr(X_cat, "toarray"):
        X_cat = X_cat.toarray()  # convert sparse to dense if needed

    # Concatenate numeric and categorical features
    X = np.hstack([df_num_scaled.values, X_cat])
    return X


# Predict button
if st.button("🔮 Predict RainTomorrow"):
    # Combine numeric and categorical inputs into one row
    row = {**numeric_inputs, **categorical_inputs}
    X_in = pd.DataFrame([row])

    try:
        X_ready = preprocess_row(X_in)
        prob = float(model.predict_proba(X_ready)[0, 1])
        pred = "Yes" if prob >= 0.5 else "No" # threshold = 0.5

        st.success(f"RainTomorrow prediction: **{pred}**")
        st.metric("Rain probability", f"{prob * 100:.1f}%")

        with st.expander("Input details"):
            st.json(row)

    except Exception as e:
        st.error("Error during preprocessing or model prediction.")
        st.exception(e)

