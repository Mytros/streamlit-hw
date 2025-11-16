# -----------------------------------------------------------------------------
# Randomization logic using Streamlit session_state
# -----------------------------------------------------------------------------

# Initialize session state on first load
if "inputs_initialized" not in st.session_state:
    st.session_state.inputs_initialized = True
    # Initialize numeric values with medians
    for col in NUM:
        _, _, med = num_stats[col]
        st.session_state[col] = med
    # Initialize categorical values with default first option
    for col in CAT:
        st.session_state[col] = cat_values[col][0]

# Randomize button
if st.button("🎲 Randomize Inputs"):
    # Random numeric
    for col in NUM:
        mn, mx, _ = num_stats[col]
        st.session_state[col] = float(np.random.uniform(mn, mx))
    # Random categorical
    for col in CAT:
        st.session_state[col] = np.random.choice(cat_values[col])

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
            key=col
        )

categorical_inputs = {}
for i, col in enumerate(CAT):
    options = [str(o) for o in cat_values[col]]

    with (col_left if i % 2 == 0 else col_right):
        categorical_inputs[col] = st.selectbox(
            label=col,
            options=options,
            index=options.index(str(st.session_state[col])),
            key=col
        )
