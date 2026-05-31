import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Concrete AI Assistant",
    page_icon="🏗️",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

.block-container {
    padding-top: 1rem;
}

.metric-card {
    background-color: white;
    padding: 1rem;
    border-radius: 10px;
}

h1, h2, h3 {
    color: #1f2937;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    pump_model = joblib.load(
        "models/pumpability_classifier.pkl"
    )

    strength_model = joblib.load(
        "models/strength_regressor.pkl"
    )

    metadata = joblib.load(
        "models/project_metadata.pkl"
    )

    cls_importance = joblib.load(
        "models/classification_importance.pkl"
    )

    reg_importance = joblib.load(
        "models/strength_importance.pkl"
    )

    strength_features = joblib.load(
    "models/strength_features.pkl"
    
    )

    return (
        pump_model,
        strength_model,
        metadata,
        cls_importance,
        reg_importance,
        strength_features
    )


(
    pump_model,
    strength_model,
    metadata,
    cls_importance,
    reg_importance,
    strength_features
) = load_models()

# =====================================================
# FEATURE ENGINEERING
# =====================================================

def create_features(
    cement,
    fine,
    coarse,
    water,
    wra,
    flyash,
    accelerator,
    silica
):

    binder = (
        cement +
        flyash +
        silica
    )

    w_c_ratio = (
        water / cement
        if cement > 0 else 0
    )

    fine_ratio = (
        fine /
        (fine + coarse)
        if (fine + coarse) > 0 else 0
    )

    paste_content = (
        cement +
        water +
        flyash +
        silica
    )

    wra_ratio = (
        wra / cement
        if cement > 0 else 0
    )

    return {

        "Cement": cement,
        "FineAggr": fine,
        "CoarseAggr": coarse,
        "Water": water,
        "WRA": wra,
        "FlyAsh": flyash,
        "Accelerator": accelerator,
        "SilicaFume": silica,

        "W_C_Ratio": w_c_ratio,
        "Fine_Ratio": fine_ratio,
        "WRA_Ratio": wra_ratio,
        "Binder": binder,
        "Paste_Content": paste_content
    }

def generate_random_mix():

    mix = {

        feature: np.random.uniform(
            low,
            high
        )

        for feature, (low, high)
        in MIX_RANGES.items()
    }

    return mix

def is_valid_mix(features):

    if not (
        0.30 <= features["W_C_Ratio"] <= 0.65
    ):
        return False

    if not (
        0.35 <= features["Fine_Ratio"] <= 0.60
    ):
        return False

    if not (
        300 <= features["Binder"] <= 600
    ):
        return False

    return True

# =====================================================
# MIX RANGES FOR RANDOM GENERATION
# =====================================================

MIX_RANGES = {
    "Cement": (250, 550),
    "FineAggr": (500, 1000),
    "CoarseAggr": (700, 1300),
    "Water": (130, 230),
    "WRA": (0, 15),
    "FlyAsh": (0, 150),
    "Accelerator": (0, 10),
    "SilicaFume": (0, 50)
}

def predict_strength_from_mix(
    mix,
    age=28
):

    features = create_features(
        mix["Cement"],
        mix["FineAggr"],
        mix["CoarseAggr"],
        mix["Water"],
        mix["WRA"],
        mix["FlyAsh"],
        mix["Accelerator"],
        mix["SilicaFume"]
    )

    features["Time"] = age

    X = pd.DataFrame([features])
    strength = strength_model.predict(X[strength_features])[0]

    return strength

def generate_candidate_mixes(
    n_samples=5000
):

    mixes = []

    while len(mixes) < n_samples:

        mix = generate_random_mix()

        features = create_features(
            mix["Cement"],
            mix["FineAggr"],
            mix["CoarseAggr"],
            mix["Water"],
            mix["WRA"],
            mix["FlyAsh"],
            mix["Accelerator"],
            mix["SilicaFume"]
        )

        if is_valid_mix(features):

            mixes.append(mix)

    return mixes

def evaluate_candidate_mixes(
    candidate_mixes,
    age=28
):

    results = []

    for mix in candidate_mixes:

        strength = predict_strength_from_mix(
            mix,
            age
        )

        row = mix.copy()

        row["Predicted_Strength"] = strength

        results.append(row)

    return pd.DataFrame(results)

def find_best_mixes(
    target_strength,
    age=28,
    n_designs=5,
    n_candidates=5000
):

    candidate_mixes = generate_candidate_mixes(
        n_candidates
    )

    results_df = evaluate_candidate_mixes(
        candidate_mixes,
        age
    )

    results_df["Error"] = (
        results_df["Predicted_Strength"]
        -
        target_strength
    ).abs()

    best_designs = (
        results_df
        .sort_values("Error")
        .head(n_designs)
        .reset_index(drop=True)
    )

    return best_designs

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.subheader("MIX DESIGN INPUTS")

cement = st.sidebar.slider(
    "Cement (kg/m³)",
    100,
    700,
    350
)

water = st.sidebar.slider(
    "Water (kg/m³)",
    100,
    300,
    180
)

fine = st.sidebar.slider(
    "Fine Aggregate (kg/m³)",
    300,
    1200,
    750
)

coarse = st.sidebar.slider(
    "Coarse Aggregate (kg/m³)",
    300,
    1500,
    1000
)

wra = st.sidebar.slider(
    "WRA (kg/m³)",
    0.0,
    20.0,
    5.0
)

flyash = st.sidebar.slider(
    "Fly Ash (kg/m³)",
    0,
    250,
    0
)

accelerator = st.sidebar.slider(
    "Accelerator (kg/m³)",
    0.0,
    20.0,
    0.0
)

silica = st.sidebar.slider(
    "Silica Fume (kg/m³)",
    0,
    100,
    0
)

# Add QUICK INFO section
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ QUICK INFO")

with st.sidebar.info("""
**Adjust the mix design values using the sliders and click 'Predict Pumpability' to evaluate the concrete mix.**

- **Green**: Within recommended range
- **Orange/Red**: Outside recommended range
- **Pumpable**: High probability of successful placement
"""):
    pass

# =====================================================
# HOME HEADER
# =====================================================

st.markdown("""
<div style='text-align:center'>
<h1>🏗️ Concrete AI Assistant</h1>
<p>
AI-Based Pumpability Classification & Strength Prediction
</p>
</div>
""",
unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=[
        "Pumpability",
        "Strength",
        "AI Mix Design",
        "Model Insights"
        
    ],
    icons=[
        "droplet-half",
        "bar-chart",
        "cpu-fill",
        "book"
    ],
    orientation="horizontal",
    default_index=0
)
page = selected

col1,col2,col3,col4 = st.columns(4)
# =====================================================
# PUMPABILITY
# =====================================================

def create_pumpability_status_df(features):

    checks = [

        {
            "Parameter": "W/C Ratio",
            "Value": features["W_C_Ratio"],
            "Min": 0.35,
            "Max": 0.55
        },

        {
            "Parameter": "Fine Ratio",
            "Value": features["Fine_Ratio"],
            "Min": 0.40,
            "Max": 0.60
        },

        {
            "Parameter": "Binder",
            "Value": features["Binder"],
            "Min": 300,
            "Max": 550
        },

        {
            "Parameter": "Paste Content",
            "Value": features["Paste_Content"],
            "Min": 450,
            "Max": 700
        }

    ]

    df = pd.DataFrame(checks)

    df["Status"] = np.where(

        (df["Value"] >= df["Min"]) &
        (df["Value"] <= df["Max"]),

        "OK",

        "Out of Range"
    )

    return df

if page == "Pumpability":

    st.subheader("Pumpability Classification")

    # Get features
    features = create_features(
        cement,
        fine,
        coarse,
        water,
        wra,
        flyash,
        accelerator,
        silica
    )

    X = pd.DataFrame([features])

    # Display metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "W/C Ratio",
            f"{features['W_C_Ratio']:.3f}"
        )

    with col2:
        st.metric(
            "Binder (kg/m³)",
            f"{features['Binder']:.1f}"
        )

    with col3:
        st.metric(
            "Paste Content (kg/m³)",
            f"{features['Paste_Content']:.1f}"
        )

    with col4:
        st.metric(
            "WRA Ratio (%)",
            f"{features['WRA_Ratio']*100:.2f}"
        )

    # Make prediction
    if st.button(
        "Predict Pumpability",
        use_container_width=True
    ):
        pred = pump_model.predict(X)[0]
        prob = pump_model.predict_proba(X)[0][1]
        
        # Store in session state for persistence
        st.session_state.pred = pred
        st.session_state.prob = prob
        st.session_state.features = features
    
    # Display results if available
    if hasattr(st.session_state, 'prob'):
        
        # Engineering Parameter Gauges
        st.subheader("Engineering Parameter Gauges")
        
        gauge_col1, gauge_col2, gauge_col3, gauge_col4 = st.columns(4)
        
        with gauge_col1:
            fig_wc = go.Figure(go.Indicator(
                mode="gauge+number",
                value=st.session_state.features['W_C_Ratio'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "W/C Ratio"},
                gauge={'axis': {'range': [0, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 0.35], 'color': "#ff6b6b"},
                           {'range': [0.35, 0.55], 'color': "#51cf66"},
                           {'range': [0.55, 1], 'color': "#ff6b6b"}
                       ]}
            ))
            st.plotly_chart(fig_wc, use_container_width=True)
        
        with gauge_col2:
            fig_fine = go.Figure(go.Indicator(
                mode="gauge+number",
                value=st.session_state.features['Fine_Ratio'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fine Ratio"},
                gauge={'axis': {'range': [0, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 0.40], 'color': "#ff6b6b"},
                           {'range': [0.40, 0.60], 'color': "#51cf66"},
                           {'range': [0.60, 1], 'color': "#ff6b6b"}
                       ]}
            ))
            st.plotly_chart(fig_fine, use_container_width=True)
        
        with gauge_col3:
            fig_binder = go.Figure(go.Indicator(
                mode="gauge+number",
                value=st.session_state.features['Binder'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Binder (kg/m³)"},
                gauge={'axis': {'range': [0, 800]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 300], 'color': "#ff6b6b"},
                           {'range': [300, 550], 'color': "#51cf66"},
                           {'range': [550, 800], 'color': "#ff6b6b"}
                       ]}
            ))
            st.plotly_chart(fig_binder, use_container_width=True)
        
        with gauge_col4:
            fig_paste = go.Figure(go.Indicator(
                mode="gauge+number",
                value=st.session_state.features['Paste_Content'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Paste Content (kg/m³)"},
                gauge={'axis': {'range': [0, 900]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 450], 'color': "#ff6b6b"},
                           {'range': [450, 700], 'color': "#51cf66"},
                           {'range': [700, 900], 'color': "#ff6b6b"}
                       ]}
            ))
            st.plotly_chart(fig_paste, use_container_width=True)

        # Overall Prediction
        st.subheader("Overall Prediction")
        
        pred_col1, pred_col2 = st.columns([1, 2])
        
        with pred_col1:
            prob_display = st.session_state.prob * 100
            st.metric("Pumpability Probability", f"{prob_display:.1f}%")
            
            if st.session_state.pred == 1:
                st.success("🟢 Pumpable")
            else:
                st.error("🔴 Not Pumpable")
        
        with pred_col2:
            st.progress(float(st.session_state.prob))
        
        # Engineering Checks Summary
        st.subheader("Engineering Checks Summary")
        
        status_df = create_pumpability_status_df(st.session_state.features)
        
        # Add deviation column
        status_df['Deviation'] = status_df.apply(
            lambda row: '–' if row['Status'] == 'OK' 
            else f"{abs(row['Value'] - row['Min'] if row['Value'] < row['Min'] else row['Value'] - row['Max']):.2f}",
            axis=1
        )
        
        # Rename columns for display
        display_df = status_df.rename(columns={
            'Parameter': 'Parameter',
            'Value': 'Current Value',
            'Min': 'Recommended Range',
            'Max': '',
            'Status': 'Status',
            'Deviation': 'Deviation'
        })
        
        # Format the range display
        status_df['Range'] = status_df.apply(
            lambda row: f"{row['Min']:.2f} – {row['Max']:.2f}" 
            if isinstance(row['Min'], (int, float)) and isinstance(row['Max'], (int, float))
            else f"{row['Min']} – {row['Max']}",
            axis=1
        )
        
        display_df_final = pd.DataFrame({
            'Parameter': status_df['Parameter'],
            'Current Value': status_df['Value'].apply(lambda x: f"{x:.3f}" if isinstance(x, float) else x),
            'Recommended Range': status_df['Range'],
            'Status': status_df['Status'],
            'Deviation': status_df['Deviation']
        })
        
        st.dataframe(display_df_final, use_container_width=True, hide_index=True)
# =====================================================
# STRENGTH
# =====================================================

elif page == "Strength":

    st.header(
        "Compressive Strength Prediction"
    )

    time = st.slider(
        "Curing Time (Days)",
        1,
        365,
        28
    )

    features = create_features(
        cement,
        fine,
        coarse,
        water,
        wra,
        flyash,
        accelerator,
        silica
    )

    features["Time"] = time

    X = pd.DataFrame([features])

    if st.button(
        "Predict Strength",
        use_container_width=True
    ):

        strength = strength_model.predict(
            X[strength_features]
        )[0]

        st.metric(
            "Predicted Strength",
            f"{strength:.2f} MPa"
        )

        if strength < 25:

            st.warning(
                "Low Strength Concrete"
            )

        elif strength < 50:

            st.success(
                "Normal Structural Concrete"
            )

        else:

            st.info(
                "High Strength Concrete"
            )

# =====================================================
# AI MIX DESIGN
# =====================================================

elif page == "AI Mix Design":

    st.header(
        "AI Mix Design Generator"
    )

    target_strength = st.slider(
        "Target Strength (MPa)",
        10,
        75,
        40
    )

    age = st.slider(
        "Age (Days)",
        1,
        365,
        28
    )

    if st.button(
        "Generate Mix Designs",
        use_container_width=True
    ):

        with st.spinner(
            "Generating optimal mix designs..."
        ):

            designs = find_best_mixes(
                target_strength=target_strength,
                age=age,
                n_designs=5
            )

        st.success(
            f"{len(designs)} candidate mix designs generated."
        )

        cols = st.columns(2)

        for i, (_, row) in enumerate(
            designs.iterrows()
        ):

            with cols[i % 2]:

                st.markdown(
                    f"""
                    ### 🏗 Mix Design #{i+1}
                    """
                )

                c1, c2 = st.columns(2)

                c1.metric(
                    "Strength",
                    f"{row['Predicted_Strength']:.2f} MPa"
                )

                c2.metric(
                    "Error",
                    f"{row['Error']:.3f}"
                )

                st.dataframe(
                    pd.DataFrame(
                        {
                            "Material":[
                                "Cement",
                                "Water",
                                "FineAggr",
                                "CoarseAggr",
                                "FlyAsh",
                                "SilicaFume",
                                "WRA"
                            ],
                            "kg/m³":[
                                round(row["Cement"],1),
                                round(row["Water"],1),
                                round(row["FineAggr"],1),
                                round(row["CoarseAggr"],1),
                                round(row["FlyAsh"],1),
                                round(row["SilicaFume"],1),
                                round(row["WRA"],1)
                            ]
                        }
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("---")
# =====================================================
# MODEL INFO
# =====================================================

elif page == "Model Insights":

    st.header("Model Information")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Pumpability Classifier"
        )

        st.metric(
            "Model",
            metadata["classification"]["model_name"]
        )

        st.metric(
            "Best F1",
            f"{metadata['classification']['best_f1']:.3f}"
        )

    with col2:

        st.subheader(
            "Strength Regressor"
        )

        st.metric(
            "Model",
            metadata["regression"]["model_name"]
        )

        st.metric(
            "Best R²",
            f"{metadata['regression']['r2']:.3f}"
        )

    st.markdown("---")

    st.subheader(
        "Classification Feature Importance"
    )

    fig_cls = px.bar(
        cls_importance,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(
        fig_cls,
        use_container_width=True
    )

    st.subheader(
        "Strength Feature Importance"
    )

    fig_reg = px.bar(
        reg_importance,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(
        fig_reg,
        use_container_width=True
    )

    with st.expander(
        "Show Raw Feature Importance Tables"
    ):

        st.dataframe(
            cls_importance,
            use_container_width=True
        )

        st.dataframe(
            reg_importance,
            use_container_width=True
        )