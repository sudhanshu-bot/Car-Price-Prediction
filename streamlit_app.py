import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas.api.types import is_numeric_dtype

from model_pipeline import FEATURES, MODEL_PATH, TARGET, load_clean_data, predict_one, train_and_save


st.set_page_config(page_title="Indian Cars Intelligence", page_icon="C", layout="wide", initial_sidebar_state="collapsed")

NOTEBOOK_METRICS = {
    "Random Forest": {"MAE": 149663.323727, "RMSE": 320467.424072, "MAPE": 8.486981, "R2": 0.978769},
    "Linear Regression": {"MAE": 238985.060384, "RMSE": 388509.884028, "MAPE": 12.390477, "R2": 0.968796},
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    :root { --ink: #17262b; --muted: #6c7b80; --accent: #147d82; --warm: #e46b3f; --paper: #f5f7f3; }
    .stApp { background: var(--paper); color: var(--ink); }
    .block-container { padding-top: 2.2rem; padding-bottom: 3.5rem; max-width: 1380px; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Manrope', sans-serif; letter-spacing: 0; color: var(--ink); }
    h2 { margin-top: 1.75rem; }
    .hero { position: relative; overflow: hidden; background: linear-gradient(115deg, #123f46 0%, #147d82 62%, #e46b3f 160%); border-radius: 8px; padding: 2.25rem 2.5rem; margin-bottom: 1.5rem; box-shadow: 0 12px 28px rgba(20, 77, 82, .16); }
    .hero:after { content: ''; position: absolute; right: -70px; top: -100px; width: 280px; height: 280px; border: 1px solid rgba(255,255,255,.22); border-radius: 50%; }
    .hero .eyebrow { color: #ffd9c9; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero h1 { position: relative; z-index: 1; color: #ffffff; margin: .3rem 0 0; font-size: clamp(1.8rem, 4vw, 3rem); }
    .hero p { position: relative; z-index: 1; color: #dcebed; margin: .55rem 0 0; max-width: 680px; font-size: 1.02rem; }
    div[data-testid="stSidebar"] { background: #edf2f0; border-right: 1px solid #d6e1de; }
    div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 { color: #123f46; }
    div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #dbe5e2; border-radius: 8px; padding: 1.15rem 1.2rem; box-shadow: 0 5px 15px rgba(22, 47, 49, .05); }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p { color: #5f7075 !important; font-weight: 600 !important; opacity: 1 !important; }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] div { color: var(--ink) !important; font-family: 'Manrope', sans-serif; }
    div[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] label, label p { color: #5f7075 !important; font-weight: 600 !important; opacity: 1 !important; }
    .stCaption, [data-testid="stCaptionContainer"] p { color: #6c7b80 !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-color: #dbe5e2; }
    .section-note { color: var(--muted); font-size: .92rem; margin-top: -.6rem; margin-bottom: 1rem; }
    @media (max-width: 700px) { .block-container { padding: 1rem .8rem 2.5rem; } .hero { padding: 1.5rem; } .hero h1 { font-size: 1.8rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#ffffff", "figure.facecolor": "#ffffff", "grid.color": "#e7eeeb", "axes.edgecolor": "#dbe5e2", "font.family": "sans-serif"})


@st.cache_data(show_spinner=False)
def get_data():
    data = load_clean_data().copy()
    categorical_columns = data.select_dtypes(include=["object", "string"]).columns
    for column in categorical_columns:
        data[column] = data[column].astype(object).where(data[column].notna(), "Not specified")
    return data


@st.cache_resource(show_spinner=False)
def get_artifact():
    if not MODEL_PATH.exists():
        train_and_save()
    import joblib
    return joblib.load(MODEL_PATH)


def pretty_name(value):
    return value.replace("_", " ").replace("/", " / ")


data = get_data()
artifact = get_artifact()
metrics = artifact.get("metrics", {})

st.markdown(
    '<div class="hero"><div class="eyebrow">2021 market intelligence</div><h1>Indian Cars Intelligence</h1><p>Explore the Indian cars market, understand pricing patterns, and estimate ex-showroom price from vehicle specifications.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Dashboard")
    view = st.radio("View", ["Overview", "Market Explorer", "Predictor"], label_visibility="collapsed")
    st.divider()
    st.caption("Source: Indian Cars Dataset, Kaggle")
    st.caption("Model: Random Forest Regressor")


if view == "Overview":
    st.subheader("Market snapshot")
    st.markdown('<div class="section-note">A concise view of the dataset, pricing landscape, and validated notebook benchmark.</div>', unsafe_allow_html=True)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Cars analyzed", f"{len(data):,}")
    metric_columns[1].metric("Brands", f"{data['Make'].nunique():,}")
    metric_columns[2].metric("Median price", f"Rs. {data[TARGET].median():,.0f}")
    metric_columns[3].metric("Notebook R2", f"{NOTEBOOK_METRICS['Random Forest']['R2']:.3f}")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Price distribution")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(data[TARGET], bins=30, color="#176b87", ax=ax)
        ax.set_xlabel("Ex-showroom price")
        ax.set_ylabel("Cars")
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with chart_right:
        st.markdown("#### Brands with the most listings")
        brand_counts = data["Make"].value_counts().head(10).sort_values()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        brand_counts.plot.barh(color="#d95f02", ax=ax)
        ax.set_xlabel("Listings")
        ax.set_ylabel("")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.markdown("#### Model quality")
    result_columns = st.columns(3)
    result_columns[0].metric("Random Forest MAE", f"Rs. {NOTEBOOK_METRICS['Random Forest']['MAE']:,.0f}")
    result_columns[1].metric("Random Forest RMSE", f"Rs. {NOTEBOOK_METRICS['Random Forest']['RMSE']:,.0f}")
    result_columns[2].metric("Linear Regression R2", f"{NOTEBOOK_METRICS['Linear Regression']['R2']:.4f}")
    st.caption("Official notebook benchmark: Random Forest R2 = 0.9788; Linear Regression R2 = 0.9688. The API artifact is a separate deployment pipeline.")


elif view == "Market Explorer":
    st.subheader("Market explorer")
    st.markdown('<div class="section-note">Filter the market to compare segments, prices, and the highest-priced listings.</div>', unsafe_allow_html=True)
    filters = st.columns(3)
    make_options = ["All brands"] + sorted(data["Make"].dropna().astype(str).unique().tolist())
    selected_make = filters[0].selectbox("Brand", make_options)
    body_options = ["All body types"] + sorted(data["Body_Type"].dropna().astype(str).unique().tolist())
    selected_body = filters[1].selectbox("Body type", body_options)
    fuel_options = ["All fuel types"] + sorted(data["Fuel_Type"].dropna().astype(str).unique().tolist())
    selected_fuel = filters[2].selectbox("Fuel type", fuel_options)

    filtered = data.copy()
    if selected_make != "All brands":
        filtered = filtered[filtered["Make"].astype(str) == selected_make]
    if selected_body != "All body types":
        filtered = filtered[filtered["Body_Type"].astype(str) == selected_body]
    if selected_fuel != "All fuel types":
        filtered = filtered[filtered["Fuel_Type"].astype(str) == selected_fuel]

    st.caption(f"Showing {len(filtered):,} of {len(data):,} cars")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Price by body type")
        if len(filtered):
            grouped = filtered.groupby("Body_Type")[TARGET].median().sort_values().tail(12)
            fig, ax = plt.subplots(figsize=(8, 5))
            grouped.plot.barh(color="#176b87", ax=ax)
            ax.set_xlabel("Median ex-showroom price")
            ax.set_ylabel("")
            st.pyplot(fig, width="stretch")
            plt.close(fig)
    with chart_right:
        st.markdown("#### Price versus engine displacement")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=filtered, x="Displacement", y=TARGET, hue="Fuel_Type", alpha=.65, ax=ax, legend=False)
        ax.set_xlabel("Displacement (cc)")
        ax.set_ylabel("Ex-showroom price")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.markdown("#### Highest-priced listings in the current selection")
    display_columns = [column for column in ["Make", "Model", "Variant", "Body_Type", "Fuel_Type", TARGET] if column in filtered.columns]
    st.dataframe(filtered[display_columns].sort_values(TARGET, ascending=False).head(15), width="stretch", hide_index=True)


else:
    st.subheader("Price predictor")
    st.markdown('<div class="section-note">Shape a vehicle profile and request an estimated ex-showroom price.</div>', unsafe_allow_html=True)
    defaults = data.iloc[0].to_dict()
    values = {}
    numeric_features = [feature for feature in FEATURES if feature in data.columns and is_numeric_dtype(data[feature])]
    categorical_features = [feature for feature in FEATURES if feature in data.columns and feature not in numeric_features]

    with st.form("prediction_form"):
        st.markdown("#### Vehicle identity")
        identity = st.columns(3)
        for index, feature in enumerate(["Make", "Model", "Variant"]):
            if feature in categorical_features:
                options = sorted(data[feature].dropna().astype(str).unique().tolist())
                values[feature] = identity[index].selectbox(pretty_name(feature), options, index=options.index(str(defaults[feature])) if str(defaults[feature]) in options else 0)

        st.markdown("#### Core specifications")
        core = st.columns(3)
        for index, feature in enumerate(["Displacement", "Power", "Torque", "Fuel_Tank_Capacity", "Seating_Capacity", "Kerb_Weight"]):
            if feature in numeric_features:
                default = defaults[feature]
                values[feature] = core[index % 3].number_input(pretty_name(feature), value=float(default) if pd.notna(default) else 0.0, key=f"input_{feature}")

        with st.expander("More specifications"):
            extra_columns = st.columns(3)
            for index, feature in enumerate([item for item in FEATURES if item not in values and item in data.columns]):
                target_column = extra_columns[index % 3]
                if feature in numeric_features:
                    default = defaults[feature]
                    values[feature] = target_column.number_input(pretty_name(feature), value=float(default) if pd.notna(default) else 0.0, key=f"extra_{feature}")
                else:
                    options = sorted(data[feature].dropna().astype(str).unique().tolist())
                    if options:
                        values[feature] = target_column.selectbox(pretty_name(feature), options, index=options.index(str(defaults[feature])) if str(defaults[feature]) in options else 0, key=f"extra_{feature}")

        submitted = st.form_submit_button("Predict price", type="primary", width="stretch")

    if submitted:
        with st.spinner("Calculating estimate..."):
            estimate = predict_one(values)
        st.success(f"Estimated ex-showroom price: Rs. {estimate:,.0f}")