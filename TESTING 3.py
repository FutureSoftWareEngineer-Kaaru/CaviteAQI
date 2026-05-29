
# =========================================================
# IMPORT LIBRARIES
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    f1_score
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AQI Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS DESIGN
# =========================================================
st.markdown("""
<style>

/* =====================================================
SIDEBAR BUTTONS
===================================================== */

div.stButton > button {
    font-family: 'Poppins', sans-serif;
    background-color: #0f2747;
    color: white;
    border-radius: 12px;
    height: 50px;
    border: 1px solid #1e3a5f;
    font-weight: bold;
    margin-bottom: 8px;
    transition: 0.3s;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    border: 1px solid #60a5fa;
    transform: scale(1.02);
}

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #03142c 0%, #020d1d 100%);
    color: white;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #04162f 0%, #071d3b 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.sidebar-title {
    font-size: 30px;
    font-weight: 700;
    color: white;
}

section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

.sidebar-sub {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 30px;
}

.main-title {
    font-size: 42px;
    font-weight: 700;
    color: white;
    margin-bottom: 0px;
}

.sub-title {
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 20px;
}

.metric-card {
    background: linear-gradient(145deg, #0b1e3d, #07152b);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 25px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    margin-bottom: 15px;
}

.metric-title {
    color: #cbd5e1;
    font-size: 15px;
}

.metric-value {
    color: white;
    font-size: 32px;
    font-weight: 700;
}

.metric-sub {
    color: #94a3b8;
    font-size: 13px;
}

.chart-card {
    background: linear-gradient(145deg, #081a35, #061427);
    border-radius: 24px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-top: 10px;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border-radius: 12px;
    border: none;
    height: 50px;
    font-size: 16px;
    font-weight: 600;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: #08152b;
    border-radius: 12px;
}

.stNumberInput input {
    background-color: #08152b;
    color: white;
}

[data-testid="stDataFrame"] {
    background-color: #08152b;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATASET
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        "filtered_cavite_cities_combined.csv",
        encoding="latin1"
    )

    return df

# =========================================================
# LOAD DATA
# =========================================================
df = load_data()

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================
df.columns = df.columns.str.strip()

# =========================================================
# CHECK IF DATETIME COLUMN EXISTS
# =========================================================
if 'datetime' not in df.columns:

    st.error("ERROR: 'datetime' column not found in dataset")

    st.write(df.columns)

    st.stop()

# =========================================================
# CONVERT DATETIME
# =========================================================
df['datetime'] = pd.to_datetime(
    df['datetime'],
    errors='coerce',
)

# =========================================================
# REMOVE INVALID DATETIME ROWS
# =========================================================
df = df.dropna(subset=['datetime'])

# =========================================================
# REMOVE TIMEZONE SAFELY
# =========================================================
try:
    df['datetime'] = df['datetime'].dt.tz_localize(None)
except:
    pass

# =========================================================
# REMOVE DUPLICATES
# =========================================================
df = df.drop_duplicates()

# =========================================================
# NUMERIC COLUMNS
# =========================================================
numeric_cols = [
    "components.co",
    "components.no",
    "components.no2",
    "components.o3",
    "components.so2",
    "components.pm2_5",
    "components.pm10",
    "components.nh3"
]

# =========================================================
# CHECK MISSING COLUMNS
# =========================================================
missing_cols = []

for col in numeric_cols:

    if col not in df.columns:
        missing_cols.append(col)

if len(missing_cols) > 0:

    st.error(f"Missing Columns: {missing_cols}")

    st.stop()

# =========================================================
# CONVERT NUMERIC VALUES
# =========================================================
for col in numeric_cols:

    df[col] = pd.to_numeric(
        df[col],
        errors='coerce'
    )

# =========================================================
# FILL MISSING VALUES
# =========================================================
df[numeric_cols] = df[numeric_cols].fillna(
    df[numeric_cols].mean()
)

# =========================================================
# EXTRACT DATE FEATURES
# =========================================================
df['Year'] = df['datetime'].dt.year
df['Month'] = df['datetime'].dt.month
df['Day'] = df['datetime'].dt.day
df['Hour'] = df['datetime'].dt.hour

# =========================================================
# FILTER CITIES
# =========================================================
cities = [
    "Bacoor",
    "Cavite City",
    "Dasmariñas",
    "General Trias",
    "Imus"
]


df = df[df["city_name"].isin(cities)]

# =========================================================
# CLEAN POLLUTANT VALUES
# =========================================================

pollutant_columns = [
    "components.co",
    "components.no",
    "components.no2",
    "components.o3",
    "components.so2",
    "components.pm2_5",
    "components.pm10",
    "components.nh3"
]

for col in pollutant_columns:

    # Convert to numeric
    df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove negative values
    df[col] = df[col].clip(lower=0)

    # Fill missing values
    df[col] = df[col].fillna(df[col].median())

# =========================================================
# AQI COMPUTATION FORMULA
# =========================================================
def calculate_aqi(cp, bp_lo, bp_hi, i_lo, i_hi):

    aqi = ((i_hi - i_lo) / (bp_hi - bp_lo)) * \
          (cp - bp_lo) + i_lo

    return round(aqi)

# =========================================================
# PM2.5 AQI
# =========================================================
def get_pm25_aqi(cp):

    cp = max(cp, 0)

    if cp <= 12.0:
        return calculate_aqi(cp, 0.0, 12.0, 0, 50)

    elif cp <= 35.4:
        return calculate_aqi(cp, 12.1, 35.4, 51, 100)

    elif cp <= 55.4:
        return calculate_aqi(cp, 35.5, 55.4, 101, 150)

    elif cp <= 150.4:
        return calculate_aqi(cp, 55.5, 150.4, 151, 200)

    elif cp <= 250.4:
        return calculate_aqi(cp, 150.5, 250.4, 201, 300)

    elif cp <= 350.4:
        return calculate_aqi(cp, 250.5, 350.4, 301, 400)

    else:
        return calculate_aqi(cp, 350.5, 500.4, 401, 500)

# =========================================================
# PM10 AQI
# =========================================================
def get_pm10_aqi(cp):

    cp = max(cp, 0)

    if cp <= 54:
        return calculate_aqi(cp, 0, 54, 0, 50)

    elif cp <= 154:
        return calculate_aqi(cp, 55, 154, 51, 100)

    elif cp <= 254:
        return calculate_aqi(cp, 155, 254, 101, 150)

    elif cp <= 354:
        return calculate_aqi(cp, 255, 354, 151, 200)

    elif cp <= 424:
        return calculate_aqi(cp, 355, 424, 201, 300)

    elif cp <= 504:
        return calculate_aqi(cp, 425, 504, 301, 400)

    else:
        return calculate_aqi(cp, 505, 604, 401, 500)

# =========================================================
# CO AQI
# Dataset CO is in μg/m³
# Convert to ppm first
# =========================================================
def get_co_aqi(cp):

    cp = max(cp, 0)

    cp = cp / 1145

    if cp <= 4.4:
        return calculate_aqi(cp, 0.0, 4.4, 0, 50)

    elif cp <= 9.4:
        return calculate_aqi(cp, 4.5, 9.4, 51, 100)

    elif cp <= 12.4:
        return calculate_aqi(cp, 9.5, 12.4, 101, 150)

    elif cp <= 15.4:
        return calculate_aqi(cp, 12.5, 15.4, 151, 200)

    elif cp <= 30.4:
        return calculate_aqi(cp, 15.5, 30.4, 201, 300)

    else:
        return calculate_aqi(cp, 30.5, 50.4, 301, 500)

# =========================================================
# SO2 AQI
# =========================================================
def get_so2_aqi(cp):

    cp = max(cp, 0)

    if cp <= 35:
        return calculate_aqi(cp, 0, 35, 0, 50)

    elif cp <= 75:
        return calculate_aqi(cp, 36, 75, 51, 100)

    elif cp <= 185:
        return calculate_aqi(cp, 76, 185, 101, 150)

    elif cp <= 304:
        return calculate_aqi(cp, 186, 304, 151, 200)

    elif cp <= 604:
        return calculate_aqi(cp, 305, 604, 201, 300)

    else:
        return calculate_aqi(cp, 605, 1004, 301, 500)

# =========================================================
# NO2 AQI
# =========================================================
def get_no2_aqi(cp):

    cp = max(cp, 0)

    if cp <= 53:
        return calculate_aqi(cp, 0, 53, 0, 50)

    elif cp <= 100:
        return calculate_aqi(cp, 54, 100, 51, 100)

    elif cp <= 360:
        return calculate_aqi(cp, 101, 360, 101, 150)

    elif cp <= 649:
        return calculate_aqi(cp, 361, 649, 151, 200)

    elif cp <= 1249:
        return calculate_aqi(cp, 650, 1249, 201, 300)

    else:
        return calculate_aqi(cp, 1250, 2049, 301, 500)

# =========================================================
# O3 AQI
# Dataset O3 is μg/m³
# Convert to ppm first
# =========================================================
def get_o3_aqi(cp):

    cp = max(cp, 0)

    cp = cp / 1000

    if cp <= 0.054:
        return calculate_aqi(cp, 0.000, 0.054, 0, 50)

    elif cp <= 0.070:
        return calculate_aqi(cp, 0.055, 0.070, 51, 100)

    elif cp <= 0.085:
        return calculate_aqi(cp, 0.071, 0.085, 101, 150)

    elif cp <= 0.105:
        return calculate_aqi(cp, 0.086, 0.105, 151, 200)

    elif cp <= 0.200:
        return calculate_aqi(cp, 0.106, 0.200, 201, 300)

    else:
        return calculate_aqi(cp, 0.201, 0.604, 301, 500)

# =========================================================
# COMPUTE AQI PER POLLUTANT
# =========================================================
df["AQI_PM25"] = df["components.pm2_5"].apply(get_pm25_aqi)

df["AQI_PM10"] = df["components.pm10"].apply(get_pm10_aqi)

df["AQI_CO"] = df["components.co"].apply(get_co_aqi)

df["AQI_SO2"] = df["components.so2"].apply(get_so2_aqi)

df["AQI_NO2"] = df["components.no2"].apply(get_no2_aqi)

df["AQI_O3"] = df["components.o3"].apply(get_o3_aqi)

# =========================================================
# FINAL AQI
# =========================================================
pollutant_cols = [
    "AQI_PM25",
    "AQI_PM10",
    "AQI_CO",
    "AQI_SO2",
    "AQI_NO2",
    "AQI_O3"
]

df["Final_AQI"] = df[pollutant_cols].max(axis=1)

df["Final_AQI"] = df["Final_AQI"].clip(
    lower=0,
    upper=500
)

# =========================================================
# MAIN POLLUTANT
# =========================================================
df["Main_Pollutant"] = (
    df[pollutant_cols]
    .idxmax(axis=1)
    .str.replace("AQI_", "")
)

# =========================================================
# AQI CATEGORY
# =========================================================
def classify_aqi(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Fair"

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"

df["AQI_Category"] = df["Final_AQI"].apply(classify_aqi)

# =========================================================
# MACHINE LEARNING FEATURES
# =========================================================
features = [
    "components.co",
    "components.no",
    "components.no2",
    "components.o3",
    "components.so2",
    "components.pm2_5",
    "components.pm10",
    "components.nh3"
]

X = df[features]

y_reg = df["Final_AQI"]

y_class = df["AQI_Category"]

# =========================================================
# NORMALIZATION
# =========================================================
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_scaled,
    y_reg,
    test_size=0.2,
    random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_scaled,
    y_class,
    test_size=0.2,
    random_state=42
)

# =========================================================
# LINEAR REGRESSION
# =========================================================
lr_model = LinearRegression()

lr_model.fit(X_train_reg, y_train_reg)


y_pred_reg = lr_model.predict(X_test_reg)

# =========================================================
# DECISION TREE
# =========================================================
dt_model = DecisionTreeClassifier(random_state=42)


dt_model.fit(X_train_clf, y_train_clf)

# =========================================================
# RANDOM FOREST
# =========================================================
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train_clf, y_train_clf)

rf_pred = rf_model.predict(X_test_clf)

# =========================================================
# DECISION TREE PREDICTION
# =========================================================
dt_pred = dt_model.predict(X_test_clf)

# =========================================================
# RANDOM FOREST METRICS
# =========================================================
rf_accuracy = accuracy_score(y_test_clf, rf_pred)

rf_precision = precision_score(
    y_test_clf,
    rf_pred,
    average='weighted',
    zero_division=0
)

rf_recall = recall_score(
    y_test_clf,
    rf_pred,
    average='weighted',
    zero_division=0
)

rf_f1 = f1_score(
    y_test_clf,
    rf_pred,
    average='weighted',
    zero_division=0
)

rf_cm = confusion_matrix(
    y_test_clf,
    rf_pred
)

# =========================================================
# DECISION TREE METRICS
# =========================================================
dt_accuracy = accuracy_score(y_test_clf, dt_pred)

dt_precision = precision_score(
    y_test_clf,
    dt_pred,
    average='weighted',
    zero_division=0
)

dt_recall = recall_score(
    y_test_clf,
    dt_pred,
    average='weighted',
    zero_division=0
)

dt_f1 = f1_score(
    y_test_clf,
    dt_pred,
    average='weighted',
    zero_division=0
)

dt_cm = confusion_matrix(
    y_test_clf,
    dt_pred
)

# =========================================================
# EVALUATION METRICS
# =========================================================
mae = mean_absolute_error(y_test_reg, y_pred_reg)

rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

r2 = r2_score(y_test_reg, y_pred_reg)

accuracy = accuracy_score(y_test_clf, rf_pred)

precision = precision_score(
    y_test_clf,
    rf_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test_clf,
    rf_pred,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(y_test_clf, rf_pred)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("""
    <div class='sidebar-title'>☁️ AQI ANALYSIS</div>
    <div class='sidebar-sub'>CAVITE AIR QUALITY DASHBOARD</div>
    """, unsafe_allow_html=True)

    # =====================================================
    # SIDEBAR NAVIGATION BUTTONS
    # =====================================================
    st.sidebar.title("Navigation")

    # initialize session state
    if "menu" not in st.session_state:
        st.session_state.menu = "Overview"

    # =====================================================
    # BUTTONS
    # =====================================================
    if st.sidebar.button("Overview", use_container_width=True):
        st.session_state.menu = "Overview"

    if st.sidebar.button("City Dashboard", use_container_width=True):
        st.session_state.menu = "City Dashboard"

    if st.sidebar.button("Prediction", use_container_width=True):
        st.session_state.menu = "Prediction"

    if st.sidebar.button("Model Performance", use_container_width=True):
        st.session_state.menu = "Model Performance"

    if st.sidebar.button("About", use_container_width=True):
        st.session_state.menu = "About"

    # =====================================================
    # ACTIVE MENU
    # =====================================================
    menu = st.session_state.menu

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class='main-title'>Air Quality Index (AQI) Dashboard</div>
<div class='sub-title'>Real-time Analysis, Prediction & Visualization</div>
""", unsafe_allow_html=True)

# =========================================================
# OVERVIEW TAB
# =========================================================
if menu == "Overview":

    # =====================================================
    # TOP METRICS
    # =====================================================
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Average AQI", f"{df['Final_AQI'].mean():.2f}")
    c2.metric("Maximum AQI", f"{df['Final_AQI'].max():.2f}")
    c3.metric("Minimum AQI", f"{df['Final_AQI'].min():.2f}")
    c4.metric("Total Records", f"{len(df):,}")
    c5.metric("Main Pollutant", "PM2.5")

    # =====================================================
    # AQI TREND ALL CITIES
    # =====================================================
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)

    st.subheader("AQI Trend Over Time (All Cities)")

    fig, ax = plt.subplots(figsize=(14,5))

    colors = ['cyan', 'orange', 'lime', 'yellow', 'magenta']

    for city, color in zip(cities, colors):

        city_data = (
            df[df['city_name'] == city]
            .groupby(df['datetime'].dt.date)['Final_AQI']
            .mean()
        )

        ax.plot(
            city_data.index,
            city_data.values,
            marker='o',
            linewidth=3,
            label=city,
            color=color
        )

    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # BAR GRAPH + CORRELATION MATRIX
    # =====================================================
    left, right = st.columns(2)

    with left:

        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)

        st.subheader("Average AQI by City")

        city_avg = (
            df.groupby('city_name')['Final_AQI']
            .mean()
        )

        fig, ax = plt.subplots(figsize=(7,4))

        bars = ax.bar(
            city_avg.index,
            city_avg.values
        )

        for bar in bars:

            y = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width()/2,
                y + 1,
                f'{y:.1f}',
                ha='center'
            )

        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:

        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)

        st.subheader("Pollutant Correlation Matrix")

        corr = df[features].corr()

        fig, ax = plt.subplots(figsize=(8,6))

        im = ax.imshow(corr, cmap='coolwarm')

        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))

        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)

        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(
                    j,
                    i,
                    f'{corr.iloc[i, j]:.2f}',
                    ha='center',
                    va='center',
                    color='white'
                )

        plt.colorbar(im)

        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # MODEL PERFORMANCE SUMMARY
    # =====================================================
    st.subheader("Model Performance Summary")

    p1, p2, p3, p4, p5, p6 = st.columns(6)

    p1.metric("MAE", f"{mae:.2f}")
    p2.metric("RMSE", f"{rmse:.2f}")
    p3.metric("R²", f"{r2:.3f}")
    p4.metric("Accuracy", f"{accuracy:.3f}")
    p5.metric("Precision", f"{precision:.3f}")
    p6.metric("Recall", f"{recall:.3f}")

   # =====================================================
   #  ACTUAL VS PREDICTED GRAPH
   # =====================================================
    st.subheader("Actual vs Predicted AQI")

    fig, ax = plt.subplots(figsize=(7,5))

    ax.scatter(
        y_test_reg,
        y_pred_reg,
        alpha=0.7
    )

    ax.set_title(
        "Actual vs Predicted AQI",
        fontsize=12
    )

    ax.set_xlabel(
        "Actual AQI",
        fontsize=10
    )

    ax.set_ylabel(
        "Predicted AQI",
        fontsize=10
    )

    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
# =========================================================
# CITY DASHBOARD
# =========================================================
elif menu == "City Dashboard":

    st.header("City AQI Dashboard")

    selected_city = st.selectbox(
        "Select City",
        cities
    )

    city_df = df[df['city_name'] == selected_city]

    avg_aqi = city_df['Final_AQI'].mean()

    classification = classify_aqi(avg_aqi)

    c1, c2 = st.columns(2)

    c1.metric(
        "Average AQI",
        f"{avg_aqi:.2f}"
    )

    c2.metric(
        "AQI Classification",
        classification
    )

    st.subheader("Monthly AQI Trend")

    monthly = (
    city_df
    .set_index('datetime')
    .sort_index()
    .resample('ME')['Final_AQI']
    .mean()
        
    )

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        monthly.index,
        monthly.values,
        marker='o',
        linewidth=3
    )

    ax.grid(alpha=0.2)

    st.pyplot(fig)

    st.subheader("Pollutant AQI Breakdown")

    pollutant_avg = {
        "PM2.5": city_df['AQI_PM25'].mean(),
        "PM10": city_df['AQI_PM10'].mean(),
        "CO": city_df['AQI_CO'].mean(),
        "SO2": city_df['AQI_SO2'].mean(),
        "NO2": city_df['AQI_NO2'].mean(),
        "O3": city_df['AQI_O3'].mean()
    }

    pollutant_df = pd.DataFrame({
        "Pollutant": pollutant_avg.keys(),
        "AQI": pollutant_avg.values()
    })

    st.dataframe(pollutant_df)

    # =====================================================
    # AQI CLASSIFICATION GUIDE
    # =====================================================
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='chart-card'>

    <h3 style='color:white;'>AQI Classification Guide</h3>

    <table style='width:100%; border-collapse:collapse; text-align:center;'>

    <tr style='background:#0f2747; color:white;'>
        <th style='padding:12px;'>AQI Range</th>
        <th style='padding:12px;'>Classification</th>
        <th style='padding:12px;'>Air Quality Status</th>
    </tr>

    <tr style='background:#064e3b; color:white;'>
        <td style='padding:12px;'>0 - 50</td>
        <td>Good</td>
        <td>Air quality is satisfactory</td>
    </tr>

    <tr style='background:#365314; color:white;'>
        <td style='padding:12px;'>51 - 100</td>
        <td>Fair</td>
        <td>Acceptable air quality</td>
    </tr>

    <tr style='background:#854d0e; color:white;'>
        <td style='padding:12px;'>101 - 150</td>
        <td>Unhealthy for Sensitive Groups</td>
        <td>Sensitive individuals may experience effects</td>
    </tr>

    <tr style='background:#7f1d1d; color:white;'>
        <td style='padding:12px;'>151 - 200</td>
        <td>Unhealthy</td>
        <td>Health effects may occur</td>
    </tr>

    <tr style='background:#581c87; color:white;'>
        <td style='padding:12px;'>201 - 300</td>
        <td>Very Unhealthy</td>
        <td>Health alert condition</td>
    </tr>

    <tr style='background:#450a0a; color:white;'>
        <td style='padding:12px;'>301 - 500</td>
        <td>Hazardous</td>
        <td>Serious health effects</td>
    </tr>

    </table>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PREDICTION TAB
# =========================================================
elif menu == "Prediction":

    st.header("AQI Prediction")

    st.subheader("How AQI is Computed")
    
    st.latex(r'''
    I_p = \frac{I_{Hi}-I_{Lo}}{BP_{Hi}-BP_{Lo}}(C_p-BP_{Lo})+I_{Lo}
    ''')

    st.markdown("""
    The system computes AQI per pollutant.

    The pollutant with the highest AQI determines the Final AQI.

    Machine learning is then used to predict AQI levels.
    """)

    col1, col2 = st.columns(2)

    with col1:
        co = st.number_input("CO", value=180.0)
        no = st.number_input("NO", value=1.0)
        no2 = st.number_input("NO2", value=5.0)
        o3 = st.number_input("O3", value=70.0)

    with col2:
        so2 = st.number_input("SO2", value=2.0)
        pm25 = st.number_input("PM2.5", value=10.0)
        pm10 = st.number_input("PM10", value=12.0)
        nh3 = st.number_input("NH3", value=1.0)

    if st.button("Predict AQI"):

        input_data = pd.DataFrame({
            "components.co": [co],
            "components.no": [no],
            "components.no2": [no2],
            "components.o3": [o3],
            "components.so2": [so2],
            "components.pm2_5": [pm25],
            "components.pm10": [pm10],
            "components.nh3": [nh3]
        })

        input_scaled = scaler.transform(input_data)

        predicted_aqi = lr_model.predict(input_scaled)[0]

        predicted_category = classify_aqi(predicted_aqi)

        st.success(f"Predicted AQI: {predicted_aqi:.2f}")

        st.info(f"AQI Classification: {predicted_category}")
    # =====================================================
    # AQI CLASSIFICATION GUIDE
    # =====================================================
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='chart-card'>

    <h3 style='color:white;'>AQI Classification Guide</h3>

    <table style='width:100%; border-collapse:collapse; text-align:center;'>

    <tr style='background:#0f2747; color:white;'>
        <th style='padding:12px;'>AQI Range</th>
        <th style='padding:12px;'>Classification</th>
        <th style='padding:12px;'>Air Quality Status</th>
    </tr>

    <tr style='background:#064e3b; color:white;'>
        <td style='padding:12px;'>0 - 50</td>
        <td>Good</td>
        <td>Air quality is satisfactory</td>
    </tr>

    <tr style='background:#365314; color:white;'>
        <td style='padding:12px;'>51 - 100</td>
        <td>Fair</td>
        <td>Acceptable air quality</td>
    </tr>

    <tr style='background:#854d0e; color:white;'>
        <td style='padding:12px;'>101 - 150</td>
        <td>Unhealthy for Sensitive Groups</td>
        <td>Sensitive individuals may experience effects</td>
    </tr>

    <tr style='background:#7f1d1d; color:white;'>
        <td style='padding:12px;'>151 - 200</td>
        <td>Unhealthy</td>
        <td>Health effects may occur</td>
    </tr>

    <tr style='background:#581c87; color:white;'>
        <td style='padding:12px;'>201 - 300</td>
        <td>Very Unhealthy</td>
        <td>Health alert condition</td>
    </tr>

    <tr style='background:#450a0a; color:white;'>
        <td style='padding:12px;'>301 - 500</td>
        <td>Hazardous</td>
        <td>Serious health effects</td>
    </tr>

    </table>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# MODEL PERFORMANCE TAB
# =========================================================
elif menu == "Model Performance":

    st.header("Model Performance")

    # =====================================================
    # LINEAR REGRESSION
    # =====================================================
    st.subheader("Linear Regression")

    lr1, lr2, lr3 = st.columns(3)

    with lr1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>MAE</div>
            <div class='metric-value'>{mae:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with lr2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>RMSE</div>
            <div class='metric-value'>{rmse:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with lr3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>R² Score</div>
            <div class='metric-value'>{r2:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    st.subheader("Random Forest")

    # =====================================================
    # METRIC CARDS
    # =====================================================
    rf1, rf2, rf3, rf4 = st.columns(4)

    with rf1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Accuracy</div>
            <div class='metric-value'>{rf_accuracy:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with rf2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Precision</div>
            <div class='metric-value'>{rf_precision:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with rf3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Recall</div>
            <div class='metric-value'>{rf_recall:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with rf4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>F1-Score</div>
            <div class='metric-value'>{rf_f1:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================
    st.markdown("""
    <div class='chart-card'>
    <h4 style='color:white;'>Random Forest Confusion Matrix</h4>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(3.8,3.2))

    im = ax.imshow(rf_cm, cmap='Blues')

    labels = [
        "Good",
        "Fair",
        "Sensitive",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous"
    ]

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=15, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)

    for i in range(len(rf_cm)):
        for j in range(len(rf_cm)):
            ax.text(
                j,
                i,
                rf_cm[i, j],
                ha='center',
                va='center',
                color='white',
                fontsize=8
            )

    st.pyplot(fig)
    st.markdown("---")

    # =====================================================
    # DECISION TREE
    # =====================================================
    st.subheader("Decision Tree")

    # =====================================================
    # METRIC CARDS
    # =====================================================
    dt1, dt2, dt3, dt4 = st.columns(4)

    with dt1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Accuracy</div>
            <div class='metric-value'>{dt_accuracy:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with dt2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Precision</div>
            <div class='metric-value'>{dt_precision:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with dt3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Recall</div>
            <div class='metric-value'>{dt_recall:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with dt4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>F1-Score</div>
            <div class='metric-value'>{dt_f1:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================
    st.markdown("""
    <div class='chart-card'>
    <h4 style='color:white;'>Decision Tree Confusion Matrix</h4>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(3.8,3.2))

    im = ax.imshow(dt_cm, cmap='Purples')

    labels = [
        "Good",
        "Fair",
        "Sensitive",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous"
    ]

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=15, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)

    for i in range(len(dt_cm)):
        for j in range(len(dt_cm)):
            ax.text(
                j,
                i,
                dt_cm[i, j],
                ha='center',
                va='center',
                color='white',
                fontsize=8
            )

    st.pyplot(fig)

    st.markdown("---")

# =====================================================
# ABOUT PAGE
# =====================================================
elif menu == "About":

    st.header("About the System")

    st.markdown("""
    <div class='chart-card'>

    <h3 style='color:white;'>
    Air Quality Index (AQI) Analysis and Prediction System
    </h3>

    <p style='font-size:17px; line-height:1.8; color:#d1d5db;'>

    This system was developed to analyze, visualize, classify,
    and predict Air Quality Index (AQI) levels in selected
    cities of Cavite using Machine Learning algorithms.

    The dashboard provides real-time visualization,
    AQI computation, pollutant monitoring,
    and predictive analytics to help users better understand
    environmental air quality conditions.

    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # INFORMATION CARDS
    # =====================================================
    col1, col2 = st.columns(2)

    # =====================================================
    # LEFT CARD
    # =====================================================
    with col1:

        st.markdown("""
        <div class='metric-card'>

        <h3 style='color:white;'>
        Developers / Researchers
        </h3>

        <ul style='line-height:2; font-size:16px;'>
            <li>Carl Vincent Alera</li>
            <li>Lawrence Delos Santos</li>
            <li>Kim Cyrus Hainto</li>
            <li>Sheryn Jamero</li>
            <li>Alexis Labayen</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class='metric-card'>

        <h3 style='color:white;'>
        Program
        </h3>

        <p style='font-size:16px; line-height:1.8;'>
        Bachelor of Science in Computer Engineering
        </p>

        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # RIGHT CARD
    # =====================================================
    with col2:

        st.markdown("""
        <div class='metric-card'>

        <h3 style='color:white;'>
        System Features
        </h3>

        <ul style='line-height:2; font-size:16px;'>
            <li>AQI Analysis</li>
            <li>AQI Prediction</li>
            <li>Machine Learning Classification</li>
            <li>Data Visualization Dashboard</li>
            <li>Pollutant Monitoring</li>
            <li>Interactive Charts & Graphs</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class='metric-card'>

        <h3 style='color:white;'>
        Machine Learning Models
        </h3>

        <ul style='line-height:2; font-size:16px;'>
            <li>Linear Regression</li>
            <li>Random Forest Classifier</li>
            <li>Decision Tree Classifier</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # FOOTER
    # =====================================================
    st.markdown("""
    <div style='
        text-align:center;
        padding:20px;
        color:#9ca3af;
        font-size:15px;
    '>

    AQI Analysis and Prediction System © 2026

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown(
    "Developed using Streamlit • Scikit-learn • Pandas • Matplotlib"
)
