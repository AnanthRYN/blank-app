import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Calculated vs Predicted WQI",
    layout="centered"
)

st.title("💧 Calculated vs Predicted Water Quality Index")
st.caption("Physics-based WQI vs Linear ML surrogate (extrapolation-safe)")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Fish Ponds.csv")
    df = df[["TEMP", "PH", "TURBIDITY", "DO"]].dropna()
    df = df[(df["DO"] >= 0) & (df["PH"] >= 0)]
    return df

df = load_data()

# Subsample for stability + speed
df = df.sample(1000, random_state=42).reset_index(drop=True)

# --------------------------------------------------
# REFERENCE TEMPERATURE (AMBIENT PROXY)
# --------------------------------------------------
T_ref = df["TEMP"].mean()

# --------------------------------------------------
# WQI CALCULATION (GROUND TRUTH)
# --------------------------------------------------
def compute_wqi(temp, ph, turb, do, T_ref):
    Q_pH = (abs(ph - 7.0) / 1.5) * 100

    Q_DO = ((8.0 - do) / 3.0) * 100
    Q_DO = np.clip(Q_DO, 0, 100)

    Q_Turb = (turb / 5.0) * 100
    Q_Temp = (abs(temp - T_ref) / 5.0) * 100

    WQI = (
        0.35 * Q_DO +
        0.25 * Q_pH +
        0.25 * Q_Turb +
        0.15 * Q_Temp
    )
    return WQI

# --------------------------------------------------
# CALCULATED WQI FOR DATASET
# --------------------------------------------------
df["WQI_calc"] = df.apply(
    lambda r: compute_wqi(
        r["TEMP"], r["PH"], r["TURBIDITY"], r["DO"], T_ref
    ),
    axis=1
)

# --------------------------------------------------
# ML SURROGATE MODEL (LINEAR REGRESSION)
# --------------------------------------------------
X = df[["TEMP", "PH", "TURBIDITY", "DO"]]
y = df["WQI_calc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

df.loc[X_test.index, "WQI_pred"] = model.predict(X_test)

# --------------------------------------------------
# PERFORMANCE METRICS
# --------------------------------------------------
rmse = np.sqrt(mean_squared_error(
    df.loc[X_test.index, "WQI_calc"],
    df.loc[X_test.index, "WQI_pred"]
))

r2 = r2_score(
    df.loc[X_test.index, "WQI_calc"],
    df.loc[X_test.index, "WQI_pred"]
)

st.subheader("📊 Model Performance")
c1, c2 = st.columns(2)
c1.metric("RMSE", f"{rmse:.2f}")
c2.metric("R²", f"{r2:.3f}")

# --------------------------------------------------
# CORE VISUAL: CALCULATED vs PREDICTED
# --------------------------------------------------
st.subheader("📈 Calculated vs Predicted WQI")

fig, ax = plt.subplots()
ax.scatter(df["WQI_calc"], df["WQI_pred"], alpha=0.5)

max_val = max(df["WQI_calc"].max(), df["WQI_pred"].max())
ax.plot([0, max_val], [0, max_val], linestyle="--")

ax.set_xlabel("Calculated WQI")
ax.set_ylabel("Predicted WQI")

st.pyplot(fig)
st.caption("Dashed line = ideal agreement (Predicted = Calculated)")

# --------------------------------------------------
# FEATURE BOUNDS (DOMAIN CHECK)
# --------------------------------------------------
feature_bounds = {
    col: (X[col].min(), X[col].max())
    for col in X.columns
}

# --------------------------------------------------
# MANUAL SAMPLE COMPARISON
# --------------------------------------------------
st.divider()
st.subheader("🧪 Manual Sample Comparison")

temp = st.number_input("Temperature (°C)", 0.0, 50.0, float(round(T_ref, 1)))
ph = st.number_input("pH", 0.0, 14.0, 7.0)
turb = st.number_input("Turbidity (NTU)", 0.0, 100.0, 5.0)
do = st.number_input("DO (mg/L)", 0.0, 15.0, 5.0)

if st.button("Compare WQI"):
    # Ground truth
    wqi_calc = compute_wqi(temp, ph, turb, do, T_ref)

    # ML prediction (unbounded)
    wqi_pred = model.predict(pd.DataFrame(
        [[temp, ph, turb, do]],
        columns=["TEMP", "PH", "TURBIDITY", "DO"]
    ))[0]

    # Domain validity check
    out_of_domain = False
    for col, val in zip(
        ["TEMP", "PH", "TURBIDITY", "DO"],
        [temp, ph, turb, do]
    ):
        low, high = feature_bounds[col]
        if val < low or val > high:
            out_of_domain = True

    c1, c2 = st.columns(2)
    c1.metric("Calculated WQI", f"{wqi_calc:.2f}")
    c2.metric("Predicted WQI (ML)", f"{wqi_pred:.2f}")

    if out_of_domain:
        st.warning(
            "⚠️ Input lies outside the training data range. "
            "Linear model extrapolation shown."
        )
