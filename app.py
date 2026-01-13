import streamlit as st
import pandas as pd
import numpy as np

from backend_forecast import predict_from_future_file

# =========================
# CONFIG
# =========================
FUTURE_CSV = "future_month.csv"
TRAIN_CSV  = "mydata.csv"
MODEL_PKL  = "rf_best_model.pkl"

st.set_page_config(
    page_title="Solar Energy Forecast (RF + Lags)",
    page_icon="☀️",
    layout="centered"
)

st.title("☀️ Solar Energy Forecast – RF + Lag Model")
st.write(
    "هذا التطبيق يستخدم نموذج Random Forest مدرّب على lag features "
    "للتنبؤ بالطاقة الشمسية لفترة مستقبلية اعتمادًا على ملف future_month.csv."
)

# =========================
# 1) Panel inputs
# =========================
st.subheader("🔌 Solar Panel Inputs")

col1, col2 = st.columns(2)
with col1:
    num_panels = st.number_input(
        "Number of panels (عدد الألواح)",
        min_value=1,
        value=10,
        step=1
    )
    panel_area = st.number_input(
        "Area of **one** panel (m²) – مساحة اللوح الواحد",
        min_value=0.1,
        value=1.7,
        step=0.1
    )

with col2:
    panel_eff_percent = st.number_input(
        "Panel efficiency (%) – كفاءة اللوح",
        min_value=1.0,
        max_value=100.0,
        value=18.0,
        step=1.0
    )
    performance_ratio = st.number_input(
        "Performance ratio (0–1, e.g. 0.9)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05
    )

panel_efficiency = panel_eff_percent / 100.0  # نحوله من % إلى 0–1

st.markdown("---")

# =========================
# 2) Forecast horizon
# =========================
st.subheader("⏱️ Forecast Horizon – مدة التنبؤ")

horizon_label = st.radio(
    "اختر مدة التنبؤ:",
    [
        "3 days",
        "1 week",
        "2 weeks",
        "1 month (full future_month.csv)"
    ],
    index=1
)

# عدد الساعات لكل خيار
if horizon_label == "3 days":
    horizon_hours = 3 * 24
elif horizon_label == "1 week":
    horizon_hours = 7 * 24
elif horizon_label == "2 weeks":
    horizon_hours = 14 * 24
else:
    horizon_hours = None  # يعني استخدم كل البيانات في future_month.csv

st.markdown("---")

# =========================
# 3) Run prediction
# =========================
if st.button("🔮 Run Forecast"):
    try:
        # backend يعطيك كل الفترات في future_month.csv
        result = predict_from_future_file(
            future_csv_path=FUTURE_CSV,
            training_csv_path=TRAIN_CSV,
            model_path=MODEL_PKL,
            num_panels=num_panels,
            panel_area=panel_area,
            panel_efficiency=panel_efficiency,
            performance_ratio=performance_ratio,
        )

        preds_df = result["predictions"].copy()  # index = timestamp

        # نتأكد إنه datetime index
        preds_df.index = pd.to_datetime(preds_df.index)

        # لو المستخدم اختار فترة أقل من شهر → نعمل slicing
        if horizon_hours is not None and horizon_hours < len(preds_df):
            preds_sel = preds_df.iloc[:horizon_hours].copy()
        else:
            preds_sel = preds_df.copy()

        # نحسب مجموع الطاقة فقط للفترة المختارة
        total_energy_kwh = preds_sel["energy_kwh"].sum()

        # تجميع على مستوى الأيام
        daily_energy = preds_sel["energy_kwh"].resample("D").sum().to_frame(name="energy_kwh")
        daily_energy["avg_irradiance"] = preds_sel["predicted_irradiance"].resample("D").mean()

        # =========================
        # 4) Display results
        # =========================
        st.markdown("## 📈 Results")

        st.metric(
            label=f"Total predicted energy for {horizon_label}",
            value=f"{total_energy_kwh:.2f} kWh"
        )

        st.write("### Sample of hourly predictions (first 30 rows)")
        st.dataframe(
            preds_sel.head(30).rename(
                columns={
                    "predicted_irradiance": "Irradiance (W/m²)",
                    "energy_kwh": "Energy (kWh)"
                }
            )
        )

        st.write("### Daily Energy (kWh) and Average Irradiance")
        st.dataframe(
            daily_energy.rename(
                columns={
                    "energy_kwh": "Energy (kWh)",
                    "avg_irradiance": "Avg Irradiance (W/m²)"
                }
            ).style.format({"Energy (kWh)": "{:.2f}", "Avg Irradiance (W/m²)": "{:.1f}"})
        )

        st.write("### Energy per day (bar chart)")
        st.bar_chart(daily_energy["energy_kwh"])

        st.write("### Predicted irradiance over time (line chart)")
        st.line_chart(preds_sel["predicted_irradiance"])

    except Exception as e:
        st.error(f"Error during forecast: {e}")
