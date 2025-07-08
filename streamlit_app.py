# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RationalQuadratic, WhiteKernel
import joblib

# --- Load Model ---
model_paths = ['gpr_model_D101330TT_Temoutlet_chamber.pkl', 'gpr_model_D102260TIC_CV_High_press_Steam_damper.pkl',
               'gpr_model_D102265TIC_CV_Low_press_Steam_damper.pkl', 'gpr_model_D102265TIC_PV_Temp_inlet_chamber.pkl', 'gpr_model_D102266TIC_Main_heater_dehumidifier.pkl']
models = [joblib.load(path) for path in model_paths]

# --- UI ---
st.set_page_config(page_title="Prediksi Dryer dengan GPR (j)", page_icon="🌀", layout="centered")

# Logo (ganti path dengan URL atau local file jika ingin upload logo)
st.image("logo.png", width=200)  # Sesuaikan dengan logo kamu
st.markdown("## 🌀 Prediksi Setting Dryer dari Nilai **GAS**")
st.caption("Masukkan nilai **GAS_MMBTU** untuk memprediksi pengaturan dryer lainnya.")

# Input Form
gas_input = st.number_input("GAS_MMBTU", min_value=0.0, max_value=100.0, step=0.01, value=16.5)

# Tombol Prediksi
if st.button("🔍 Prediksi Setting Dryer"):
    input_array = np.array([[gas_input]])
    parameter_names = [
        'D101330TT (Tem.outlet chamber)',
        'D102265TIC_PV (Temp. inlet chamber)',
        'D102260TIC_CV (High press. Steam damper)',
        'D102265TIC_CV (Low press. Steam damper)',
        'D102266TIC (Main heater dehumidifier)'
    ]
    predictions = [model.predict(input_array)[0] for model in models]


    # Tampilkan hasil
    st.success("Hasil Prediksi:")
    result_df = pd.DataFrame({
        "Parameter": parameter_names,
        "Predicted Value": [round(val, 2) for val in predictions]
    })
    st.table(result_df)
