import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
import joblib
from datetime import datetime
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# KONFIGURASI HALAMAN & FOLDER
# ==========================================
st.set_page_config(
    page_title="SkyAir - Prediksi Kualitas Udara",
    page_icon="🌤️",
    layout="wide"
)

DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
MODEL_DIR = "models"
CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Inisialisasi Session State untuk Login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ==========================================
# FUNGSI EKSTRAKSI FITUR & MACHINE LEARNING
# ==========================================
def extract_image_features(image: Image.Image):
    """
    Mengekstrak fitur warna (RGB & HSV) dan kontras/haze dari foto langit.
    Langit berpolusi cenderung memiliki saturasi biru rendah dan kecerahan/kabut tinggi.
    """
    img = image.resize((256, 256))
    img_np = np.array(img.convert("RGB"))
    
    # Rata-rata & Standar Deviasi RGB
    r_mean, g_mean, b_mean = np.mean(img_np, axis=(0, 1))
    r_std, g_std, b_std = np.std(img_np, axis=(0, 1))
    
    # Rasio Biru terhadap Merah (Blue-Red Ratio - indikator kejernihan langit)
    br_ratio = b_mean / (r_mean + 1e-5)
    
    # Konversi ke HSV (Hue, Saturation, Value)
    hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h_mean, s_mean, v_mean = np.mean(hsv_img, axis=(0, 1))
    
    # Kontras (Laplacian variance untuk mendeteksi ketajaman/kabut)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    contrast = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        "r_mean": r_mean, "g_mean": g_mean, "b_mean": b_mean,
        "r_std": r_std, "g_std": g_std, "b_std": b_std,
        "br_ratio": br_ratio, "h_mean": h_mean, "s_mean": s_mean,
        "v_mean": v_mean, "contrast": contrast
    }

def init_dummy_data_and_models():
    """Membuat dataset awal & melatih model default jika belum ada"""
    if not os.path.exists(CSV_PATH):
        np.random.seed(42)
        n_samples = 50
        # Simulasi data: BR ratio & saturasi tinggi -> AQI rendah (udara bersih)
        br_ratio = np.random.uniform(0.7, 1.6, n_samples)
        s_mean = np.random.uniform(30, 180, n_samples)
        v_mean = np.random.uniform(120, 240, n_samples)
        
        df = pd.DataFrame({
            "filename": [f"sample_{i}.jpg" for i in range(n_samples)],
            "r_mean": np.random.uniform(100, 200, n_samples),
            "g_mean": np.random.uniform(120, 210, n_samples),
            "b_mean": np.random.uniform(130, 245, n_samples),
            "r_std": np.random.uniform(10, 40, n_samples),
            "g_std": np.random.uniform(10, 40, n_samples),
            "b_std": np.random.uniform(10, 40, n_samples),
            "br_ratio": br_ratio,
            "h_mean": np.random.uniform(90, 120, n_samples),
            "s_mean": s_mean,
            "v_mean": v_mean,
            "contrast": np.random.uniform(50, 500, n_samples),
        })
        # Formula sintetis AQI (0 - 300)
        df["aqi"] = np.clip(250 - (br_ratio * 80) - (s_mean * 0.4) + np.random.normal(0, 10, n_samples), 15, 300).round(1)
        df.to_csv(CSV_PATH, index=False)
        train_models()

def train_models():
    """Melatih ulang model SVR dan XGBoost dari dataset.csv"""
    df = pd.read_csv(CSV_PATH)
    feature_cols = [
        "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
        "br_ratio", "h_mean", "s_mean", "v_mean", "contrast"
    ]
    X = df[feature_cols]
    y = df["aqi"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train SVR
    svr = SVR(kernel='rbf', C=100, gamma='scale')
    svr.fit(X_scaled, y)
    
    # Train XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    xgb.fit(X_scaled, y)
    
    # Simpan model dan scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(svr, os.path.join(MODEL_DIR, "svr_model.pkl"))
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))
    
    # Hitung metrik evaluasi sederhana
    svr_mae = mean_absolute_error(y, svr.predict(X_scaled))
    xgb_mae = mean_absolute_error(y, xgb.predict(X_scaled))
    return svr_mae, xgb_mae

def get_aqi_category(aqi):
    """Mengembalikan kategori kualitas udara dan warna indikator"""
    if aqi <= 50:
        return "Baik (Good)", "🟢", "Udara bersih dan sehat untuk aktivitas luar ruangan."
    elif aqi <= 100:
        return "Sedang (Moderate)", "🟡", "Kualitas udara dapat diterima, namun sensitif bagi sebagian kecil orang."
    elif aqi <= 150:
        return "Tidak Sehat bagi Kelompok Sensitif", "🟠", "Penderita asma/gangguan pernapasan sebaiknya mengurangi aktivitas luar."
    elif aqi <= 200:
        return "Tidak Sehat (Unhealthy)", "🔴", "Setiap orang mulai dapat merasakan dampak kesehatan."
    else:
        return "Sangat Tidak Sehat / Berbahaya", "🟣", "Peringatan darurat kesehatan. Hindari aktivitas di luar ruangan."

# Jalankan inisialisasi saat aplikasi dibuka
init_dummy_data_and_models()

# ==========================================
# NAVIGASI SIDEBAR
# ==========================================
st.sidebar.title("🌤️ SkyAir Monitor")
st.sidebar.markdown("Prediksi Kualitas Udara Berbasis Citra Langit")

menu_options = ["Home", "Prediksi", "About"]
if st.session_state.logged_in:
    menu_options.append("Admin")
else:
    menu_options.append("Login")

choice = st.sidebar.radio("Navigasi Halaman", menu_options)

if st.session_state.logged_in:
    st.sidebar.success(f"Login sebagai: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# ==========================================
# HALAMAN 1: HOME
# ==========================================
if choice == "Home":
    st.title("🌤️ Sistem Prediksi Kualitas Udara Berbasis Foto Langit")
    st.markdown("""
    Selamat datang di **SkyAir**, platform pemantauan dan prediksi indeks kualitas udara (*Air Quality Index* / AQI) 
    menggunakan teknologi **Computer Vision** dan **Machine Learning** (**SVR** & **XGBoost**).
    """)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 📸 1. Unggah Foto Langit\nAmbil foto langit terbuka menggunakan kamera ponsel atau unggah gambar yang sudah ada.")
    with col2:
        st.info("### ⚙️ 2. Ekstraksi Fitur Visual\nSistem menganalisis spektrum warna RGB, saturasi HSV, rasio biru-merah, dan tingkat kabut (*haze*).")
    with col3:
        st.info("### 📊 3. Prediksi Ganda\nDapatkan estimasi nilai AQI secara instan dari perbandingan model **SVR** dan **XGBoost**.")

    st.divider()
    
    # Statistik Dataset Saat Ini
    df_current = pd.read_csv(CSV_PATH)
    st.subheader("📈 Ringkasan Dataset Sistem")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Data Latih", f"{len(df_current)} Sampel")
    m2.metric("Rata-rata AQI Dataset", f"{df_current['aqi'].mean():.1f}")
    m3.metric("Algoritma Aktif", "SVR & XGBoost")

# ==========================================
# HALAMAN 2: PREDIKSI
# ==========================================
elif choice == "Prediksi":
    st.title("🔍 Prediksi Kualitas Udara dari Foto Langit")
    st.write("Unggah foto langit untuk memprediksi nilai AQI menggunakan model **SVR** dan **XGBoost**.")
    
    uploaded_file = st.file_uploader("Pilih gambar langit (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col_img, col_res = st.columns([1, 1.2])
        
        with col_img:
            st.image(image, caption="Foto Langit yang Diunggah", use_container_width=True)
            
        with col_res:
            if st.button("🚀 Analisis & Prediksi Sekarang", type="primary", use_container_width=True):
                with st.spinner("Mengekstrak fitur visual dan menjalankan model..."):
                    # Ekstrak fitur
                    features = extract_image_features(image)
                    feature_df = pd.DataFrame([features])
                    
                    # Load model & scaler
                    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
                    svr = joblib.load(os.path.join(MODEL_DIR, "svr_model.pkl"))
                    xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
                    
                    X_scaled = scaler.transform(feature_df)
                    
                    pred_svr = max(0, float(svr.predict(X_scaled)[0]))
                    pred_xgb = max(0, float(xgb.predict(X_scaled)[0]))
                    avg_aqi = (pred_svr + pred_xgb) / 2
                    
                    kat, ikon, saran = get_aqi_category(avg_aqi)
                    
                    st.subheader(f"Status Udara: {ikon} {kat}")
                    st.write(saran)
                    
                    st.divider()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Prediksi SVR", f"{pred_svr:.1f} AQI")
                    c2.metric("Prediksi XGBoost", f"{pred_xgb:.1f} AQI")
                    c3.metric("Rata-rata Gabungan", f"{avg_aqi:.1f} AQI")
                    
                    with st.expander("Lihat Detail Fitur Citra yang Diekstrak"):
                        st.dataframe(feature_df.style.format("{:.2f}"))

# ==========================================
# HALAMAN 3: ABOUT
# ==========================================
elif choice == "About":
    st.title("ℹ️ Tentang Aplikasi & Metodologi")
    
    st.markdown("""
    ### Bagaimana Foto Langit Bisa Memprediksi Kualitas Udara?
    Partikel polutan di atmosfer seperti **PM2.5** dan **PM10** menyebarkan cahaya matahari (hamburan *Mie* dan *Rayleigh*). 
    Semakin tinggi konsentrasi polutan di udara:
    - Warna langit bergeser dari **biru pekat** menjadi **putih keabu-abuan atau kecokelatan**.
    - **Rasio Biru-Merah (*Blue-Red Ratio*)** menurun drastis.
    - **Saturasi warna** berkurang dan tingkat kekaburan (*haze*) meningkat.
    
    ### Algoritma yang Digunakan
    1. **Support Vector Regression (SVR):**
       Model regresi berbasis *Support Vector Machine* dengan kernel RBF (*Radial Basis Function*) yang efektif memetakan hubungan non-linear antara fitur warna citra dan nilai indeks kualitas udara pada dataset berukuran kecil hingga menengah.
    2. **Extreme Gradient Boosting (XGBoost):**
       Algoritma *ensemble learning* berbasis pohon keputusan (*decision tree*) yang menggabungkan banyak model lemah secara sekuensial untuk meminimalkan kesalahan prediksi (*residual error*) dengan akurasi tinggi.
    """)

# ==========================================
# HALAMAN 4: LOGIN
# ==========================================
elif choice == "Login":
    st.title("🔐 Login Administrator")
    st.write("Masuk untuk mengakses halaman manajemen dataset dan pelatihan ulang model.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Masuk")
        
        if submit:
            # Kredensial default (bisa diubah atau dihubungkan ke database/st.secrets)
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login berhasil! Mengalihkan ke halaman Admin...")
                st.rerun()
            else:
                st.error("Username atau Password salah! (Default: admin / admin123)")

# ==========================================
# HALAMAN 5: ADMIN (TAMBAH DATASET & TRAIN)
# ==========================================
elif choice == "Admin":
    if not st.session_state.logged_in:
        st.warning("Anda harus login terlebih dahulu untuk mengakses halaman ini.")
    else:
        st.title("🛠️ Dashboard Admin - Manajemen Dataset")
        
        tab1, tab2 = st.tabs(["➕ Tambah Dataset Baru", "📂 Kelola Dataset & Latih Model"])
        
        # TAB 1: UPLOAD FOTO & LABEL AQI
        with tab1:
            st.subheader("Tambahkan Foto Langit & Nilai AQI Aktual")
            with st.form("add_dataset_form", clear_on_submit=True):
                new_img = st.file_uploader("Upload Foto Langit", type=["jpg", "jpeg", "png"])
                actual_aqi = st.number_input("Nilai AQI Aktual (dari alat ukur / stasiun pemantau)", min_value=0.0, max_value=500.0, value=50.0, step=1.0)
                save_btn = st.form_submit_button("Simpan ke Dataset")
                
                if save_btn:
                    if new_img is not None:
                        img = Image.open(new_img)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"sky_{timestamp}_{new_img.name}"
                        save_path = os.path.join(IMG_DIR, filename)
                        img.save(save_path)
                        
                        # Ekstrak fitur & tambahkan ke CSV
                        feats = extract_image_features(img)
                        feats["filename"] = filename
                        feats["aqi"] = actual_aqi
                        
                        df = pd.read_csv(CSV_PATH)
                        df = pd.concat([df, pd.DataFrame([feats])], ignore_index=True)
                        df.to_csv(CSV_PATH, index=False)
                        
                        st.success(f"Data berhasil ditambahkan! Total dataset sekarang: {len(df)} baris.")
                    else:
                        st.error("Harap unggah file gambar terlebih dahulu.")
                        
        # TAB 2: LIHAT DATASET & RE-TRAIN MODEL
        with tab2:
            st.subheader("Dataset Tersimpan")
            df = pd.read_csv(CSV_PATH)
            st.dataframe(df.tail(15), use_container_width=True)
            
            col_train, col_dl = st.columns(2)
            with col_train:
                if st.button("🔄 Latih Ulang Model (SVR & XGBoost)", type="primary"):
                    with st.spinner("Melatih ulang model dengan dataset terbaru..."):
                        svr_mae, xgb_mae = train_models()
                        st.success(f"Model berhasil diperbarui! MAE SVR: {svr_mae:.2f} | MAE XGBoost: {xgb_mae:.2f}")
            
            with col_dl:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh Dataset (CSV)",
                    data=csv_data,
                    file_name="dataset_kualitas_udara.csv",
                    mime="text/csv"
                )