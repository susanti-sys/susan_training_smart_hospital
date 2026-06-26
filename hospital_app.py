import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os



#Sebelum membuat isi website, kita atur dulu judul tab browser, icon, dan layout halaman.
st.set_page_config(
    page_title="Smart Hospital Patient Navigator",
    page_icon="🏥",
    layout="wide"
)

#Kita membuka file model AI yang sudah kita training sebelumnya.
@st.cache_resource
def load_model():
    with open("hospital_model.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_model()

#Di dalam file pickle terdapat banyak objek, jadi kita keluarkan satu per satu.
model         = bundle["model"]
scaler        = bundle["scaler"]
features      = bundle["features"]
cols_to_scale = bundle["cols_to_scale"]
dept_map_inv  = bundle["dept_map_inv"]

#digunakan untuk mengubah teks menjadi angka yang bisa dipahami model.
# contoh 
# Male → 1
# Female → 0
gender_map = bundle["gender_map"]
temp_map   = bundle["temp_map"]
hr_map     = bundle["hr_map"]
dur_map    = bundle["dur_map"]
cc_map     = bundle["cc_map"]

# Kita membuat database kecil untuk setiap departemen rumah sakit.
# Isinya:
# icon
# deskripsi
# langkah selanjutnya
DEPT_INFO = {
    'Respiratory Medicine': {
        'icon':'🫁',
        'desc':'Specialises in conditions affecting the lungs and airways.',
        'next':[
            'Visit Level 2, Wing B',
            'Estimated wait: 15–25 min',
            'Please wear a mask'
        ]
    },

    'Cardiology': {
        'icon':'❤️',
        'desc':'Specialises in heart and cardiovascular conditions.',
        'next':[
            'Visit Level 3, Wing A',
            'Estimated wait: 20–30 min',
            'Bring any previous ECG reports'
        ]
    },

    'Gastroenterology': {
        'icon':'🫃',
        'desc':'Specialises in digestive system and abdominal conditions.',
        'next':[
            'Visit Level 1, Wing C',
            'Estimated wait: 10–20 min',
            'Avoid eating before consultation'
        ]
    },

    'Neurology': {
        'icon':'🧠',
        'desc':'Specialises in brain, spine, and nervous system conditions.',
        'next':[
            'Visit Level 4, Wing A',
            'Estimated wait: 25–35 min',
            'Bring list of current medications'
        ]
    },

    'General Medicine': {
        'icon':'🩺',
        'desc':'Handles general health concerns and non-specialist conditions.',
        'next':[
            'Visit Level 1, Wing A',
            'Estimated wait: 10–15 min',
            'Registration desk is open 24/7'
        ]
    },

    'Dermatology': {
        'icon':'🔬',
        'desc':'Specialises in skin, hair, and nail conditions.',
        'next':[
            'Visit Level 2, Wing D',
            'Estimated wait: 15–20 min',
            'Bring photos of affected area if possible'
        ]
    }
}

#Ini bagian tampilan paling atas yang akan dilihat user.
st.title("🏥 Smart Hospital Patient Navigator")
st.caption("Future Classroom • Machine Learning")
st.write(
    "Find the Right Department for Your Symptoms"
)
st.divider()

with st.form("triage_form"):
    st.subheader("1️⃣ Symptoms")
    
    #Untuk membagi layar menjadi 4 kolom.
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        fever = st.checkbox("🌡️ Fever")
        cough = st.checkbox("🤧 Cough")
    with c2:
        headache = st.checkbox("🤕 Headache")
        chest_pain = st.checkbox("💔 Chest Pain")

    with c3:
        stomach_pain = st.checkbox("🤢 Stomach Pain")
        shortness_breath = st.checkbox("😮 hortness of Breath")

    with c4:
        nausea_vomiting = st.checkbox("🤮 Nausea / Vomiting")
        dizziness = st.checkbox("😵 Dizziness")

    skin_rash = st.checkbox("🔴 Skin Rash")

    st.divider()

    st.subheader("2️⃣ Duration")

    col1, col2 = st.columns(2)
    with col1:
        chief_complaint = st.selectbox(
            "Chief Complaint",
            options=list(cc_map.keys())
        )

    with col2:
        duration = st.selectbox(
            "Duration",
            options=list(dur_map.keys()),
            index=1
        )

    st.divider()

    st.subheader("3️⃣ Severity")

    col1, col2 = st.columns(2)

    with col1:
        temperature_level = st.selectbox(
            "Temperature",
            options=list(temp_map.keys()),
            index=1
        )

    with col2:
        heart_rate_level = st.selectbox(
            "Heart Rate",
            options=list(hr_map.keys()),
            index=1
        )

    st.divider()

    st.subheader("4️⃣ Medical History")

    col1, col2, col3 = st.columns(3)

    with col1:
        hypertension = st.checkbox("🩺 High Blood Pressure")

    with col2:
        heart_disease = st.checkbox("❤️ Heart Disease")

    with col3:
        asthma = st.checkbox("💨 Asthma")

    st.divider()

    st.subheader("5️⃣ Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=35
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )


    submitted = st.form_submit_button(
        "Get AI Recommendation"
    )

# =========================
# PREDICTION
# =========================

if submitted:
    patient = pd.DataFrame([{
        'age': age,
        'gender': gender_map.get(gender, 0),

        'fever': int(fever),
        'cough': int(cough),
        'headache': int(headache),
        'chest_pain': int(chest_pain),
        'stomach_pain': int(stomach_pain),
        'shortness_breath': int(shortness_breath),
        'nausea_vomiting': int(nausea_vomiting),
        'dizziness': int(dizziness),
        'skin_rash': int(skin_rash),

        'temperature_level': temp_map.get(
            temperature_level, 1
        ),

        'heart_rate_level': hr_map.get(
            heart_rate_level, 1
        ),

        'duration': dur_map.get(
            duration, 1
        ),

        'asthma': int(asthma),
        'hypertension': int(hypertension),
        'heart_disease': int(heart_disease),

        'chief_complaint': cc_map.get(
            chief_complaint, 9
        )
    }])

    # Membuat salinan data pasien agar data asli tidak berubah
    patient_scaled = patient.copy()

    # Melakukan normalisasi (scaling) pada kolom-kolom tertentu
    # supaya format datanya sama seperti saat model Machine Learning dilatih
    patient_scaled[cols_to_scale] = scaler.transform(
        patient[cols_to_scale]
    )

    # Melakukan prediksi department tujuan pasien
    # Hasilnya berupa angka (misalnya 0, 1, 2, dst.)
    pred = model.predict(
        patient_scaled[features]
    )[0]

    # Mengambil probabilitas (tingkat keyakinan) untuk setiap department
    # Hasilnya berupa list, misalnya [0.10, 0.70, 0.05, ...]
    proba = model.predict_proba(
        patient_scaled[features]
    )[0]

    # Mengubah hasil prediksi dari angka menjadi nama department
    # Contoh: 1 -> Cardiology
    dept_name = dept_map_inv[pred]

    # Mengambil confidence dari department yang diprediksi
    # lalu mengubahnya menjadi persen
    confidence = proba[pred] * 100

    # Mengambil informasi lengkap mengenai department tersebut
    # seperti ikon, deskripsi, atau rekomendasi yang akan ditampilkan
    info = DEPT_INFO[dept_name]

    st.divider()

    st.header("🤖 AI Recommendation")
    st.success(
        f"{info['icon']} {dept_name}"
    )
    st.write(info["desc"])

   #st.metric() digunakan untuk menampilkan satu nilai utama dalam bentuk kartu (card).
    st.metric(
        label="Prediction Confidence",
        value=f"{confidence:.1f}%"
    )
    st.subheader("📍 Next Steps")

    for step in info["next"]:
        st.write("•", step)

    st.warning(
        "This is an AI suggestion, not a medical diagnosis. Please consult a doctor."
    )

    st.divider()

    st.subheader("📊 Confidence by Department")

    # sorted() digunakan untuk mengurutkan semua department. Data yang diurutkan berasal dari dept_map_inv.items(), yaitu pasangan (kode_department, nama_department). Dengan key=lambda x: proba[x[0]], program mengambil probabilitas setiap department sebagai dasar pengurutan. Terakhir, reverse=True membuat urutan dimulai dari probabilitas yang paling tinggi sehingga department yang paling mungkin muncul di posisi pertama.

    sorted_depts = sorted(
        dept_map_inv.items(),
        key=lambda x: proba[x[0]],
        reverse=True
    )

    for idx, dname in sorted_depts:

        pct = proba[idx] * 100

        st.write(
            f"{DEPT_INFO[dname]['icon']} {dname} ({pct:.1f}%)"
        )

        st.progress(float(pct) / 100)

    
    st.info(
        "Model: KNN (k=7) • 102,000 patients • 99.5% accuracy"
    )

