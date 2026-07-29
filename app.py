import sys
import os

# Tambah path biar bisa import eda.py & prediction.py dari folder yang sama
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import eda
import prediction


ASSET_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title='YourMajor Recommendation',
    layout='wide'
)

def main():
    # Logo
    st.sidebar.image('logo.png', use_container_width=True)
    st.sidebar.markdown('---')

    # Custom CSS sidebar
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] .sidebar-title {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 0.5em;
        color: #ffffff;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #b0b0b0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #0f3460 !important;
        border: 1px solid #e94560 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"]:hover {
        border-color: #533483 !important;
    }
    [data-testid="stSidebar"] p {
        color: #cccccc !important;
    }
    </style>
    """, unsafe_allow_html=True)

    page = st.sidebar.selectbox(
        'Pilih Halaman',
        ('Home', 'EDA', 'Prediksi')
    )

    st.sidebar.markdown('---')
    st.sidebar.markdown('Rekomendasi Jurusan & Universitas')
    st.sidebar.markdown('Berdasarkan Nilai UTBK 2019 Saintek')
    st.sidebar.markdown('')
    st.sidebar.markdown('by **Muhammad Izzat - Ridhan Firdaus - Nicholas Calvin**')

    if page == 'Home':
        show_home()
    elif page == 'EDA':
        eda.run()
    elif page == 'Prediksi':
        prediction.run()


def show_home():
    st.title('🎓 Your Major Recommendation')
    st.markdown('---')

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Welcome!

        Aplikasi ini membantu calon mahasiswa menemukan **jurusan dan universitas**
        yang paling sesuai dengan **nilai UTBK** mereka.

        ### Workflow

        Aplikasi menggunakan pendekatan **K-Nearest Neighbors** berdasarkan nilai UTBK:

        - **Input** 8 nilai UTBK
        - **Cari** 100 siswa paling mirip dari 86.569 data UTBK 2019 Saintek
        - **Ranking** jurusan berdasarkan kemiripan

        ### Pages
        
        1. **Home** Beranda & Informasi Project
        1. **EDA** Eksplorasi & Visualisasi Data
        2. **Prediksi** Rekomendasi Jurusan
        """)

    with col2:
        st.markdown("""
        **Info Dataset**

        | Item | Detail |
        |:-----|:------:|
        | **Sumber** | UTBK 2019 Saintek |
        | **Siswa** | 86.569 |
        | **Nilai** | 8 mata uji |
        | **Jurusan** | 279 pilihan |
        | **Bidang** | 7 kategori |

        """)

    st.markdown('---')
    st.markdown('**2026 YourMajor**')


if __name__ == '__main__':
    main()
