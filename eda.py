import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

sns.set_style('whitegrid')

# Path ke model pipeline
PIPELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'your_major_recomendation_pipeline.pkl')

@st.cache_resource
def load_model():
    """Load pipeline artefacts"""
    df = joblib.load(PIPELINE_PATH)['dataset_lengkap']
    return df

def run():
    st.title('📊 Exploratory Data Analysis (EDA)')
    st.markdown('''
    Halaman ini menampilkan hasil eksplorasi data dari **UTBK 2019 Saintek**
    yang digunakan untuk sistem rekomendasi.
    ''')
    st.markdown('---')

    df = load_model()

    # 1. Distribusi per kategori
    st.subheader('1. Distribusi Siswa per Bidang')

    cat_dist = df['kategori_jurusan'].value_counts()
    total = len(df)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.Set2(range(len(cat_dist)))
        bars = ax.bar(cat_dist.index, cat_dist.values, color=colors)
        ax.set_title('Jumlah Siswa per Bidang', fontsize=14, fontweight='bold')
        ax.set_ylabel('Jumlah Siswa')
        ax.tick_params(axis='x', rotation=30)

        for bar, val in zip(bars, cat_dist.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                    f'{val:,}\n({val/total*100:.1f}%)', ha='center', fontsize=9)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('**Distribusi:**')
        for cat, cnt in cat_dist.items():
            st.markdown(f'- **{cat}**: {cnt:,} ({cnt/total*100:.1f}%)')
        st.markdown(f'**Total: {total:,} siswa**')

    # 2. Statistik Nilai per Kategori
    st.subheader('2. Rata-rata Nilai per Bidang')

    nilai_cols = ['nilai_biologi', 'nilai_fisika', 'nilai_kimia', 'nilai_matematika',
                  'nilai_kmb', 'nilai_kpu', 'nilai_kua', 'nilai_ppu']
    cols_short = ['Bio', 'Fis', 'Kim', 'Mat', 'KMB', 'KPU', 'KUA', 'PPU']

    avg_df = df.groupby('kategori_jurusan')[nilai_cols].mean().round(0)
    avg_df.columns = cols_short

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(avg_df.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(cols_short)))
    ax.set_xticklabels(cols_short)
    ax.set_yticks(range(len(avg_df.index)))
    ax.set_yticklabels(avg_df.index)
    ax.set_title('Rata-rata Nilai per Bidang', fontsize=14, fontweight='bold')

    for i in range(len(avg_df.index)):
        for j in range(len(cols_short)):
            ax.text(j, i, f'{int(avg_df.values[i, j])}', ha='center', va='center', fontsize=9, color='black')

    plt.colorbar(im, ax=ax, label='Nilai Rata-rata')
    st.pyplot(fig)
    plt.close()

    st.markdown('''
    > 💡 **Insight:** Rata-rata nilai antar kategori hampir sama (526-557).
    > Inilah kenapa **classifier tidak bisa akurat** — pilihan jurusan lebih ditentukan
    > oleh **preferensi** daripada nilai semata.
    ''')

    # 3. Top Jurusan per Bidang
    st.subheader('3. Top 5 Jurusan per Bidang')

    for cat in df['kategori_jurusan'].unique():
        df_cat = df[df['kategori_jurusan'] == cat]
        top5 = df_cat['jurusan_tujuan'].value_counts().head(5)

        st.markdown(f'**{cat}** ({len(df_cat):,} siswa)')
        for rank, (j, cnt) in enumerate(top5.items(), 1):
            st.markdown(f'&nbsp;&nbsp;{rank}. {j} — {cnt:,} siswa')
        st.markdown('')

    st.markdown('---')
    st.markdown('**© 2026 Muhammad Izzat — Final Project**')
