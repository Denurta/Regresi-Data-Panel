import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# Fungsi untuk analisis data panel
def panel_analysis(data, dependent_var, independent_vars, group_var, method):
    # Konversi tipe data jika diperlukan
    data[dependent_var] = pd.to_numeric(data[dependent_var], errors='coerce')
    data[group_var] = pd.to_numeric(data[group_var], errors='coerce')
    for var in independent_vars:
        data[var] = pd.to_numeric(data[var], errors='coerce')

    data = data.dropna(subset=[dependent_var, group_var] + independent_vars)

    y = data[dependent_var]
    X = data[independent_vars]
    X = sm.add_constant(X)
    
    if method == 'CEM':
        model = sm.OLS(y, X).fit()
    elif method == 'FEM':
        model = sm.OLS(y, X).fit()
    elif method == 'REM':
        model = sm.MixedLM(y, X, groups=data[group_var]).fit()
    else:
        model = None
    return model

# Fungsi untuk visualisasi
def plot_visualization(data, dependent_var, independent_var1):
    fig, ax = plt.subplots()
    data.boxplot(column=dependent_var, by='group_var', ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.scatter(data[independent_var1], data[dependent_var])
    ax.set_xlabel(independent_var1)
    ax.set_ylabel(dependent_var)
    st.pyplot(fig)

# Halaman Analisis
def page_analysis():
    st.header('Analisis Data Panel')
    method = st.selectbox('Pilih Metode', ['CEM', 'FEM', 'REM'])
    
    uploaded_file = st.file_uploader("Upload file Excel", type="xlsx")
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file, engine='openpyxl')
        st.write(data.head())
        
        # Pilih kolom
        dependent_var = st.selectbox('Pilih Kolom Dependent Variable', data.columns)
        independent_vars = st.multiselect('Pilih Kolom Independent Variables', data.columns)
        group_var = st.selectbox('Pilih Kolom Group Variable', data.columns)
        
        if len(independent_vars) >= 1 and dependent_var and group_var:
            model = panel_analysis(data, dependent_var, independent_vars, group_var, method)
            if model is not None:
                st.write(model.summary())
                plot_visualization(data, dependent_var, independent_vars[0])
                
                # Simpan hasil prediksi ke dalam history
                predictions = model.predict()
                data['predictions'] = predictions
                history.append(data)
                st.write(predictions.head(5))
        else:
            st.write("Harap pilih variabel dependen, variabel independen, dan variabel grup.")

# Halaman History
def page_history():
    st.header('History Prediksi')
    if history:
        for i, hist in enumerate(history[-5:]):
            st.write(f"History {i+1}")
            st.write(hist.head())
            if st.button(f"Hapus History {i+1}", key=f'delete_{i}'):
                del history[i]

        if st.button("Hapus Semua History"):
            history.clear()
            st.write("Semua history telah dihapus")
    else:
        st.write("Belum ada history prediksi.")

# Main App
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman", ["Analisis", "History"])

# Inisialisasi history
if 'history' not in st.session_state:
    st.session_state['history'] = []

history = st.session_state['history']

if page == "Analisis":
    page_analysis()
elif page == "History":
    page_history()
