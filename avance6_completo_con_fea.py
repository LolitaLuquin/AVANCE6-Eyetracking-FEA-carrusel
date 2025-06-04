# AVANCE6 en Streamlit con carrusel de gráficas agrupadas

import streamlit as st
import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from io import StringIO, BytesIO
from collections import Counter

# --- FUNCIONES DE CARGA ---
def make_unique(headers):
    counts = Counter()
    new_headers = []
    for h in headers:
        counts[h] += 1
        if counts[h] > 1:
            new_headers.append(f"{h}_{counts[h]-1}")
        else:
            new_headers.append(h)
    return new_headers

def read_imotions_csv(file, participant_name, header_index, tipo):
    content = file.read().decode('utf-8')
    lines = content.splitlines()
    headers = make_unique(lines[header_index].strip().split(","))
    data = "\n".join(lines[header_index + 1:])
    df = pd.read_csv(StringIO(data), names=headers)
    df["Participant"] = participant_name
    df["Tipo"] = tipo
    return df

def upload_and_concat(tipo, header_index):
    uploaded_files = st.file_uploader(f"Sube archivos de {tipo} (CSV)", accept_multiple_files=True, type="csv")
    dfs = []
    if uploaded_files:
        for file in uploaded_files:
            participant = file.name.replace(".csv", "").strip()
            df = read_imotions_csv(file, participant, header_index, tipo)
            dfs.append(df)
        df_merged = pd.concat(dfs, ignore_index=True)
        st.success(f"{tipo} fusionado con {len(dfs)} archivo(s).")
        st.download_button(
            f"Descargar {tipo} mergeado",
            df_merged.to_csv(index=False).encode(),
            file_name=f"{tipo.lower()}_merged.csv",
            mime='text/csv'
        )
        return df_merged
    return pd.DataFrame()

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(layout="wide")
st.title("AVANCE6 - Análisis de Eyetracking con carrusel")

with st.sidebar:
    st.header("Carga de archivos")
    df_et = upload_and_concat("Eyetracking", 25)
    df_fea = upload_and_concat("FEA", 25)
    df_gsr = upload_and_concat("GSR", 27)

# --- ANÁLISIS DE EYETRACKING ---
if not df_et.empty:
    st.header("Análisis de Eyetracking")

    df_et["ET_TimeSignal"] = pd.to_numeric(df_et["ET_TimeSignal"], errors="coerce")
    df_et = df_et.dropna(subset=["ET_TimeSignal", "SourceStimuliName"])

    # --- TABLA DE ESTADÍSTICOS ---
    tabla_et = df_et.groupby("SourceStimuliName").agg(
        Tiempo_Medio=("ET_TimeSignal", "mean"),
        Desviacion_Estandar=("ET_TimeSignal", "std"),
        Conteo=("ET_TimeSignal", "count")
    ).reset_index()

    st.subheader("Tabla de estadísticos por Estímulo")
    st.dataframe(tabla_et)
    st.download_button(
        "Descargar tabla estadística",
        tabla_et.to_csv(index=False).encode(),
        file_name="tabla_eyetracking.csv",
        mime='text/csv'
    )

    # --- ANOVA ---
    estimulos = df_et["SourceStimuliName"].unique()
    data_por_estimulo = [df_et[df_et["SourceStimuliName"] == stim]["ET_TimeSignal"] for stim in estimulos]
    anova_result = stats.f_oneway(*data_por_estimulo)
    f_stat = anova_result.statistic
    p_value = anova_result.pvalue
    f_squared = (f_stat * (len(estimulos) - 1)) / (len(df_et) - len(estimulos)) if len(estimulos) > 1 else None

    estad_txt = f"ANOVA F-statistic: {f_stat:.4f}\n"
    estad_txt += f"p-value: {p_value:.4e}\n"
    if f_squared:
        estad_txt += f"F-squared: {f_squared:.4f}\n"

    st.text_area("Estadísticos", estad_txt, height=100)
    st.download_button(
        "Descargar estadísticos",
        estad_txt,
        file_name="estadisticos_eyetracking.txt"
    )

    # --- GRÁFICAS ---
    st.subheader("Gráficas de Eyetracking")
    sns.set(style="whitegrid")

    comparativas = []
    distribuciones = []

    fig1, ax1 = plt.subplots(figsize=(10,6))
    sns.barplot(data=df_et, x="SourceStimuliName", y="ET_TimeSignal", ci="sd", capsize=0.1, ax=ax1)
    ax1.set_title("Tiempo promedio por Estímulo")
    ax1.set_ylabel("Tiempo de permanencia")
    ax1.tick_params(axis='x', rotation=45)
    comparativas.append(fig1)

    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.violinplot(data=df_et, x="SourceStimuliName", y="ET_TimeSignal", ax=ax2)
    ax2.set_title("Distribución del Tiempo por Estímulo")
    ax2.tick_params(axis='x', rotation=45)
    distribuciones.append(fig2)

    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.boxplot(data=df_et, x="SourceStimuliName", y="ET_TimeSignal", ax=ax3)
    ax3.set_title("Boxplot por Estímulo")
    ax3.tick_params(axis='x', rotation=45)
    distribuciones.append(fig3)

    colores = sns.color_palette("tab10", n_colors=len(estimulos))
    for idx, stim in enumerate(estimulos):
        subset = df_et[df_et["SourceStimuliName"] == stim]
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(subset["ET_TimeSignal"], kde=True, color=colores[idx], ax=ax)
        ax.set_title(f"Histograma - {stim}")
        ax.set_xlabel("Tiempo de permanencia")
        distribuciones.append(fig)

    # --- CARRUSEL AGRUPADO ---
    grupo = st.radio("Selecciona grupo de gráficas:", ["Comparativas", "Distribuciones"])
    if grupo == "Comparativas":
        opciones = [f"{i+1}. {fig.axes[0].get_title()}" for i, fig in enumerate(comparativas)]
        seleccion = st.selectbox("Selecciona una gráfica comparativa:", opciones, key="comp")
        st.pyplot(comparativas[opciones.index(seleccion)])
    elif grupo == "Distribuciones":
        opciones = [f"{i+1}. {fig.axes[0].get_title()}" for i, fig in enumerate(distribuciones)]
        seleccion = st.selectbox("Selecciona una gráfica de distribución:", opciones, key="dist")
        st.pyplot(distribuciones[opciones.index(seleccion)])


    # --- EXPORTAR TODAS LAS GRÁFICAS A ZIP ---
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        zip_path = f"{tmpdir}/graficas_eyetracking.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for i, fig in enumerate(comparativas + distribuciones):
                fig_path = f"{tmpdir}/grafica_{i+1}.png"
                fig.savefig(fig_path)
                zipf.write(fig_path, arcname=os.path.basename(fig_path))

        with open(zip_path, "rb") as f:
            st.download_button("Descargar todas las gráficas (ZIP)", f.read(), file_name="graficas_eyetracking.zip")

# --- ANÁLISIS DE FEA ---
with st.expander("Análisis de FEA (emociones y valencia)", expanded=False):
    if not df_fea.empty:
        st.subheader("Cálculos base de FEA")

        emociones = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']
        df_fea["Engagement_Promedio"] = df_fea[emociones].mean(axis=1)
        df_fea["Valence_Class"] = df_fea["Valence"].apply(lambda x: "Positiva" if x > 0 else ("Negativa" if x < 0 else "Neutra"))

        # Tabla resumen
        tabla_fea = df_fea.groupby("SourceStimuliName").agg({
            "Valence": ["mean", "std"],
            "Engagement_Promedio": ["mean", "std"]
        }).reset_index()
        tabla_fea.columns = ["Estímulo", "Valencia_Media", "Valencia_SD", "Engagement_Media", "Engagement_SD"]

        st.subheader("Tabla resumen FEA")
        st.dataframe(tabla_fea)
        st.download_button("Descargar tabla FEA", tabla_fea.to_csv(index=False).encode(), file_name="tabla_resumen_fea.csv", mime="text/csv")

        # Estadísticos ANOVA
        from scipy import stats
        import numpy as np

        def f_squared(anova_result):
            return anova_result.statistic / (anova_result.statistic + df_fea.shape[0] - 1) if not np.isnan(anova_result.statistic) else np.nan

        stimuli_groups_val = [g["Valence"].dropna() for _, g in df_fea.groupby("SourceStimuliName") if len(g["Valence"].dropna()) > 1]
        stimuli_groups_eng = [g["Engagement_Promedio"].dropna() for _, g in df_fea.groupby("SourceStimuliName") if len(g["Engagement_Promedio"].dropna()) > 1]

        anova_valencia = stats.f_oneway(*stimuli_groups_val) if len(stimuli_groups_val) > 1 else None
        anova_engagement = stats.f_oneway(*stimuli_groups_eng) if len(stimuli_groups_eng) > 1 else None

        estadisticos_fea = {
            "Valencia": {
                "F": anova_valencia.statistic if anova_valencia else "No aplica",
                "p-value": anova_valencia.pvalue if anova_valencia else "No aplica",
                "F²": f_squared(anova_valencia) if anova_valencia else "No aplica"
            },
            "Engagement": {
                "F": anova_engagement.statistic if anova_engagement else "No aplica",
                "p-value": anova_engagement.pvalue if anova_engagement else "No aplica",
                "F²": f_squared(anova_engagement) if anova_engagement else "No aplica"
            }
        }

        est_txt = ""
        for var, stats_ in estadisticos_fea.items():
            est_txt += f"{var}:
"
            for k, v in stats_.items():
                est_txt += f"  {k}: {v}
"
            est_txt += "\n"
        st.text_area("Estadísticos FEA", est_txt, height=140)
        st.download_button("Descargar estadísticos FEA", est_txt, file_name="estadisticos_fea.txt")

        # Filtro por emoción específica
        st.subheader("Filtrar por emoción específica")
        emocion_seleccionada = st.selectbox("Selecciona una emoción para analizar:", emociones)
        fig_emocion, ax_em = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df_fea, x="SourceStimuliName", y=emocion_seleccionada, ax=ax_em)
        ax_em.set_title(f"Boxplot de {emocion_seleccionada} por Estímulo")
        ax_em.tick_params(axis='x', rotation=45)
        st.pyplot(fig_emocion)

        # Carrusel FEA por tipo
        st.subheader("Carrusel de gráficas FEA")
        comparativas_fea = []
        distribuciones_fea = []

        fig_val, ax_val = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df_fea, x="SourceStimuliName", y="Valence", ax=ax_val)
        ax_val.set_title("Boxplot de Valencia por Estímulo")
        ax_val.tick_params(axis='x', rotation=45)
        comparativas_fea.append(fig_val)

        fig_eng, ax_eng = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df_fea, x="SourceStimuliName", y="Engagement_Promedio", ax=ax_eng)
        ax_eng.set_title("Boxplot de Engagement Promedio por Estímulo")
        ax_eng.tick_params(axis='x', rotation=45)
        comparativas_fea.append(fig_eng)

        colores = sns.color_palette("tab10", n_colors=len(df_fea["SourceStimuliName"].unique()))
        for idx, stim in enumerate(df_fea["SourceStimuliName"].unique()):
            subset = df_fea[df_fea["SourceStimuliName"] == stim]
            fig, ax = plt.subplots(figsize=(8,5))
            sns.histplot(subset["Valence"], kde=True, color=colores[idx], ax=ax)
            ax.set_title(f"Histograma de Valencia - {stim}")
            ax.set_xlabel("Valencia")
            distribuciones_fea.append(fig)

        grupo_fea = st.radio("Selecciona grupo de gráficas FEA:", ["Comparativas", "Distribuciones"], key="fea_grupo")
        if grupo_fea == "Comparativas":
            opciones = [f"{i+1}. {fig.axes[0].get_title()}" for i, fig in enumerate(comparativas_fea)]
            seleccion = st.selectbox("Selecciona una gráfica comparativa FEA:", opciones, key="comp_fea")
            st.pyplot(comparativas_fea[opciones.index(seleccion)])
        elif grupo_fea == "Distribuciones":
            opciones = [f"{i+1}. {fig.axes[0].get_title()}" for i, fig in enumerate(distribuciones_fea)]
            seleccion = st.selectbox("Selecciona una gráfica de distribución FEA:", opciones, key="dist_fea")
            st.pyplot(distribuciones_fea[opciones.index(seleccion)])

        # Botón para descargar todas las gráficas FEA
        from tempfile import TemporaryDirectory
        import os

        with TemporaryDirectory() as tmpdir:
            zip_path = f"{tmpdir}/graficas_fea.zip"
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for i, fig in enumerate(comparativas_fea + distribuciones_fea + [fig_emocion]):
                    path = f"{tmpdir}/grafica_fea_{i+1}.png"
                    fig.savefig(path)
                    zipf.write(path, arcname=os.path.basename(path))
            with open(zip_path, "rb") as f:
                st.download_button("Descargar todas las gráficas FEA (ZIP)", f.read(), file_name="graficas_fea.zip")
