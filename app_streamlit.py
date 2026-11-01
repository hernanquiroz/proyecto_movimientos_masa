import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
import folium
from streamlit_folium import st_folium

@st.cache_resource
def get_model():
    return load_model('modelo_movimientos_masa_pipeline')

model = get_model()

st.set_page_config(page_title="Predictor de Movimientos en Masa", page_icon="🏔️", layout="wide")

st.title("🏔️ Sistema de Pronóstico de Movimientos en Masa con Análisis Espacial")

col_form, col_map = st.columns([1, 1])

with col_form:
    st.subheader("Parámetros de Entrada")
    with st.form("prediction_form"):
        coord_x = st.number_input("Coordenada X (Longitud)", value=-75.5, format="%.6f")
        coord_y = st.number_input("Coordenada Y (Latitud)", value=6.2, format="%.6f")
        f35_dov = st.number_input("F35DOV_TIP (Código Geotécnico)", value=69, step=1)
        subtipo_mov = st.number_input("Subtipo Movimiento (Código)", value=4, step=1)

        submit_button = st.form_submit_button(label="Ejecutar Pronóstico")

    if submit_button:
        input_data = pd.DataFrame({
            'x': [coord_x],
            'y': [coord_y],
            'F35DOV_TIP': [f35_dov],
            'Subtipo_Movimiento': [subtipo_mov]
        })

        predictions = predict_model(model, data=input_data)
        predicted_label = predictions['prediction_label'].iloc[0]
        predicted_score = predictions['prediction_score'].iloc[0] * 100

        st.success(f"**Tipo Predicho:** `{predicted_label}`")
        st.metric(label="Confianza", value=f"{predicted_score:.2f}%")

with col_map:
    st.subheader("🗺️ Ubicación Geográfica")
    m = folium.Map(location=[coord_y, coord_x], zoom_start=11)
    folium.Marker(
        [coord_y, coord_x],
        popup=f"Punto de Consulta<br>X: {coord_x}, Y: {coord_y}",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    st_folium(m, width=500, height=400)
