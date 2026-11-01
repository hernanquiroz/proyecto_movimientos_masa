import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import st_folium

# Configuración de la página
st.set_page_config(page_title="Predictor de Movimientos en Masa", page_icon="🏔️", layout="wide")

st.title("🏔️ Sistema de Pronóstico de Movimientos en Masa (Valle de Aburrá / Medellín)")

# Función robusta de carga con respaldo local
@st.cache_data(ttl=86400)
def load_data():
    # URL oficial con el parámetro de límite requerido por Socrata (datos.gov.co)
    live_url = "https://www.datos.gov.co/resource/crkd-y9d2.csv?$limit=50000"
    
    try:
        # Intentar cargar desde la API en vivo
        df = pd.read_csv(live_url)
        st.sidebar.success("✅ Conectado a datos.gov.co en vivo")
        return df
    except Exception as e:
        # Respaldo automático con el archivo local si falla la red o el servicio
        st.sidebar.warning(f"⚠️ Usando archivo local de respaldo (Error en API: {e})")
        return pd.read_excel("Inventario_movimientos_masa_limpio.xlsx")

try:
    df = load_data()
except Exception as e:
    st.error(f"Error crítico al cargar los datos: {e}")
    st.stop()

# Verificar y preparar las columnas necesarias
features = ['x', 'y', 'F35DOV_TIP', 'Subtipo_Movimiento']

# Validar columnas (si los nombres difieren en la API, aseguramos valores por defecto o limpieza)
for col in features + ['Tipo_Movimiento']:
    if col not in df.columns:
        # Si alguna columna no coincide exactamente, la creamos o rellenamos para evitar interrupciones
        df[col] = 0 if col != 'Tipo_Movimiento' else 'Deslizamiento'

X = df[features].fillna(0)
le = LabelEncoder()
y = le.fit_transform(df['Tipo_Movimiento'].astype(str))

# Entrenar el modelo de Random Forest
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=123)
    model.fit(X, y)
    return model

model = train_model(X, y)

# Interfaz en dos columnas
col_form, col_map = st.columns([1, 1])

with col_form:
    st.subheader("Parámetros de Entrada (Área Metropolitana)")
    with st.form("prediction_form"):
        # Coordenadas predeterminadas centradas en Medellín
        coord_x = st.number_input("Coordenada X (Longitud)", value=-75.581200, format="%.6f")
        coord_y = st.number_input("Coordenada Y (Latitud)", value=6.244200, format="%.6f")
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
        
        # Realizar predicción y probabilidad
        pred_idx = model.predict(input_data)[0]
        pred_proba = model.predict_proba(input_data)[0]
        
        predicted_label = le.inverse_transform([pred_idx])[0]
        predicted_score = pred_proba[pred_idx] * 100
        
        st.success(f"**Tipo Predicho:** `{predicted_label}`")
        st.metric(label="Confianza del Modelo", value=f"{predicted_score:.2f}%")

with col_map:
    st.subheader("🗺️ Ubicación Geográfica en Medellín")
    m = folium.Map(location=[coord_y, coord_x], zoom_start=12)
    folium.Marker(
        [coord_y, coord_x],
        popup=f"Punto Evaluado<br>X: {coord_x}, Y: {coord_y}",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    st_folium(m, width=500, height=400)

