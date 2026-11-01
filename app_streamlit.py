
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import st_folium

# Configuración de la página
st.set_page_config(page_title="Predictor de Movimientos en Masa - En Vivo", page_icon="🏔️", layout="wide")

st.title("🏔️ Sistema de Pronóstico de Movimientos en Masa (Conectado a Datos Oficiales en Vivo)")

# URL directa del dataset oficial en datos.gov.co (en formato CSV)
DATA_URL = "https://www.datos.gov.co/resource/crkd-y9d2.csv"

# Cargar los datos directamente desde la API oficial de forma optimizada
@st.cache_data(ttl=86400) # Se actualiza automáticamente cada 24 horas
def load_live_data():
    df = pd.read_csv(DATA_URL)
    return df

try:
    df = load_live_data()
    st.sidebar.success(f"✅ Conectado a datos.gov.co (Registros cargados: {len(df)})")
except Exception as e:
    st.error(f"Error al conectar con el portal oficial: {e}")
    st.stop()

# Verificar las columnas disponibles en el dataset oficial para el modelado
# Nota: Ajusta los nombres de las columnas ('x', 'y', etc.) si difieren en el dataset remoto
features = ['x', 'y', 'F35DOV_TIP', 'Subtipo_Movimiento']

# Validar que existan las columnas necesarias
missing_cols = [col for col in features + ['Tipo_Movimiento'] if col not in df.columns]
if missing_cols:
    st.warning(f"⚠️ Las siguientes columnas requeridas no están en el dataset oficial: {missing_cols}")
    st.stop()

X = df[features].fillna(0)
le = LabelEncoder()
y = le.fit_transform(df['Tipo_Movimiento'].fillna('Desconocido'))

# Entrenar el modelo de Random Forest en tiempo real con los datos oficiales
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
        
        submit_button = st.form_submit_button(label="Ejecutar Pronóstico en Vivo")

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
    st.subheader("🗺️ Ubicación Geográfica")
    m = folium.Map(location=[coord_y, coord_x], zoom_start=12)
    folium.Marker(
        [coord_y, coord_x],
        popup=f"Punto Evaluado<br>X: {coord_x}, Y: {coord_y}",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    st_folium(m, width=500, height=400)
