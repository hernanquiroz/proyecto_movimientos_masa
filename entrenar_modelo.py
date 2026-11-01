import pandas as pd
from pycaret.classification import *

# Cargar el archivo Excel limpio
file_path = "Inventario_movimientos_masa_limpio.xlsx"
df = pd.read_excel(file_path)

# Seleccionar variables relevantes del inventario
model_data = df[['x', 'y', 'F35DOV_TIP', 'Subtipo_Movimiento', 'Tipo_Movimiento']].copy()

# Configurar el entorno de PyCaret
clf_setup = setup(
    data=model_data, 
    target='Tipo_Movimiento', 
    session_id=123,
    normalize=True,
    html=False,
    verbose=False
)

# Comparar y seleccionar el mejor modelo predictivo
best_model = compare_models(n_select=1)

# Finalizar y guardar el modelo entrenado
final_model = finalize_model(best_model)
save_model(final_model, 'modelo_movimientos_masa_pipeline')
print("¡Modelo entrenado y guardado exitosamente!")
