import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Cargar dataset
df = pd.read_csv("datos.csv")

# Eliminar Duplicado
df = df.drop_duplicates()

# Rellenar valores nulos
df = df.fillna(df.mean(numeric_only=True))

# Codificar variables categorias
df = pd.get_dummies(df)

# Normalizacion
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Guardar dataset limpio
df.to_csv("datos_limpio.csv", index=False)

print("Preprocesamiento completado")
