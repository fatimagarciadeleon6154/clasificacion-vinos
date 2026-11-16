import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("WineQT.csv")

# 2. Mostrar estadísticas descriptivas
print("=== VALORES NULOS ===")
print(df.isnull().sum())

print("\n=== INFORMACIÓN GENERAL ===")
print(df.info())

print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
print(df.describe())

print("\n=== MUESTRA DE DATOS ===")
print(df.sample(10))

# 3. Limpieza de datos (eliminar columna Id si existe y filas nulas/duplicadas)
if "Id" in df.columns:
    df = df.drop(columns=["Id"])
df = df.dropna().drop_duplicates()

# Guardar datos limpios para el entrenamiento
df.to_csv("WineQT_limpio.csv", index=False)
print("\nDataset limpio guardado como 'WineQT_limpio.csv'.")

# 4. Visualizar distribuciones
# A) Histogramas
df.hist(bins=15, figsize=(15, 10))
plt.suptitle("Histogramas de las variables del vino")
plt.tight_layout()
plt.show()

# B) Matriz de correlación
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.tight_layout()
plt.show()

# C) Boxplots
features_to_plot = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(features_to_plot):
    sns.boxplot(x='quality', y=col, hue='quality', data=df, ax=axes[i], palette='Set2', legend=False)
    axes[i].set_title(f'Distribución de {col} por Calidad de Vino')
    axes[i].set_xlabel('Calidad (Clase)')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()
