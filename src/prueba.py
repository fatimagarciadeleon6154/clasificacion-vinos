import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support

# 1. Cargar el modelo guardado y los datos de prueba
modelo = joblib.load("mejor_modelo.pkl")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")["quality"]

# 2. Evaluar el modelo en datos de prueba
y_pred = modelo.predict(X_test)

# 3. Reportar métricas finales
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

print("=== MÉTRICAS FINALES DEL MEJOR MODELO ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\n=== REPORTE DE CLASIFICACIÓN DETALLADO ===")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n=== MATRIZ DE CONFUSIÓN (TEXTO) ===")
print(confusion_matrix(y_test, y_pred))

# 4. Mostrar matriz de confusión gráfica
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title("Matriz de Confusión - Mejor Modelo")
plt.tight_layout()
plt.show()