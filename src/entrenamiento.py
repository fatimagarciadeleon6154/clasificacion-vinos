import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Cargar datos limpios
df = pd.read_csv("WineQT_limpio.csv")
X = df.drop(columns=["quality"])
y = df["quality"]

# 2. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardar datos de prueba para usarlos en prueba.py
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# 3. Definir y entrenar los modelos
models = {
    'Support Vector Machine (SVM)': SVC(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(),
    'Naive Bayes (GaussianNB)': GaussianNB(),
    'Neural Network (MLP)': MLPClassifier(max_iter=1000, random_state=42)
}

results = []
mejor_accuracy = 0
mejor_modelo = None
mejor_nombre = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
    
    results.append({
        'Modelo': name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4)
    })
    
    # Verificar si es el mejor modelo hasta ahora
    if accuracy > mejor_accuracy:
        mejor_accuracy = accuracy
        mejor_modelo = model
        mejor_nombre = name

# 4. Mostrar tabla comparativa en consola
df_comparativo = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
print("=== COMPARACIÓN DE MODELOS ===")
print(df_comparativo.to_string(index=False))

# 5. Guardar el mejor modelo
joblib.dump(mejor_modelo, "mejor_modelo.pkl")
print(f"\nMejor modelo ({mejor_nombre} con Accuracy: {mejor_accuracy:.4f}) guardado como 'mejor_modelo.pkl'.")