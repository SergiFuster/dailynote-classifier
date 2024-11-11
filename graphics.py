import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Cargar los datos
with open("dataset.json", mode='r', encoding="utf-8") as f:
    data = json.load(f)

# Contar las etiquetas en cada conjunto
train_labels = data['train']['label_text']
test_labels = data['test']['label_text']

train_counts = Counter(train_labels)
test_counts = Counter(test_labels)

# Ordenar las etiquetas para que coincidan en ambos conjuntos
labels = sorted(set(train_labels + test_labels))
train_values = [train_counts.get(label, 0) for label in labels]
test_values = [test_counts.get(label, 0) for label in labels]

# Crear el gráfico
plt.figure(figsize=(10, 6))
sns.barplot(x=labels, y=train_values, color="blue", alpha=0.6, label="Train")
sns.barplot(x=labels, y=test_values, color="red", alpha=0.6, label="Test")

# Añadir etiquetas y leyenda
plt.xlabel("Etiquetas")
plt.ylabel("Número de apariciones")
plt.title("Número de apariciones de cada etiqueta en los conjuntos de entrenamiento y prueba")
plt.legend()
plt.show()
