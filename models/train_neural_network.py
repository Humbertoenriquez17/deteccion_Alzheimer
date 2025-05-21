import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

df_pos_train = pd.read_csv('data/Dataset_positivo_Train.csv', header=None)
df_pos_test = pd.read_csv('data/Dataset_positivo_Test.csv', header=None)
df_neg_train = pd.read_csv('data/Dataset_negativo_Train.csv', header=None)
df_neg_test = pd.read_csv('data/Dataset_negativo_Test.csv', header=None)

k = 8 

df_pos_train['label'] = 1
df_pos_test['label'] = 1
df_neg_train['label'] = 0
df_neg_test['label'] = 0

df_all = pd.concat([df_pos_train, df_pos_test, df_neg_train, df_neg_test], ignore_index=True)

X = df_all.iloc[:, :-1].values  
y = df_all['label'].values      

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

selector = SelectKBest(score_func=f_classif, k=k)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

selected_columns = selector.get_support(indices=True)
print(f"Columnas seleccionadas por SelectKBest (índices): {selected_columns}")

model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)

y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
f1 = f1_score(y_test, y_pred, zero_division=0)
print("F1-score:", f1)
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, zero_division=0))

if f1 >= 0.92:
    model.save('models/modelo_f1_92.h5')
    print("Modelo guardado como modelo_f1_92.h5")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()
