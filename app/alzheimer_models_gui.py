import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, filedialog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import tensorflow as tf
from tensorflow import keras

df_pos_train = pd.read_csv('C:/Users/humbe/Desktop/Adqui/Proyecto/Dataset_positivo_Train.csv', header=None)
df_pos_test = pd.read_csv('C:/Users/humbe/Desktop/Adqui/Proyecto/Dataset_positivo_Test.csv', header=None)
df_neg_train = pd.read_csv('C:/Users/humbe/Desktop/Adqui/Proyecto/Dataset_negativo_Train.csv', header=None)
df_neg_test = pd.read_csv('C:/Users/humbe/Desktop/Adqui/Proyecto/Dataset_negativo_Test.csv', header=None)

df_pos_train['label'] = 1
df_pos_test['label'] = 1
df_neg_train['label'] = 0
df_neg_test['label'] = 0

df_all = pd.concat([df_pos_train, df_pos_test, df_neg_train, df_neg_test], ignore_index=True)
X_all = df_all.iloc[:, :-1].values
y_all = df_all['label'].values

scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

k = 8
selector = SelectKBest(score_func=f_classif, k=k)
selector.fit(X_all_scaled, y_all)
selected_columns = selector.get_support(indices=True)
print(f"Columnas seleccionadas por SelectKBest (índices): {selected_columns}")

model = keras.models.load_model(r'c:\Users\humbe\modelo_f1_91.h5')

Test_bien = pd.read_csv('C:/Users/humbe/Desktop/Adqui/Proyecto/Dataset_positivo_Test.csv', header=None).values
Test_mal = pd.read_csv('C:/Users/humbe/Desktop/Adqui/Proyecto/Dataset_negativo_Test.csv', header=None).values
Train_bien = pd.read_csv('C:/Users/humbe/Desktop/Adqui/Proyecto/Dataset_positivo_Train.csv', header=None).values
Train_mal = pd.read_csv('C:/Users/humbe/Desktop/Adqui/Proyecto/Dataset_negativo_Train.csv', header=None).values

X_train = np.vstack((Train_mal, Train_bien))
y_train = np.concatenate((np.zeros(len(Train_mal)), np.ones(len(Train_bien))))
X_test = np.vstack((Test_mal, Test_bien))
y_test = np.concatenate((np.zeros(len(Test_mal)), np.ones(len(Test_bien))))

ymax, ymin = 1, 0.1
valMin = np.min(np.vstack((X_train, X_test)), axis=0)
valMax = np.max(np.vstack((X_train, X_test)), axis=0)

def normalize(X):
    return ((ymax - ymin) * (X - valMin)) / (valMax - valMin + 1e-8) + ymin

X_train_norm = normalize(X_train)
X_test_norm = normalize(X_test)

root = tk.Tk()
root.title("Sistema de Clasificación Médica")
root.geometry("600x650")
root.configure(bg="#ffffff")
root.resizable(False, False)

FONT_TITLE = ("Arial", 16, "bold")
FONT_LABEL = ("Arial", 12)
FONT_ENTRY = ("Arial", 12)
FONT_BUTTON = ("Arial", 12, "bold")

logo_frame = tk.Frame(root, bg="#ffffff")
logo_frame.pack(pady=10)
tk.Label(logo_frame, text="🩺 Deteccion de Alzheimer con Machine Learning", font=FONT_TITLE, bg="#ffffff", fg="#0078D7").pack()

mode_var = tk.StringVar(value="fila")
tk.Label(root, text="Seleccione el modo de entrada:", font=FONT_LABEL, bg="#ffffff", fg="#333333").pack(pady=(20, 5))
tk.Radiobutton(root, text="Usar fila del dataset", variable=mode_var, value="fila", bg="#ffffff", font=FONT_LABEL).pack()

tk.Label(root, text=f"Número de fila (1 a 152 - No Presencia, 153 a 304 - Presencia):", font=FONT_LABEL, bg="#ffffff", fg="#333333").pack(pady=(20, 5))
entry = tk.Entry(root, font=FONT_ENTRY, justify="center", width=10, bd=2, relief="groove")
entry.pack(pady=5)

tk.Radiobutton(root, text="Cargar archivo de paciente (.csv)", variable=mode_var, value="archivo", bg="#ffffff", font=FONT_LABEL).pack()

style = {
    "font": FONT_BUTTON,
    "width": 30,
    "height": 2,
    "bg": "#0078D7",
    "fg": "white",
    "activebackground": "#005A9E",
    "activeforeground": "white",
    "bd": 0,
    "cursor": "hand2"
}

archivo_paciente = [None]  

def cargar_archivo():
    file_path = filedialog.askopenfilename(
        title="Seleccionar archivo de paciente",
        filetypes=[("CSV files", "*.csv"), ("Todos los archivos", "*.*")]
    )
    if file_path:
        try:
            df = pd.read_csv(file_path, header=None)
            if df.shape != (1, 32):
                messagebox.showerror("Error", "El archivo debe tener exactamente 1 fila y 32 columnas.")
                archivo_paciente[0] = None
            else:
                archivo_paciente[0] = df.values.flatten()
                messagebox.showinfo("Archivo cargado", "Archivo de paciente cargado correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el archivo: {e}")
            archivo_paciente[0] = None

tk.Button(root, text="Cargar archivo de paciente", command=cargar_archivo, **{**style, "width": 30, "height": 1}).pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#ffffff", fg="#0078D7")
result_label.pack(pady=(10, 0))

def mostrar_resultado(pred):
    if pred == 0:
        result_label.config(text="No presencia de Alzheimer", fg="#228B22")
    elif pred == 1:
        result_label.config(text="Posible presencia, haga más estudios", fg="#D2691E")
    else:
        result_label.config(text="Resultado desconocido", fg="#FF0000")

def get_input_vector():
    """Obtiene el vector de entrada según el modo seleccionado."""
    if mode_var.get() == "archivo":
        if archivo_paciente[0] is None:
            messagebox.showerror("Error", "No se ha cargado un archivo de paciente válido.")
            return None
        return archivo_paciente[0]
    else:
        try:
            idx = int(entry.get())
            if idx < 1 or idx > X_test.shape[0]:
                raise ValueError
            return X_test[idx - 1]
        except:
            messagebox.showerror("Error", "Número de fila inválido.")
            return None

def predict_nn():
    x = get_input_vector()
    if x is None:
        return
    x_scaled = scaler.transform([x])
    x_selected = selector.transform(x_scaled)
    y_pred_prob = model.predict(x_selected)[0][0]
    pred = int(y_pred_prob > 0.5)
    mostrar_resultado(pred)

def gaussian_naive_bayes_predict(X_train, y_train, x_sample):
    classes = np.unique(y_train)
    log_probs = []
    for c in classes:
        X_c = X_train[y_train == c]
        mean = np.mean(X_c, axis=0)
        var = np.var(X_c, axis=0, ddof=0) + 1e-8
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var))
        log_likelihood -= 0.5 * np.sum(((x_sample - mean) ** 2) / var)
        log_probs.append(log_likelihood)
    return int(np.argmax(log_probs))

def predict_bayes():
    x = get_input_vector()
    if x is None:
        return
    x_norm = normalize(x.reshape(1, -1))[0]
    pred = gaussian_naive_bayes_predict(X_train_norm, y_train, x_norm)
    mostrar_resultado(pred)

def train_svm():
    svm_model = SVC(kernel='linear', C=0.1)
    svm_model.fit(X_train_norm, y_train)
    x = get_input_vector()
    if x is None:
        return
    x_norm = normalize(x.reshape(1, -1))[0]
    pred = svm_model.predict([x_norm])
    mostrar_resultado(int(pred[0]))

tk.Button(root, text="Clasificar con Red Neuronal", command=lambda: predict_nn(), **style).pack(pady=10)
tk.Label(root, text="F1 Score: 92%", font=("Arial", 10, "italic"), bg="#ffffff", fg="#0078D7").pack()

tk.Button(root, text="Clasificar con Bayesiano", command=lambda: predict_bayes(), **style).pack(pady=5)
tk.Label(root, text="F1 Score: 85%", font=("Arial", 10, "italic"), bg="#ffffff", fg="#0078D7").pack()

tk.Button(root, text="Clasificar con SVM", command=lambda: train_svm(), **style).pack(pady=5)
tk.Label(root, text="F1 Score: 86%", font=("Arial", 10, "italic"), bg="#ffffff", fg="#0078D7").pack()

tk.Label(root, text="© 2025 Sistema Médico de Deteccion de Alzheimer", font=("Arial", 10), bg="#ffffff", fg="#888888").pack(pady=(20, 0))

root.eval('tk::PlaceWindow . center')
root.mainloop()