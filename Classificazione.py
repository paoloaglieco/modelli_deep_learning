import pandas as pd  # Importiamo la libreria pandas per la manipolazione dei dati
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Importiamo gli strumenti per la standardizzazione e la codifica
from sklearn.model_selection import train_test_split  # Importiamo la funzione per dividere i dati in training e test set
from collections import Counter  # Importiamo la classe Counter per contare gli elementi
from sklearn.compose import ColumnTransformer  # Importiamo lo strumento per applicare trasformazioni a colonne specifiche
from tensorflow.keras.models import Sequential  # Importiamo la classe per creare modelli sequenziali
from tensorflow.keras.layers import Dense, InputLayer  # Importiamo i layer densi e di input
from sklearn.metrics import classification_report  # Importiamo la funzione per generare il classification report
from tensorflow.keras.utils import to_categorical  # Importiamo la funzione per convertire il target in formato one-hot encoding
import numpy as np  # Importiamo la libreria numpy per operazioni numeriche

# Carica il dataset e visualizza le informazioni di base
data = pd.read_csv('heart_failure.csv')
print(data.info())

# Conta il numero di eventi di morte (0: no morte, 1: morte)
print(Counter(data['death_event']))

# Separa le features dal target
y = data["death_event"]
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]

# Crea variabili dummy per le features categoriche
x = pd.get_dummies(x)

# Suddivide i dati in training set e test set
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# Definisce un trasformatore per standardizzare le features numeriche
ct = ColumnTransformer([("numeric", StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])

# Applica la standardizzazione ai dati di training e test
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)  # Nota: qui si usa transform invece di fit_transform per non riscalare i dati di test

# Codifica il target in formato numerico
le = LabelEncoder()
Y_train = le.fit_transform(Y_train.astype(str))
Y_test = le.transform(Y_test.astype(str))  # Nota: qui si usa transform per mantenere la stessa codifica del training set

# Converte il target in formato one-hot encoding
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Crea il modello sequenziale
model = Sequential()
model.add(InputLayer(input_shape=(X_train.shape[1],)))  # Input layer con la dimensione delle features
model.add(Dense(12, activation='relu'))  # Primo layer nascosto con 12 neuroni e attivazione ReLU
model.add(Dense(2, activation='softmax'))  # Layer di output con 2 neuroni (per le due classi) e attivazione softmax

# Compila il modello
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Addestra il modello
model.fit(X_train, Y_train, epochs = 50, batch_size = 2, verbose=1)

# Valuta il modello sul set di test
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print("Loss", loss, "Accuracy:", acc)

# Effettua predizioni sul set di test
y_estimate = model.predict(X_test, verbose=0)

# Converte le predizioni in formato numerico
y_estimate = np.argmax(y_estimate, axis=1)
Y_test = np.argmax(Y_test, axis=1)

# Genera il classification report
print(classification_report(Y_test, y_estimate))