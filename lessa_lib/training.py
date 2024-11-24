

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pickle
import os

def train_static_model():
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, 'models')

    X_train = np.load(os.path.join(data_path, 'X_train_estaticas.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test_estaticas.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train_estaticas.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test_estaticas.npy'))

    with open(os.path.join(data_path, 'label_encoder_estaticas.pkl'), 'rb') as file:
        le = pickle.load(file)

    num_classes = len(le.classes_)

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=16,
              validation_data=(X_test, y_test), callbacks=[early_stopping])

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Precisión en el conjunto de prueba: {accuracy*100:.2f}%')

    model.save(os.path.join(data_path, 'modelo_senas_estaticas.h5'))
    print("Modelo estático entrenado y guardado exitosamente.")

def train_dynamic_model():
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, 'models')

    X_train = np.load(os.path.join(data_path, 'X_train_dinamicas.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test_dinamicas.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train_dinamicas.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test_dinamicas.npy'))

    with open(os.path.join(data_path, 'label_encoder_dinamicas.pkl'), 'rb') as file:
        le = pickle.load(file)

    num_classes = len(le.classes_)

    model = Sequential([
        LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=32,
              validation_data=(X_test, y_test), callbacks=[early_stopping])

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Precisión en el conjunto de prueba: {accuracy*100:.2f}%')

    model.save(os.path.join(data_path, 'modelo_senas_dinamicas.h5'))
    print("Modelo dinámico entrenado y guardado exitosamente.")
