# lessa_lib/preprocessing.py

import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp

def preprocess_static_data(data_dir='data/estaticas', output_dir='models'):
    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, data_dir)
    output_dir = os.path.join(base_path, output_dir)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.2  
    )

    X = []  # Datos de entrada
    y = []  # Etiquetas
    labels = os.listdir(data_dir)
    print(f"Etiquetas encontradas: {labels}")
    total_images = 0
    total_processed = 0
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        print(f"Procesando etiqueta: {label}")
        image_count = 0
        processed_count = 0
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Imagen no encontrada o no se pudo leer: {img_path}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                X.append(landmarks)
                y.append(label)
                processed_count += 1
            else:
                print(f"No se detectó mano en la imagen: {img_path}. Eliminando...")
                try:
                    os.remove(img_path)
                    print(f"Imagen eliminada: {img_path}")
                except Exception as e:
                    print(f"Error al eliminar {img_path}: {e}")
            image_count += 1
        print(f"Total de imágenes en {label}: {image_count}")
        print(f"Imágenes procesadas exitosamente en {label}: {processed_count}")
        total_images += image_count
        total_processed += processed_count
    print(f"Total de imágenes: {total_images}")
    print(f"Total de imágenes procesadas: {total_processed}")

    X = np.array(X)
    y = np.array(y)

    # Preprocesamiento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.1, stratify=y_encoded, random_state=42)

    # Guardar los conjuntos de datos
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, 'X_train_estaticas.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test_estaticas.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train_estaticas.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test_estaticas.npy'), y_test)

    # Guardar el codificador de etiquetas y el scaler
    with open(os.path.join(output_dir, 'label_encoder_estaticas.pkl'), 'wb') as file:
        pickle.dump(le, file)

    with open(os.path.join(output_dir, 'scaler_estaticas.pkl'), 'wb') as file:
        pickle.dump(scaler, file)

    print(f"Datos preprocesados guardados en la carpeta '{output_dir}'.")

def preprocess_dynamic_data(data_dir='data/dinamicas', output_dir='models'):
    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, data_dir)
    output_dir = os.path.join(base_path, output_dir)

    X = []
    y = []
    labels = os.listdir(data_dir)
    print(f"Etiquetas encontradas para dinámicas: {labels}")

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        print(f"Cargando secuencias para la etiqueta: {label}")

        for file_name in os.listdir(label_dir):
            if file_name.endswith('.npy'):
                file_path = os.path.join(label_dir, file_name)
                try:
                    sequence = np.load(file_path)
                    X.append(sequence)
                    y.append(label)
                except Exception as e:
                    print(f"Error al cargar {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"Total de secuencias cargadas: {X.shape[0]}")

    if X.shape[0] == 0:
        print("No se encontraron secuencias dinámicas para procesar.")
        return

    # Aplanar y normalizar
    num_samples, timesteps, features = X.shape
    X_flat = X.reshape((num_samples, timesteps * features))

    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)

    X_scaled = X_flat_scaled.reshape((num_samples, timesteps, features))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    # Guardar los conjuntos de datos
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, 'X_train_dinamicas.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test_dinamicas.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train_dinamicas.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test_dinamicas.npy'), y_test)

    # Guardar el codificador de etiquetas y el scaler
    with open(os.path.join(output_dir, 'label_encoder_dinamicas.pkl'), 'wb') as file:
        pickle.dump(le, file)

    with open(os.path.join(output_dir, 'scaler_dinamicas.pkl'), 'wb') as file:
        pickle.dump(scaler, file)

    print(f"Datos preprocesados guardados en la carpeta '{output_dir}'.")
