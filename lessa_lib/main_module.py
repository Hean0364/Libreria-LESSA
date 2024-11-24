# lessa_lib/main_module.py

import cv2
import numpy as np
import pickle
import os
import threading
from tensorflow.keras.models import load_model
from .modules.detector_manos import Detector_Manos
from .modules.frame_saver import FrameSaver

class SignLanguageRecognizer:
    def __init__(self):
        self.is_active = False
        self.current_text = ''
        self.modo = 'reconocimiento_estatico'  # Modo inicial de reconocimiento estático
        self.modelo = None
        self.le = None 
        self.scaler = None
        self.detector = Detector_Manos(max_num_hands=2, detection_confidence=0.5)
        self.saver = FrameSaver()
        self.secuencias_frames = []
        self.secuencias = 30  # Número de frames en cada secuencia
        self.current_label = None
        self.last_class = None
        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()
        self.recognized_signs = []  # Lista para almacenar las señas reconocidas
        self.capturing_sequence = False  # Bandera para capturar secuencia dinámica
        self.sequence_count = 1  # Contador para secuencias dinámicas
        self.frame_count = 1  # Contador para imágenes estáticas
        self.load_models(self.modo)
        
    def load_models(self, modo):
        base_path = os.path.dirname(__file__)
        model_path = ''
        encoder_path = ''
        scaler_path = ''
        if modo == 'reconocimiento_estatico':
            model_path = os.path.join(base_path, 'models', 'modelo_senas_estaticas.h5')
            encoder_path = os.path.join(base_path, 'models', 'label_encoder_estaticas.pkl')
            scaler_path = os.path.join(base_path, 'models', 'scaler_estaticas.pkl')
        elif modo == 'reconocimiento_dinamico':
            model_path = os.path.join(base_path, 'models', 'modelo_senas_dinamicas.h5')
            encoder_path = os.path.join(base_path, 'models', 'label_encoder_dinamicas.pkl')
            scaler_path = os.path.join(base_path, 'models', 'scaler_dinamicas.pkl')
        else:
            self.modelo = None
            self.le = None
            self.scaler = None
            return

        # Cargar el modelo
        if os.path.exists(model_path):
            self.modelo = load_model(model_path)
            print(f"Modelo cargado desde {model_path}")
        else:
            print(f"Modelo no encontrado en {model_path}")
            self.modelo = None

        # Cargar el LabelEncoder
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as file:
                self.le = pickle.load(file)
            print(f"LabelEncoder cargado desde {encoder_path}")
        else:
            print(f"LabelEncoder no encontrado en {encoder_path}")
            self.le = None

        # Cargar el Scaler
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as file:
                self.scaler = pickle.load(file)
            print(f"Scaler cargado desde {scaler_path}")
        else:
            print(f"Scaler no encontrado en {scaler_path}")
            self.scaler = None

    def process_frame(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("No se pudo acceder a la cámara.")
                break
            frame = cv2.flip(frame, 1)
            frame, all_hands = self.detector.encontrar_manos(frame, draw=True)
            key = cv2.waitKey(1) & 0xFF

            if all_hands:
                if self.modo.startswith('reconocimiento'):
                    if self.modelo is None or self.le is None or self.scaler is None:
                        cv2.putText(frame, 'Modelo, Encoder o Scaler no cargado', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        self.reconocer_manos(frame, all_hands)
                elif self.modo.startswith('captura'):
                    if self.capturing_sequence:
                        self.capturar_secuencia_dinamica(frame, all_hands)
                    else:
                        # Modo captura estático: mostrar etiqueta actual
                        cv2.putText(frame, f'Modo captura: {self.current_label}', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                self.last_class = None

            cv2.putText(frame, f'Modo: {self.modo}', (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Mostrar última palabra reconocida en la pantalla
            if self.last_class:
                cv2.putText(frame, f'Reconocido: {self.last_class}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Lenguaje de señas', frame)

            # Manejar atajos de teclado
            if key == ord('q'):
                self.force_stop()
                break
            elif key == ord('m'):
                self.cambiar_modo()
            elif key == ord('c'):
                self.activar_captura()
            elif key == ord('g'):
                if self.modo.startswith('captura'):
                    self.guardar_datos(frame, all_hands)

        self.clean_up()

    def reconocer_manos(self, frame, all_hands):
        if self.modo == 'reconocimiento_estatico':
            for hand_landmarks in all_hands:
                class_name = self.predecir_estatico(hand_landmarks)
        elif self.modo == 'reconocimiento_dinamico':
            class_name = self.predecir_dinamico(frame, all_hands)

    def predecir_estatico(self, hand_landmarks):
        landmarks = np.array(hand_landmarks).flatten().reshape(1, -1)
        landmarks_scaled = self.scaler.transform(landmarks)
        y_pred = self.modelo.predict(landmarks_scaled)
        class_id = np.argmax(y_pred)
        class_name = self.le.inverse_transform([class_id])[0]

        if class_name != self.last_class:
            self.current_text += ' ' + class_name if self.current_text else class_name
            self.last_class = class_name
            self.recognized_signs.append(class_name)  # Agregar a la lista de señas reconocidas

        return class_name

    def predecir_dinamico(self, frame, all_hands):
        # Inicializar listas para las landmarks de cada mano
        hand1_landmarks = np.zeros(63)  # 21 puntos * 3 coordenadas
        hand2_landmarks = np.zeros(63)

        if len(all_hands) >= 1:
            hand1_landmarks = np.array(all_hands[0]).flatten()
        if len(all_hands) >= 2:
            hand2_landmarks = np.array(all_hands[1]).flatten()

        combined_landmarks = np.concatenate([hand1_landmarks, hand2_landmarks])
        self.secuencias_frames.append(combined_landmarks)

        cv2.putText(frame, f'Capturando secuencia: {len(self.secuencias_frames)}/{self.secuencias}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        if len(self.secuencias_frames) == self.secuencias:
            sequence = np.array(self.secuencias_frames)
            sequence_flat = sequence.flatten().reshape(1, -1)
            sequence_scaled = self.scaler.transform(sequence_flat)
            sequence_scaled = sequence_scaled.reshape(1, self.secuencias, -1)
            y_pred = self.modelo.predict(sequence_scaled)
            class_id = np.argmax(y_pred)
            class_name = self.le.inverse_transform([class_id])[0]

            if class_name != self.last_class:
                self.current_text += ' ' + class_name if self.current_text else class_name
                self.last_class = class_name
                self.recognized_signs.append(class_name)  # Agregar a la lista de señas reconocidas

            self.secuencias_frames = []
            return class_name
        return None

    def guardar_datos(self, frame, all_hands):
        if self.current_label is None:
            print("No hay etiqueta establecida. Presione 'c' para iniciar la captura y establecer una etiqueta.")
            return
        if self.modo == 'captura_estatica':
            # Guardar imagen estática
            self.saver.save_image(frame, self.current_label, self.frame_count)
            print(f"Imagen guardada como {self.current_label}_{self.frame_count}.jpg")
            self.frame_count += 1
        elif self.modo == 'captura_dinamica':
            if not self.capturing_sequence:
                # Iniciar captura automática de secuencia
                self.capturing_sequence = True
                self.secuencias_frames = []
                print(f"Iniciando captura de secuencia para {self.current_label}")
            # La captura de secuencia se maneja en el método capturar_secuencia_dinamica
        else:
            print("No está en modo de captura. Presione 'c' para activar el modo de captura.")

    def capturar_secuencia_dinamica(self, frame, all_hands):
        if not self.capturing_sequence:
            return

        # Capturar landmarks de las manos
        hand1_landmarks = np.zeros(63)
        hand2_landmarks = np.zeros(63)

        if len(all_hands) >= 1:
            hand1_landmarks = np.array(all_hands[0]).flatten()
        if len(all_hands) >= 2:
            hand2_landmarks = np.array(all_hands[1]).flatten()

        combined_landmarks = np.concatenate([hand1_landmarks, hand2_landmarks])
        self.secuencias_frames.append(combined_landmarks)

        cv2.putText(frame, f'Capturando secuencia: {len(self.secuencias_frames)}/{self.secuencias}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        if len(self.secuencias_frames) == self.secuencias:
            # Guardar secuencia cuando se completa
            sequence = np.array(self.secuencias_frames)
            self.saver.save_sequence(sequence, self.current_label, self.sequence_count)
            print(f"Secuencia guardada como {self.current_label}_{self.sequence_count}")
            self.secuencias_frames = []
            self.sequence_count += 1
            self.capturing_sequence = False  # Detener la captura automática

    def cambiar_modo(self):
        if self.modo == 'reconocimiento_estatico':
            self.modo = 'reconocimiento_dinamico'
        elif self.modo == 'reconocimiento_dinamico':
            self.modo = 'reconocimiento_estatico'
        elif self.modo == 'captura_estatica':
            self.modo = 'reconocimiento_estatico'
        elif self.modo == 'captura_dinamica':
            self.modo = 'reconocimiento_dinamico'
        self.load_models(self.modo)
        self.secuencias_frames = []
        self.capturing_sequence = False  # Asegurarse de detener cualquier captura en curso
        print(f"Cambiado a modo: {self.modo}")

    def activar_captura(self):
        if self.modo == 'reconocimiento_estatico' or self.modo == 'reconocimiento_dinamico':
            if self.modo == 'reconocimiento_estatico':
                self.modo = 'captura_estatica'
            elif self.modo == 'reconocimiento_dinamico':
                self.modo = 'captura_dinamica'
            # Solicitar etiqueta una vez por letra
            if self.current_label is None:
                self.current_label = input("Ingrese la etiqueta para la captura: ").strip()
                if self.modo == 'captura_estatica':
                    self.frame_count = 1
                elif self.modo == 'captura_dinamica':
                    self.sequence_count = 1
            print(f"Modo de captura activado: {self.modo} para la etiqueta: {self.current_label}")
        elif self.modo.startswith('captura'):
            # Si ya estamos en modo captura, volver al modo de reconocimiento
            if self.modo == 'captura_estatica':
                self.modo = 'reconocimiento_estatico'
            elif self.modo == 'captura_dinamica':
                self.modo = 'reconocimiento_dinamico'
            self.current_label = None
            self.secuencias_frames = []
            self.capturing_sequence = False  # Asegurarse de detener cualquier captura en curso
            print(f"Captura detenida. Modo actual: {self.modo}")

    def start(self):
        if not self.is_active:
            self.is_active = True
            # Reiniciar información acumulada al iniciar
            self.current_text = ''
            self.recognized_signs = []
            self.last_class = None
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.stop_event.clear()
            self.thread = threading.Thread(target=self.process_frame)
            self.thread.start()
            print("Reconocimiento iniciado.")

    def force_stop(self):
        if self.is_active:
            self.is_active = False
            self.stop_event.set()
            self.thread.join()
            self.clean_up()
            print("Reconocimiento detenido.")

    def clean_up(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def preload(self):
        # Cargar los modelos necesarios
        self.load_models(self.modo)
        print("Modelos precargados.")

    def get_current_data(self):
        # Guardar las señas reconocidas en 'senas_reconocidas.txt' en la carpeta lessa_lib
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        filename = os.path.join(base_path, 'senas_reconocidas.txt')
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(' '.join(self.recognized_signs))
            print(f"Señas reconocidas guardadas en el archivo: {filename}")
        except Exception as e:
            print(f"Error al guardar las señas reconocidas: {e}")
        return {
            'is_active': self.is_active,
            'current_text': self.current_text.strip(),
            'last_class': self.last_class,
            'recognized_signs': self.recognized_signs
        }


# Instancia global para usar en funciones externas
recognizer = SignLanguageRecognizer()

# Funciones expuestas
def start():
    recognizer.start()

def force_stop():
    recognizer.force_stop()

def preload():
    recognizer.preload()

def get_current_data():
    return recognizer.get_current_data()
