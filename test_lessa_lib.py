# test_lessa_lib.py

import time
import os
import threading
from lessa_lib.main_module import start, force_stop, get_current_data

def main():
    try:
        # Iniciar el reconocimiento
        start()
        print("Reconocimiento iniciado. Realiza algunas señas frente a la cámara.")
        print("Escribe 'q' y presiona Enter en la consola para detener el reconocimiento.")

        # Evento para controlar la parada del reconocimiento
        stop_event = threading.Event()

        # Hilo para esperar la entrada del usuario sin bloquear el reconocimiento
        def wait_for_user_input():
            while not stop_event.is_set():
                user_input = input()
                if user_input.strip().lower() == 'q':
                    stop_event.set()

        input_thread = threading.Thread(target=wait_for_user_input)
        input_thread.daemon = True  # Permite que el hilo se cierre cuando el programa termine
        input_thread.start()

        # Bucle principal que se ejecuta hasta que el usuario decide detener el reconocimiento
        while not stop_event.is_set():
            time.sleep(0.1)
            # Opcionalmente, puedes obtener y mostrar datos actuales
            # data = get_current_data()
            # print(f"Texto acumulado: {data['current_text']}")
            # time.sleep(1)  # Mostrar cada 1 segundo

        # Obtener datos actuales al finalizar
        data = get_current_data()
        print("\nDatos obtenidos:")
        print(f"Reconocimiento activo: {data['is_active']}")
        print(f"Texto acumulado: {data['current_text']}")
        print(f"Última clase reconocida: {data['last_class']}")
        print(f"Señas reconocidas: {data['recognized_signs']}")

        # Mostrar contenido del archivo 'senas_reconocidas.txt'
        filename = os.path.join('lessa_lib', 'senas_reconocidas.txt')
        if not os.path.exists(filename):
            filename = 'senas_reconocidas.txt'  # Intentar en el directorio actual
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                contenido = f.read()
            print(f"\nContenido de '{filename}': {contenido}")
        except Exception as e:
            print(f"No se pudo leer el archivo '{filename}': {e}")

    finally:
        # Detener el reconocimiento
        force_stop()
        print("Reconocimiento detenido.")

if __name__ == "__main__":
    main()
