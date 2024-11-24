

from lessa_lib.main_module import start, force_stop, preload, activar_captura, cambiar_modo, is_active
import time

def main():
    # Precargar modelos
    preload()

    # Iniciar el reconocimiento (abrirá la ventana de la cámara)
    start()

    print("Presiona 'c' para activar el modo de captura.")
    print("Una vez en modo de captura, sigue las instrucciones en la consola.")

    try:
        while is_active():
            # Mantener el script en ejecución mientras la captura está activa
            time.sleep(1)
    except KeyboardInterrupt:
        print("Deteniendo captura de datos.")
        force_stop()

if __name__ == "__main__":
    main()
