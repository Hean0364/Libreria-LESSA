# test_lessa_lib.py

from lessa_lib import start, force_stop, preload, get_current_data, save_recognized_signs
import time

# Pre-cargar modelos y recursos necesarios
preload()

# Iniciar el reconocimiento
start()

print("Iniciando reconocimiento. Presiona 'm' para cambiar de modo, 'c' para capturar datos, 'q' para salir.")

try:
    while True:
        # Esperar un pequeño intervalo antes de verificar si el reconocimiento sigue activo
        time.sleep(1)
        data = get_current_data()
        if not data['is_active']:
            break
except KeyboardInterrupt:
    # Detener el reconocimiento al presionar Ctrl+C
    force_stop()

# Al finalizar, obtener los datos y mostrar la lista de señas reconocidas
data = get_current_data()
print("El reconocimiento ha finalizado.")
print("Señas y letras reconocidas durante la sesión:")
print(data['recognized_signs'])

# Guardar las señas reconocidas en un archivo
filename = 'senas_reconocidas.txt'
save_recognized_signs(filename)
