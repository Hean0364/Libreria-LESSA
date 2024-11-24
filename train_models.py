# train_models.py

from lessa_lib.training import train_static_model, train_dynamic_model

def main():
    # Entrenar modelo estático
    print("Entrenando modelo estático...")
    train_static_model()

    # Entrenar modelo dinámico
    print("Entrenando modelo dinámico...")
    train_dynamic_model()

if __name__ == "__main__":
    main()
