# main.py

import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ======================================================
# Configurações
# ======================================================

SEED = 42

warnings.filterwarnings("ignore")

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ======================================================
# Funções
# ======================================================

def carregar_dados(caminho):
    """
    Carrega um arquivo CSV.
    """
    return pd.read_csv(caminho)


def preparar_dados(df):
    """
    Exemplo de preparação dos dados.
    Altere conforme seu projeto.
    """
    print("Dimensão do dataset:", df.shape)
    print(df.head())

    return df


def criar_modelo(entrada):
    """
    Modelo básico usando Keras.
    """
    modelo = keras.Sequential([
        keras.layers.Input(shape=(entrada,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1)
    ])

    modelo.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return modelo


# ======================================================
# Programa Principal
# ======================================================

def main():

    print("=" * 50)
    print("Projeto de Inteligência Artificial")
    print("TensorFlow:", tf.__version__)
    print("Keras:", keras.__version__)
    print("=" * 50)

    # Exemplo:
    # df = carregar_dados("dados.csv")
    # df = preparar_dados(df)

    print("Projeto iniciado com sucesso.")


if __name__ == "__main__":
    main()
