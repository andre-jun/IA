import os
import sys
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# DATASET_DIR = "Dataset"
SR = 22050
DURATION = 6
N_MELS = 128

def carregar_audio(caminho):
    y, _ = librosa.load(caminho, sr=SR, duration=DURATION, mono=True)
    if len(y) < SR * DURATION:
        y = np.pad(y, (0, SR * DURATION - len(y)))
    return y

def extrair_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def processar_audios(name, df):
    X, y = [], []

    for _, row in df.iterrows():
        caminho = row['filename']
        rotulo = 1 if str(row['is_cry']).strip().lower() == 'yes' else 0
        try:
            y_audio = carregar_audio(caminho)
            mel = extrair_mel(y_audio)
            X.append(mel)
            y.append(rotulo)
        except Exception as e:
            print(f"Erro ao processar {caminho}: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"Áudios processados: {len(X)}. Salvando...")
    arquivo = "preProcesso" + name + ".npz"
    np.savez_compressed(arquivo, X=X, y=y)
    print(f"Arquivo salvo como '{arquivo}'.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preProcesso.py cry_data/data.csv")
        sys.exit(1)
    df = pd.read_csv(sys.argv[1])
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=df["is_cry"])
    for name, df_subset in zip(["Treinamento", "Teste"], [train_df, test_df]):
        processar_audios(name, df_subset)

