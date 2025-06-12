import os
import librosa
import numpy as np

DATASET_DIR = "Dataset"
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

def processar_audios():
    X, y = [], []
    for rotulo, classe in enumerate(["not_cry", "cry"]):
        pasta = os.path.join(DATASET_DIR, classe)
        for arquivo in os.listdir(pasta):
            if arquivo.endswith(".wav"):
                caminho = os.path.join(pasta, arquivo)
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

    np.savez_compressed("preProcesso.npz", X=X, y=y)
    print("Arquivo salvo como 'preProcesso.npz'.")

if __name__ == "__main__":
    processar_audios()

