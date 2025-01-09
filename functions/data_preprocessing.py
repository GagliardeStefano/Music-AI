
import numpy as np
from tensorflow.python.keras.utils import np_utils

from functions.data_exploration import get_pitchnames


def crea_X_y(notes, compositore, split):
    sequence_length = 100

    # Prende tutti i nomi dei pitch
    pitchnames = get_pitchnames(notes)
    n_vocab = len(pitchnames)

    print("dim vocab: ",n_vocab)

    # Crea un dizionario per mappare le altezze delle note in numeri interi
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    X = []
    y = []

    # Crea sequenze
    print("creazione finestre...")
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        X.append([note_to_int[char] for char in sequence_in])
        y.append(note_to_int[sequence_out])

    n_patterns = len(X)

    # Rimodella l'input in un formato compatibile con LSTM
    X = np.reshape(X, (n_patterns, sequence_length, 1))

    # Normalizzazione
    X = X / float(n_vocab)
    y = np_utils.to_categorical(y)

    save_X_y(X, y, compositore, split)

def save_X_y(X, y, compositore, split):
    # Salva le finestre
    print("salvataggio...")

    path_dir = f"finestre_scorrimento/{compositore}/{split}"
    np.save(f'{path_dir}/X.npy', X)
    np.save(f'{path_dir}/y.npy', y)

    print(f"salvati in")
    print(f"{path_dir}/X.npy")
    print(f"{path_dir}/y.npy")