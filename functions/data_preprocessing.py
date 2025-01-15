
import numpy as np
from tensorflow.python.keras.utils import np_utils

from functions.data_exploration import get_pitchnames


def crea_X_y(notes, compositore, split):
    length = 100
    features = []
    targets = []
    pithnames = get_pitchnames(notes)
    mapping,_ = build_dictonary(notes)

    for i in range(0, len(notes) - length, 1):
        feature = notes[i:i + length]
        target = notes[i + length]
        features.append([mapping[j] for j in feature])
        targets.append(mapping[target])

    # reshape X and normalize
    X = (np.reshape(features, (len(targets), length, 1))) / float(len(pithnames))
    # one hot encode the output variable
    y = np_utils.to_categorical(targets)

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

def load_X_y(compositore, split):
    path_dir = f"finestre_scorrimento/{compositore}/{split}"
    X = np.load(f"{path_dir}/X.npy")
    y = np.load(f"{path_dir}/y.npy")

    print(f"caricate finestre {path_dir}/X.npy e {path_dir}/y.npy ")
    return X, y

def build_dictonary(notes):
    pitchnames = get_pitchnames(notes)

    mapping = dict((c, i) for i, c in enumerate(pitchnames))
    reverse_mapping = dict((i, c) for i, c in enumerate(pitchnames))

    return mapping, reverse_mapping