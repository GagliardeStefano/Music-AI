import numpy as np
from music21 import note, instrument, chord, stream

from functions.data_preprocessing import load_X_y, build_dictonary
from functions.data_exploration import get_pitchnames, get_notes



def create_track(model, compositore, split, Note_Count=100):
    print("creazione traccia...")
    X_seed, _ = load_X_y(compositore, split)
    notes = get_notes(compositore, split)
    pithnames = get_pitchnames(notes)
    _, reverse_mapping = build_dictonary(notes)

    seed = X_seed[np.random.randint(0,len(X_seed)-1)]
    Music = ""
    Notes_Generated=[]
    for i in range(Note_Count):
        seed = seed.reshape(1,100,1)
        prediction = model.predict(seed, verbose=0)[0]
        prediction = np.log(prediction) / 1.5 #diversità
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index/ float(len(pithnames))
        Notes_Generated.append(index)
        Music = [reverse_mapping[char] for char in Notes_Generated]
        seed = np.insert(seed[0],len(seed[0]),index_N)
        seed = seed[1:]
    return Music


def save_music_to_midi(Music, filename="tracce_generate/Frederic_Chopin/generated_music2.mid"):
    """
    Salva la musica generata in un file MIDI.

    Args:
        Music (list): Lista di note generate dalla funzione `create_track`.
        filename (str): Nome del file MIDI in cui salvare la musica.
    """
    midi_stream = stream.Stream()

    for item in Music:
        if '.' in item:  # Riconosce un accordo
            try:
                notes_in_chord = [int(interval) for interval in item.split('.')]
                c = chord.Chord(notes_in_chord)  # Crea un accordo MIDI
                midi_stream.append(c)
            except ValueError:
                print(f"Formato di accordo non valido ignorato: {item}")
        else:  # Riconosce una nota
            try:
                n = note.Note(item)  # Crea una nota MIDI
                midi_stream.append(n)
            except:
                print(f"Nota non valida ignorata: {item}")

    midi_stream.write('midi', fp=filename)
    print(f"Musica salvata in formato MIDI in '{filename}'")