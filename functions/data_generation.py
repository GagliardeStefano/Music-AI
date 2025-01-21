import os

import numpy as np
from music21 import note, instrument, chord, stream

from functions.data_preprocessing import load_X_y, build_dictonary
from functions.data_exploration import get_pitchnames, get_notes

def chord_n_notes(notes):
    melody = []
    offset = 0
    out_str = []

    for n in notes:
        #accordo
        if "." in n or n.isdigit():
            out_str.append("accordo")
            chord_notes = n.split(".")
            notes_in_chord = []

            for nc in chord_notes:
                int_note = int(nc)
                note_snip = note.Note(int_note)
                notes_in_chord.append(note_snip)

                chord_snip = chord.Chord(notes_in_chord)
                chord_snip.offset = offset
                melody.append(chord_snip)
        else:
            #nota
            out_str.append("nota")
            note_snip = note.Note(n)
            note_snip.offset = offset
            melody.append(note_snip)

        offset += 1
    melody_midi = stream.Stream(melody)
    #print(out_str)
    return melody_midi

def create_track(model, compositore, X, note_count=200):
    print("caricamento informazioni...")
    X_seed = X
    notes = get_notes(compositore, "train")
    pithnames = get_pitchnames(notes)
    _, reverse_mapping = build_dictonary(notes)

    print("creazione traccia...")
    seed = X_seed[np.random.randint(0,len(X_seed)-1)]
    Music = ""
    Notes_Generated=[]
    for i in range(note_count):
        seed = seed.reshape(1,100,1)
        prediction = model.predict(seed, verbose=0)[0]
        prediction = np.log(prediction) / 1 #diversità
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index/ float(len(pithnames))
        Notes_Generated.append(index)
        Music = [reverse_mapping[char] for char in Notes_Generated]
        seed = np.insert(seed[0],len(seed[0]),index_N)
        seed = seed[1:]
    melody = chord_n_notes(Music)
    melody_midi = stream.Stream(melody)
    return Music, melody_midi

def save_track(compositore, melody_midi):
    print("salvataggio traccia...")
    dir = f"tracce_generate/{compositore}"

    file_index = 1
    while os.path.exists(f"{dir}/output_melody_{file_index}.mid"):
        file_index += 1

    midi_filename = f"output_melody_{file_index}.mid"
    full_path = dir + "/" + midi_filename

    melody_midi.write('midi', fp=full_path)
    print(f"File MIDI salvato come {full_path}")