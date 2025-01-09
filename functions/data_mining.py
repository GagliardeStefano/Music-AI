import json
import os

from music21 import converter, instrument, note, chord

from functions.data_exploration import get_stream_file


def extract_data_midi(compositore, split):
    dir_path = f"dataset/midi/{compositore}/{split}"

    notes = []
    for file in os.listdir(dir_path):
        print(file)
        midi = converter.parse(dir_path+"/"+file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)

        if parts:  # File contiene parti strumentali
            notes_to_parse = parts.parts[0].recurse()
        else:  # Struttura piatta
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note): # Nota
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord): # Accordo
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

def save_notes(compositore, split, notes):
    path_file = f"dataset/notes/{compositore}/{split}/notes.json"
    f = get_stream_file(path_file, "w")
    json.dump(notes, f)
    print(f"file salvato in {path_file}")