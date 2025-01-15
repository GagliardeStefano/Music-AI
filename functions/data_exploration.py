import csv
import json
import shutil
import os
import matplotlib.pyplot as plt
from collections import Counter


def get_stream_file(path_file, mode):
    f = open(path_file, mode)
    return f

def copy_midi_to_dir(src_dir, filename_maestro, dest_dir, split):
    # Copia i file MIDI e audio nella cartella di destinazione
    midi_src = os.path.join(src_dir, filename_maestro)

    # Crea una sottocartella per lo split, se non esiste
    split_dir = os.path.join(dest_dir, split)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    # Controlla se i file esistono prima di copiarli
    if os.path.exists(midi_src):
        shutil.copy(midi_src, split_dir)

def write_to_new_csv(filename_maestro, dest_dir, canonical_composer, split, writer):
    # Prendi il nuovo nome del file, il nuovo path e il nuovo nome compositore

    print("print row")
    midi_filename = filename_maestro.split('/')[-1]
    path_file = dest_dir + "/" + split + "/" + midi_filename
    composer = canonical_composer.replace('é', 'e')

    # Scrivi i dati del compositore nel nuovo CSV
    writer.writerow(
        [composer, split, midi_filename, path_file])

def filtered_data_composer_from_maestro(compositore, src_dir, dest_dir, new_csv_file):
    csv_maestro = "maestro-v3.0.0/maestro-v3.0.0.csv"

    # check della cartella di destinazione
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # file CSV per scrivere i dati filtrati
    with open(new_csv_file, mode='w', newline='') as new_csv:
        writer = csv.writer(new_csv)

        # intestazione del nuovo CSV
        writer.writerow(['composer', 'filename', 'split', 'path_file'])

        # Apri il CSV originale e leggi i dati
        with open(csv_maestro,encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Salta l'intestazione

            for row in reader:
                canonical_composer, canonical_title, split, year, anno_midi_filename, audio_filename, duration = row

                if canonical_composer == compositore:

                    # Funzione per scrivere il corpo del nuovo file csv
                    write_to_new_csv(anno_midi_filename, dest_dir, canonical_composer, split, writer)

                    # Funzione per copiare midi in un'altra cartella
                    copy_midi_to_dir(src_dir, anno_midi_filename, dest_dir, split)


    print(
        f"Dati filtrati per {compositore} sono stati scritti in {new_csv_file} e i file sono stati copiati nella cartella {dest_dir}")


def create_plot_dataset_distribution(compositore):
    csv_file = "dataset/infoMidi.csv"

    f = get_stream_file(csv_file, 'r')
    csv_reader = csv.reader(f)

    splits = []

    # Leggi i dati e filtra per compositore
    for row in csv_reader:
        composer, split, filename, path_file = row
        if composer == compositore:
            splits.append(split)

    # Conta la distribuzione degli split
    split_counts = Counter(splits)

    # Crea il grafico
    plt.figure(figsize=(8, 6))
    plt.bar(split_counts.keys(), split_counts.values(), color='skyblue')

    # Aggiungi titolo e etichette
    plt.title(f"Distribuzione del dataset per {compositore}")
    plt.xlabel("Split")
    plt.ylabel("Numero di file")

    # Mostra il grafico
    plt.show()

def get_notes(compositore, split):
    path_file = f"dataset/notes/{compositore}/{split}/notes.json"
    f = get_stream_file(path_file, "r")
    notes = json.load(f)
    return notes

def get_pitchnames(notes):
    pitchnames = sorted(list(set(notes)))
    return pitchnames