from classes.Modello import Modello
from functions.data_exploration import filtered_data_composer_from_maestro, get_notes
from functions.data_mining import extract_data_midi, save_notes
from functions.data_preprocessing import crea_X_y, load_X_y


def filtra_da_maestro(compositore_da_maestro):

    compositore = compositore_da_maestro.replace('é', 'e')
    src_dir = "maestro-v3.0.0"
    dest_dir = f"dataset/midi/{compositore}"
    new_csv_file = "dataset/infoMidi.csv"
    filtered_data_composer_from_maestro(compositore, src_dir, dest_dir, new_csv_file)

def estrai_salva_note_accordi(compositore, split):
    notes = extract_data_midi(compositore, split)
    save_notes(compositore, split, notes)



if __name__ == '__main__':
    compositore = "Frederic_Chopin"
    split = "train"
    '''
    notes = get_notes(compositore, split)
    crea_X_y(notes, compositore, split)
    '''
    modello = Modello("Frederic_Chopin")

    struct_model = modello.crea_struttura()
    history, model_allenato = modello.allena_modello(struct_model)

    modello.save_model(model_allenato)
    modello.create_plot_training_history(history)

