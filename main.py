from keras.src.saving.saving_lib import load_model

from classes.Modello import Modello
from functions.data_exploration import filtered_data_composer_from_maestro, get_notes
from functions.data_generation import generate_notes, create_track
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

    ''' ALLENARE UN NUOVO MODELLO   
    modello = Modello("Frederic_Chopin")

    struct_model = modello.crea_struttura()
    history, model_allenato = modello.allena_modello(struct_model, 100)

    modello.save_model(model_allenato)
    modello.create_plot_training_history(history)
    '''

    ''' RIPRENDI ALLENAMENTO    
    modello = Modello("Frederic_Chopin")
    modello.riprendi_allenamento(99) # l'epoca (esclusa) da cui riprendere l'allenamento sul numero di epoche massimo
    '''


