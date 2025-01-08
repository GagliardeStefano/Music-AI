from functions.data_exploration import filtered_data_composer_from_maestro, create_plot_dataset_distribution


def filtra_da_maestro(compositore_da_maestro):

    compositore = compositore_da_maestro.replace('é', 'e')
    src_dir = "maestro-v3.0.0"
    dest_dir = f"dataset/midi/{compositore}"
    new_csv_file = "dataset/infoMidi.csv"
    filtered_data_composer_from_maestro(compositore, src_dir, dest_dir, new_csv_file)

if __name__ == '__main__':
    #filtra_da_maestro("Frédéric Chopin")

    create_plot_dataset_distribution("Frederic Chopin")