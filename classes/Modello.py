import os

from keras.src.optimizers import Adamax
from matplotlib import pyplot as plt
import pandas as pd

from functions.data_preprocessing import load_X_y

from sklearn.model_selection import train_test_split

import tensorflow
from keras.src.layers import LSTM, Dropout, Dense
from keras.src.models import Sequential
from keras.src.callbacks import ModelCheckpoint, LambdaCallback


def print_info_GPU():
    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
    # Controlla se ci sono GPU disponibili
    if tensorflow.config.list_physical_devices('GPU'):
        print("TensorFlow sta usando la GPU.")
    else:
        print("TensorFlow NON sta usando la GPU.")

class Modello:
    def __init__(self, compositore):
        print_info_GPU()

        self.compositore = compositore
        self.X, self.y = load_X_y(compositore, "train")


        print("X train",self.X.shape)
        print("y train",self.y.shape)


    def crea_struttura(self):
        # Initialising the Model
        model = Sequential()

        # Adding layers
        model.add(LSTM(512, input_shape=(self.X.shape[1], self.X.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(256))
        model.add(Dropout(0.2))

        model.add(Dense(256))
        model.add(Dropout(0.2))

        model.add(Dense(self.y.shape[1], activation='softmax'))

        model = self.compile_model(model)

        return model

    def compile_model(self, struct_model):
        model = struct_model

        # Compiling the model for training
        opt = Adamax(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=opt)

        return model

    def allena_modello(self, compiled_model, initial_epoch=0):

        model = compiled_model

        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


        # path per salvare i pesi dei checkpoint
        file_path_checkpoint = f"modelli_allenati/{self.compositore}/checkpoint.weights.h5"

        checkpoint = ModelCheckpoint(
            file_path_checkpoint,
            monitor='loss',  # Monitora la perdita sul set di allenamento
            verbose=1,  # Stampa mex
            save_best_only=True,  # Salva solo se migliora
            save_weights_only=True,  # Salva solo i pesi
            mode='min'  # Obiettivo minimizzare
        )

        save_history_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: pd.DataFrame({**logs, "epoch": [epoch]}).to_csv(
                f'modelli_allenati/{self.compositore}/history.csv', mode='a',
                header=not os.path.exists(f'modelli_allenati/{self.compositore}/history.csv'), index=False)
        )

        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=256,
                            initial_epoch=initial_epoch,
                            callbacks=[checkpoint, save_history_callback])

        return history, model

    def save_model(self, model):
        model.save(f'modelli_allenati/{self.compositore}/{self.compositore}.keras')
        print(f"Modello salvato in modelli_allenati/{self.compositore}/{self.compositore}.keras")

    def plot_training_validation_loss(self):
        # Percorso del file CSV
        csv_path = f'modelli_allenati/{self.compositore}/history.csv'

        try:
            # Carica il CSV
            history_df = pd.read_csv(csv_path)
            print("Dati caricati dal file CSV.")
        except FileNotFoundError:
            print(f"Errore: Il file '{csv_path}' non esiste.")
            return

        # Estrarre i dati
        loss = history_df['loss']
        #val_loss = history_df.get('val_loss', None)  # Se la colonna val_loss non esiste, None

        # Ottenere il numero di epoche
        epochs = history_df['epoch'] if 'epoch' in history_df.columns else range(1, len(loss) + 1)

        # Creare il grafico
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss, label='Training Loss', color='blue')

        #if val_loss is not None:
        #    plt.plot(epochs, val_loss, label='Validation Loss', color='orange')

        plt.title(f'Training Loss - {self.compositore}', fontsize=20)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()

        # Mostrare il grafico
        plt.show()

    def riprendi_allenamento(self, initial_epoch):
        file = f"modelli_allenati/{self.compositore}/checkpoint.weights.h5"
        model = self.crea_struttura() # ricarica la struttura
        print("Struttura caricata")

        model.load_weights(file) # carica i pesi migliori salvati
        print("Pesi caricati")

        # carica le history salvate
        history_df = pd.DataFrame()
        try:
            history_df = pd.read_csv(f'modelli_allenati/{self.compositore}/history.csv')
            print("Caricata la history precedente")
        except:
            print("File history non trovato o non esiste una history precedente")

        model_compiled = self.compile_model(model)
        print("Modello compilato")

        print("Avvio dell'allenamento...")
        history, model_allenato = self.allena_modello(model_compiled, initial_epoch)

        self.save_model(model_allenato)

        new_history_df = pd.DataFrame(history.history)
        # concatenazione history nuova con quella vecchia (del file)
        history_df = pd.concat([history_df, new_history_df], ignore_index=True)

        history_df.to_csv(f'modelli_allenati/{self.compositore}/history.csv', index=False)

        self.plot_training_validation_loss()
