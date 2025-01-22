# Music-AI
Music-AI è un generatore di musica classica sullo stile di Frédéric Chopin. 
Utilizzando il machine learning, il modello è in grado di apprendere dalle opere di Chopin e generare nuove tracce musicali nello stesso stile con un'accuratezza del 46.31%.

I file MIDI utilizzati per l'addestramento provengono dal dataset [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) V3.0.0, reso pubblico da Magenta.

Music-AI si basa su un modello LSTM (Long Short-Term Memory), un tipo di rete neurale ricorrente (RNN), per generare la musica.
## Contenuto
Nella cartella [dataset](https://github.com/GagliardeStefano/Music-AI/blob/master/dataset) 
sono presenti tutti i file necessari per l'addestramento del modello:
- [midi/Frederic_Chopin](https://github.com/GagliardeStefano/Music-AI/tree/master/dataset/midi/Frederic_Chopin)
contiene i file MIDI suddivisi per "test" e "train".
- [notes/Frederic_Chopin/train](https://github.com/GagliardeStefano/Music-AI/tree/master/dataset/notes/Frederic_Chopin/train)
contiene un documento JSON con note e accordi estratti dai file MIDI di "train".
- [plots](https://github.com/GagliardeStefano/Music-AI/tree/master/dataset/plots)
contiene un immagine PNG che mostra la distribuzione iniziale dei file MIDI
- [infoMidi.csv](https://github.com/GagliardeStefano/Music-AI/blob/master/dataset/infoMidi.csv)
con le seguenti informazioni:

| Colonna              | Descrizione                                                                                       |
|----------------------|---------------------------------------------------------------------------------------------------|
| `composer`           | Il nome del compositore a cui appartiene il brano MIDI.                                           |
| `split`              | Indica se il file MIDI appartiene alla sezione di "train" (addestramento) o di "test" (verifica). |
| `filename`           | Il nome del file MIDI.                                                                            |
| `path_file`          | Il percorso completo al file MIDI nel sistema.                                                    |

Nel package [functions](https://github.com/GagliardeStefano/Music-AI/tree/master/functions) 
si trovano le funzioni utili per lavorare con i dati e gestire il modello.

Nella cartella [modelli_allenati/Frederic_Chopin](https://github.com/GagliardeStefano/Music-AI/tree/master/modelli_allenati/Frederic_Chopin)
è presente il modello LSTM addestrato (_file_.keras), i checkpoint dei pesi
migliori (_file_.weights.h5) e un file _history.csv_ contenente i valori
di _loss_ e _val_loss_ per epoca.

Nella cartella [classes](https://github.com/GagliardeStefano/Music-AI/tree/master/classes)
è presente il file _Modello.py_, usato per creare, addestrare e salvare il modello LSTM.

Nella cartella [tracce_generate/Frederic_Chopin](https://github.com/GagliardeStefano/Music-AI/tree/master/tracce_generate/Frederic_Chopin)
sono presenti tutti i file MIDI generati dal modello LSTM.
## Costruito con
- python 3.12

Librerire principali:
- `pandas 2.2.3`: utilizzato per il modello LSTM
- `scikit-learn 1.6.0`: utilizzato per effettuare lo split delle finestre
di scorrimento in "train" e "validation"
- `music21 9.3.0`: utilizzato per analizzare ed estrarre informazioni dai file MIDI
- `keras 3.8.0` + `tensorflow 2.18.0`: usati per costruire il modello LSTM

## Come replicare
### 1. Requisiti
Installa tutte le dipendenze richieste usando:
```
pip install -r requirements.txt
```
### 2. Estrai eventi musicali
Una volta presi tutti i file MIDI suddivisi per "train" e "test"
, estrai tutti gli eventi musicali (note e accordi) da ogni file di "train" utilizzando
la funzione `extract_data_midi(compositore, split)` e salva il risultato in un documento JSON con `save_notes(compositore, split, notes)`.
Entrambe le funzioni sono all'interno di [functions/data_mining.py](https://github.com/GagliardeStefano/Music-AI/blob/master/functions/data_mining.py)
### 3. Crea le finestre di scorrimento
Una volta creato il documento JSON con tutti gli eventi, esegui la funzione `crea_X_y(notes, compositore, split)`,
presente in [functions/data_preprocessing.py](https://github.com/GagliardeStefano/Music-AI/blob/master/functions/data_preprocessing.py),
per creare le 
finestre di scorrimento adatte per allenare un modello LSTM.
### 4. Crea e allena il modello
Create le finestre X e y, usa la classe Modello presente nel file [classes/Modello.py](https://github.com/GagliardeStefano/Music-AI/blob/master/classes/Modello.py).
Chiama il costruttore, crea la struttura del modello, compila il modello, allenalo e poi salvalo.
Un esempio è il seguente:
```
modello = Modello("Frederic_Chopin")

struct_model_compiled = modello.crea_struttura()
history, model_allenato = modello.allena_modello(struct_model_compiled)

modello.save_model(model_allenato)
```
Opzionale, puoi aggiungere questa funzione per creare un grafico 
che mostra le performance durante l'allenamento:
```
modello.plot_training_validation_loss()
```

### 5. Genera nuove tracce musicali
Una volta salvato il modello addestrato, puoi usarlo per generare
nuove tracce musicali. Di seguito il codice:
```
model = load_model(f"modelli_allenati/Frederic_Chopin/Frederic_Chopin.keras")
X, _ = load_X_y("Frederic_Chopin", "train")

music, melody_midi = create_track(model, "Frederic_Chopin", X)
print(music)
save_track(compositore, melody_midi)
```
- `load_model()` permette di caricare il modello allenato.
- `load_X_y()` permette di caricare le finestre di scorrimento
precedentemente create -> [functions/data_preprocessing.py](https://github.com/GagliardeStefano/Music-AI/blob/master/functions/data_preprocessing.py)
- `create_track()` permette di generera una traccia musicale
dato il modello -> [functions/data_generation.py](https://github.com/GagliardeStefano/Music-AI/blob/master/functions/data_generation.py)
- `save_track()` permette di salvare la traccia musicale in formato .mid
-> [functions/data_generation.py](https://github.com/GagliardeStefano/Music-AI/blob/master/functions/data_generation.py)

### 6. Valuta il modello
Una volta create un numero considerevole di tracce puoi valutare
le performance del tuo modello. Usando questo codice:
```
generated_folder = f"tracce_generate/Frederic_Chopin"
test_folder = f"dataset/midi/Frederic_Chopin/test"

avg_results, accuracy_percentage, all_results = compare_with_all_tests(generated_folder, test_folder)
print("Risultato medio", avg_results)
print("Accuracy Modello", accuracy_percentage)
print("Tutti i risultati", all_results)
```
- `generated_folder` e `test_folder` sono rispettivamente le cartelle
che contengono i file generati dal modello e i file di test del compositore
- `compare_with_all_tests()` permette di ottenere i valori necessari
per valutare l'accuratezza del tuo modello -> 
[functions/data_accuracy.py](https://github.com/GagliardeStefano/Music-AI/blob/master/functions/data_accuracy.py)

# Riferimenti
Dataset MAESTRO:
```
Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang,
  Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck. "Enabling
  Factorized Piano Music Modeling and Generation with the MAESTRO Dataset."
  In International Conference on Learning Representations, 2019.
  
@inproceedings{
  hawthorne2018enabling,
  title={Enabling Factorized Piano Music Modeling and Generation with the {MAESTRO} Dataset},
  author={Curtis Hawthorne and Andriy Stasyuk and Adam Roberts and Ian Simon and Cheng-Zhi Anna Huang and Sander Dieleman and Erich Elsen and Jesse Engel and Douglas Eck},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=r1lYRjC9F7},
}
```


