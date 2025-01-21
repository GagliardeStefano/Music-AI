import os

import numpy as np
from music21 import converter, note, chord
from Levenshtein import distance as levenshtein_distance
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from collections import Counter

def extract_notes(file):
    midi = converter.parse(file)
    notes = []
    for element in midi.flatten().notes:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append(".".join(str(p) for p in element.pitches))

    return notes

def levenshtein_similarity(file1, file2):
    notes1 = extract_notes(file1)
    notes2 = extract_notes(file2)
    return levenshtein_distance(notes1, notes2)


def statistical_comparison(file1, file2):
    notes1 = extract_notes(file1)
    notes2 = extract_notes(file2)
    counter1 = Counter(notes1)
    counter2 = Counter(notes2)
    common_keys = set(counter1.keys()).intersection(set(counter2.keys()))
    stat_diff = {key: abs(counter1.get(key, 0) - counter2.get(key, 0)) for key in common_keys}
    return sum(stat_diff.values())

def n_gram_similarity(file1, file2, n=3):
    notes1 = extract_notes(file1)
    notes2 = extract_notes(file2)
    ngrams1 = [tuple(notes1[i:i+n]) for i in range(len(notes1)-n+1)]
    ngrams2 = [tuple(notes2[i:i+n]) for i in range(len(notes2)-n+1)]
    common = set(ngrams1).intersection(set(ngrams2))
    return len(common)

def evaluate_midi_similarity(generated_file, test_file):
    print("Levenshtein...")
    lev_dist = levenshtein_similarity(generated_file, test_file)

    print("Statistical comparison...")
    stat_diff = statistical_comparison(generated_file, test_file)

    print("N-gram...")
    ngram_match = n_gram_similarity(generated_file, test_file)

    return {
        "Levenshtein Distance": lev_dist,
        "Statistical Difference": stat_diff,
        "N-gram Matches": ngram_match
    }


def compare_with_all_tests(generated_dir, test_dir):
    all_results = []
    print("Calcolo delle metriche per tutti i file generati...")
    for generated_file in os.listdir(generated_dir):
        generated_file_path = os.path.join(generated_dir, generated_file)
        print(f"\nAnalizzando file generato: {generated_file}")

        # Risultati per ogni file generato
        generated_results = []
        for test_file in os.listdir(test_dir):
            test_file_path = os.path.join(test_dir, test_file)
            result = evaluate_midi_similarity(generated_file_path, test_file_path)
            generated_results.append(result)

        # Calcolo della media per il file generato corrente
        avg_generated_results = {
            "Levenshtein Distance": sum(r["Levenshtein Distance"] for r in generated_results) / len(generated_results),
            "Statistical Difference": sum(r["Statistical Difference"] for r in generated_results) / len(
                generated_results),
            "N-gram Matches": sum(r["N-gram Matches"] for r in generated_results) / len(generated_results),
        }

        # Aggiungi la valutazione media del file generato alla lista
        avg_generated_results['generated_file'] = generated_file
        all_results.append(avg_generated_results)

    # Calcolo della media globale delle metriche per tutti i file generati
    avg_results = {
        "Levenshtein Distance": sum(r["Levenshtein Distance"] for r in all_results) / len(all_results),
        "Statistical Difference": sum(r["Statistical Difference"] for r in all_results) / len(all_results),
        "N-gram Matches": sum(r["N-gram Matches"] for r in all_results) / len(all_results),
    }

    # Normalizzazione e calcolo della percentuale di accuratezza complessiva
    max_possible_levenshtein = max(r["Levenshtein Distance"] for r in all_results)
    max_possible_statistical = max(r["Statistical Difference"] for r in all_results)

    lev_accuracy = 1 - (avg_results["Levenshtein Distance"] / max_possible_levenshtein)
    stat_accuracy = 1 - (avg_results["Statistical Difference"] / max_possible_statistical)
    ngram_accuracy = avg_results["N-gram Matches"]

    overall_accuracy = (lev_accuracy + stat_accuracy + ngram_accuracy) / 3
    accuracy_percentage = overall_accuracy * 100

    return avg_results, accuracy_percentage, all_results
