from functools import reduce

import numpy as np
import pandas as pd


def create_corpus(train_file, src_corpus_file, tgt_corpus_file):
    """Create native script and romanized corpus text files which consists of all unique words

    Args:
        train_file (.tsv file): Raw data file
        src_corpus_file ([type]): File location where romanized word corpus is written
        tgt_corpus_file ([type]): File location where native script word corpus is written
    """
    df_train = pd.read_csv(
        train_file, sep="\t", header=None, names=["target", "source", "frequency"]
    )
    tgt_corpus = " ".join(df_train.target.unique().tolist())
    src_corpus = " ".join(df_train.source.unique().tolist())

    open(tgt_corpus_file, "a").close()
    with open(tgt_corpus_file, "w", encoding="utf-8") as writer:
        writer.write(tgt_corpus)

    open(src_corpus_file, "a").close()
    with open(src_corpus_file, "w", encoding="utf-8") as writer:
        writer.write(src_corpus)


def create_max_weighted_and_repeated_tsv(
    raw_data_file, max_file, weighted_file, repeated_file
):
    """Takes a raw data file of (native script word , romanized word, count) and create max, weighted and repeated files

    Args:
        raw_data_file (.tsv file): Raw data file
        max_file (Pathlib path): File location where max data is written
        repeated_file (Pathlib path): File location where repeated data is written
        weighted_file (Pathlib path): File location where weighted data is written
    """

    raw_data = file_parser(raw_data_file)

    weighted_data = create_weighted_data(raw_data)
    write_weighted_files(weighted_data, weighted_file)

    max_data = create_max_data(raw_data)
    write_max_files(max_data, max_file)

    repeat_data = create_repeat_data(raw_data)
    write_repeat_files(repeat_data, repeated_file)


def write_repeat_files(data, file):
    """Creates a tsv file where each line is a native script word and romanized word and each line is repeated
    as per romanized word count

    Args:
        data (dict): Dictionary of native script word and corresponding list of repeated romanized words
        file (Pathlib path): File location where data is to be written
    """
    open(file, "a").close()
    with open(file, "w", encoding="utf-8") as writer:
        for (target, source) in data.items():
            for word in source:
                seq = target + "\t" + word + "\n"
                writer.writelines(seq)


def write_weighted_files(data, file):
    """Creates a tsv file where each line is a native script word, romanized word and its weight

    Args:
        data (dict): Dictionary of native script word and corresponding of list of (romanized word, weight)
        tuples
        file (Pathlib path): File location where data is to be written
    """
    open(file, "a").close()
    with open(file, "w", encoding="utf-8") as writer:
        for (target, source) in data.items():
            for (word, weight) in source:
                seq = target + "\t" + word + "\t" + str(weight) + "\n"
                writer.writelines(seq)


def write_max_files(data, file):
    """Creates a tsv file where each line is a native script word and corresponding romanized word with
    max count

    Args:
        data (dict): Dictionary of native script word and corresponding romanized word with highest count
        file (Pathlib path): FIle location where data is to be written
    """
    open(file, "a").close()
    with open(file, "w", encoding="utf-8") as writer:
        for (target, source) in data.items():
            seq = target + "\t" + source + "\n"
            writer.writelines(seq)


def create_repeat_data(data):
    """Create a dictionary of native word and corresponding list of romanized words  in which
    each word count is same as that of freuqency

    Args:
        data (dict): Key value pairs of native script word and corresponding list of
        (romanized word, frequency) tuples

    Returns:
        dict: Dictionary of native script word and corresponding list of repeated romanized words
    """
    repeat_data = {}
    for (target, source) in data.items():
        repeat_data[target] = reduce(
            lambda x, y: x + y, [[word] * count for word, count in source]
        )
    return repeat_data


def create_weighted_data(data):
    """Normalize frequencies of romanized_words to weights in the dict output by file_parser

    Args:
        data (dict): Key value pairs of native script word and corresponding list of
        (romanized word, frequency) tuples

    Returns:
        dict: Dictionary of native script word and corresponding list of
        (romanized word, weights) tuples
    """
    weighted_data = {}
    for (target, source) in data.items():
        words, freqs = list(zip(*source))
        weights = normalize(freqs)
        weighted_data[target] = list(zip(words, weights))
    return weighted_data


def create_max_data(data):
    """Create dictionaries of native word and romanized word with highest count

    Args:
        data (dict): Key value pairs of native script word and corresponding list of
        (romanized word, frequency) tuples

    Returns:
        dict: Dictionary of native script word and corresponding romanized word with highest count
    """
    max_data = {}
    for (target, source) in data.items():
        words, freqs = list(zip(*source))
        source_word = words[np.argmax(freqs)]
        max_data[target] = source_word
    return max_data


def file_parser(file):
    """Reads the data from file and stores it in a easily usable format

    Args:
        file (.tsv file): file which contains the native script words(target), romanization lexicons(source)
        and frequency of attestation of each romanization lexicon(count)

    Returns:
        dict: Dictionary where each entry is as follows
        native_script_word : [(romanized_word_1, count_1),(romanized_word_2, count_2),....] list of all possible
        romanized words and their counts

    """
    data = {}
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            target, source, count = line.strip().split("\t")
            try:
                data[target].append((source, int(count)))
            except:
                data[target] = [(source, int(count))]
    return data


def normalize(series):
    """Normalizes a series of values so that their sum is 1

    Args:
        series (tuple): Series of values to be normalizes

    Returns:
        tuple: Series of normalized values which sum up to 1
    """
    total = sum(series)
    return tuple(round((num / total), 3) for num in series)
