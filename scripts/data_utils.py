import pandas as pd
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from . import transliteration_tokenizers

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

PAD_ID = 0


class TransliterationDataset(Dataset):
    """Dataset for Transliteration purposes.Inherits from torch Dataset class"""

    def __init__(self, file, source_tokenizer, target_tokenizer, ascending=True):
        """[summary]

        Args:
            file (Pathlib path): File which has transliteration data i.e. Native script romanized word pairs
            source_tokenizer (hugging_face tokenizer): Tokenizer for source words i.e. romanized words
            target_tokenizer (hugging_face tokenizer): Tokenizer for target words i.e. native script words
            ascending (bool, optional): How to sort the dataset by source tokens length. Defaults to True.
        """

        self.df = pd.read_csv(file, sep="\t", header=None)
        self.columns = self.df.shape[1]
        names = ["target", "source"]
        self.df.columns = names if self.columns == 2 else names + ["weights"]
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.df["source_ids"] = self.df.source.apply(
            lambda x: self.source_tokenizer.encode(x).ids
        )
        self.df["target_ids"] = self.df.target.apply(
            lambda x: self.target_tokenizer.encode(x).ids
        )
        self.df["source_len"] = self.df.source_ids.apply(lambda x: len(x))
        self.df.sort_values(
            by="source_len", ascending=ascending, ignore_index=True, inplace=True
        )

    def __len__(self):
        """Computes number of samples

        Returns:
            int: Number of samples in the dataset
        """
        return len(self.df)

    def __getitem__(self, idx):
        """Returns sample correspodning to given index

        Args:
            idx (int): index number of sample

        Returns:
            tuple of lists: While source_ids is list of source tokens, target_ids is list of target tokens.
            Weights(optional) is weight of sample
        """

        source_ids = self.df.source_ids[idx]
        target_ids = self.df.target_ids[idx]
        if self.columns > 2:
            weights = self.df.weights[idx]
            return (source_ids, target_ids, weights)
        return (source_ids, target_ids)


def pad_collate(batch):
    """Collects a batch of samples of unequal length, pads them to create tensors which can be fed to model

    Args:
        batch (List of samples): List of samples

    Returns:
        tuple of Tensors: Return source tensor, target tensor, list of lengths of each source sample
        and weights(optional)
    """

    size = len(batch[0])
    if size == 2:
        source_ids, target_ids = zip(*batch)
    else:
        source_ids, target_ids, weights = zip(*batch)

    source_ids, target_ids = (
        source_ids[::-1],
        target_ids[::-1],
    )  # Order them in decreasing length
    source_ids = [torch.tensor(Id) for Id in source_ids]
    target_ids = [torch.tensor(Id) for Id in target_ids]
    source_lens = [len(Id) for Id in source_ids]
    source_padded = pad_sequence(source_ids, batch_first=True, padding_value=PAD_ID).to(
        device
    )
    target_padded = pad_sequence(target_ids, batch_first=True, padding_value=PAD_ID).to(
        device
    )

    if size == 2:
        return source_padded, target_padded, source_lens
    else:
        return source_padded, target_padded, source_lens, weights
