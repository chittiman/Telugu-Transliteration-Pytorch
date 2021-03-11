import tokenizers  # Version 0.9.4
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


def create_source_target_tokenizers(
    src_corpus_file, tgt_corpus_file, src_vocab_size=30, tgt_vocab_size=66
):
    """Create source and target tokenizers

    Args:
        src_corpus_file (Pathlib path): Source(i.e romanized words) corpus file
        tgt_corpus_file (Pathlib path): Target(native script words) corpus file
        src_vocab_size (int, optional): Source vocabulary size. Defaults to 30. At default value tokenizer
        tokenizer acts as a character level tokenizer
        tgt_vocab_size (int, optional): Target vocabulary size. Defaults to 66. At default value tokenizer
        tokenizer acts as a character level tokenizer

    Returns:
        tuple of tokenizers: Source and Target word tokenizers
    """
    src_tokenizer = create_tokenizer(src_corpus_file, src_vocab_size)
    tgt_tokenizer = create_tokenizer(tgt_corpus_file, tgt_vocab_size)
    return (src_tokenizer, tgt_tokenizer)


def create_tokenizer(corpus_file, vocab_size):
    """Create a tokenizer from a corpus file

    Args:
        corpus_file (Pathlib path): File containng corpus i.e. all unique words for
        vocab_size (int): Vocabulary size of the tokenizer

    Returns:
        hugging_face tokenizer: Byte pair tokenizer used to tokenize text
    """
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"], vocab_size=vocab_size
    )
    tokenizer.pre_tokenizer = Whitespace()
    files = [str(corpus_file)]
    tokenizer.train(trainer, files)
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    tokenizer.enable_padding(
        pad_token="<pad>",
        pad_id=tokenizer.token_to_id("<pad>"),
    )
    return tokenizer
