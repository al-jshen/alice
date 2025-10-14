from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from .ciphers import CryptogramCipher


def is_valid_row(example):
    return not (not example["text"] or "..." in example["text"])


def is_valid_length(example, min_length, max_length):
    return min_length <= len(example["text"]) <= max_length


def is_valid_chars(example, extra_chars=".,!?-;:'\""):
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ " + extra_chars)
    return set(example["text"]).issubset(allowed_chars)


def remove_punctuation(example):
    example["text"] = "".join(
        char for char in example["text"] if char.isalpha() or char.isspace()
    )
    return example


def replace_accents(example):
    # Replace accented characters with their unaccented equivalents
    accents = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "Á": "A",
        "É": "E",
        "Í": "I",
        "Ó": "O",
        "Ú": "U",
        "ñ": "n",
        "Ñ": "N",
        "ä": "a",
        "ë": "e",
        "ï": "i",
        "ö": "o",
        "ü": "u",
        "Ä": "A",
        "Ë": "E",
        "Ï": "I",
        "Ö": "O",
        "Ü": "U",
        "à": "a",
        "è": "e",
        "ì": "i",
        "ò": "o",
        "ù": "u",
        "À": "A",
        "È": "E",
        "Ì": "I",
        "Ò": "O",
        "Ù": "U",
        "â": "a",
        "ê": "e",
        "î": "i",
        "ô": "o",
        "û": "u",
        "Â": "A",
        "Ê": "E",
        "Î": "I",
        "Ô": "O",
        "Û": "U",
        "ç": "c",
        "Ç": "C",
        "å": "a",
        "Å": "A",
        "ø": "o",
        "Ø": "O",
        "œ": "oe",
        "Œ": "OE",
        "Æ": "AE",
        "æ": "ae",
        "ß": "ss",
        "ð": "d",
        "Ð": "D",
        "þ": "th",
        "Þ": "Th",
        "ý": "y",
        "Ý": "Y",
    }
    example["text"] = "".join(accents.get(char, char) for char in example["text"])
    return example


def fix_spaces(example):
    example["text"] = (
        example["text"]
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
        .replace("  ", " ")
        .strip()
    )
    return example


# edit the dataset to fix punctuation: always have a space after punctuation except for the last punctuation mark and for apostrophes and hyphens
def fix_punctuation(example):
    text = example["text"]
    # Add space after punctuation only if not already there except for the last punctuation mark
    for punct in ".,!?;:":
        text = text.replace(punct, punct + " ")
    # Remove space before punctuation except for apostrophes and hyphens
    for punct in ".,!?;:":
        text = text.replace(" " + punct, punct)
    # Replace multiple spaces with a single space
    text = " ".join(text.split())
    # Ensure no trailing spaces
    text = text.strip()
    example["text"] = text
    return example


def capitalize_all(example):
    example["text"] = example["text"].upper()
    return example


def custom_collate(list_of_dicts):
    keys = list_of_dicts[0].keys()
    dict_of_lists = {}
    for k in keys:
        if isinstance(list_of_dicts[0][k], torch.Tensor):
            # If the value is a tensor, stack them
            dict_of_lists[k] = torch.stack([d[k] for d in list_of_dicts])
        else:
            dict_of_lists[k] = [d[k] for d in list_of_dicts]
    return dict_of_lists


def tokenize(example, mapping, max_length, pad_value):
    text_tok = torch.tensor(
        [mapping.get(char, -1) for char in example["text"]], dtype=torch.long
    )
    encrypted_tok = torch.tensor(
        [mapping.get(char, -1) for char in example["encrypted"]], dtype=torch.long
    )

    padding = torch.full((max_length - len(text_tok),), pad_value, dtype=torch.long)
    example["text_tok"] = torch.cat([text_tok, padding])
    example["encrypted_tok"] = torch.cat([encrypted_tok, padding])

    return example


# mask out the padding tokens, optionally can also mask punctuation and spaces
def masking(example, pad_value, punctuation=False, spaces=False, col_name="mask"):
    # mask only padding tokens
    if not punctuation and not spaces:
        mask = example["encrypted_tok"] != pad_value
    # everything always masks padding tokens
    # mask punctuation but not spaces
    elif punctuation and not spaces:
        mask = (example["encrypted_tok"] != pad_value) & (
            example["encrypted_tok"] < 27  # our tokenization uses 26 for space
        )
    # mask spaces but not punctuation
    elif not punctuation and spaces:
        mask = (example["encrypted_tok"] != pad_value) & (
            example["encrypted_tok"] != 26
        )
    # mask punctuation and spaces
    else:
        mask = example["encrypted_tok"] < 26

    example[col_name] = mask
    return example


def apply_cipher(example, cipher):
    example["encrypted"] = cipher.encrypt(example["text"])
    example["cipher"] = cipher
    return example


def transform_one(
    example,
    cipher_kwargs,
    cipher_type,
    vocab,
    tok_mapping,
    max_length,
    loss_mask_punctuation,
    loss_mask_spaces,
):
    cipher = dict(cryptogram=CryptogramCipher)[cipher_type](
        **cipher_kwargs  # limit_n, seed included
    )

    # add columns with the cipher applied to the text and the cipher itself
    example = apply_cipher(example, cipher)

    # tokenize
    pad_value = len(vocab)
    example = tokenize(example, tok_mapping, max_length, pad_value)

    # attention mask
    example = masking(
        example, pad_value, punctuation=False, spaces=False, col_name="mask"
    )
    example = masking(
        example,
        pad_value,
        punctuation=loss_mask_punctuation,
        spaces=loss_mask_spaces,
        col_name="loss_mask",
    )

    return example


def transform_batch(
    examples,
    cipher_kwargs,
    cipher_type,
    vocab,
    tok_mapping,
    max_length,
    loss_mask_punctuation=True,
    loss_mask_spaces=True,
):
    # examples comes in a dict with keys as col names, and values are iterables
    # so, each val has same length = number of examples in batch
    # we want to transform them into a list of dicts
    # where each dict just represents a single example in the batch
    keys = list(examples.keys())
    batch_size = len(examples[keys[0]])

    transposed = [
        dict(zip(keys, [examples[k][i] for k in keys])) for i in range(batch_size)
    ]

    transformed = [
        transform_one(
            example,
            cipher_kwargs,
            cipher_type,
            vocab,
            tok_mapping,
            max_length,
            loss_mask_punctuation,
            loss_mask_spaces,
        )
        for example in transposed
    ]

    # now we want to retranpose

    out = {k: [d[k] for d in transformed] for k in transformed[0].keys()}

    return out


def get_dataloader(
    dataset="jstet/quotes-500k",
    batch_size=32,
    shuffle=True,
    extra_chars=".,!?-;:'\"",
    num_workers=8,
    cipher_type="caesar",
    cipher_kwargs={},
    test_cipher_type=None,
    test_cipher_kwargs=None,
    min_length=15,
    max_length=500,
    test_size=0.1,
    pin_memory=True,
    drop_last=False,
    loss_mask_punctuation=True,
    loss_mask_spaces=True,
    seed=0,
):
    # load 500k quotes dataset
    ds = load_dataset(dataset)

    # rename the 'quote' column to 'text'
    if "quote" in ds.column_names["train"]:
        ds = ds.rename_column("quote", "text")

    # Apply the punctuation fixes to the dataset
    ds = ds.filter(partial(is_valid_row), num_proc=num_workers)
    ds = ds.map(capitalize_all, num_proc=num_workers)
    ds = ds.filter(
        partial(is_valid_chars, extra_chars=extra_chars), num_proc=num_workers
    )
    ds = ds.map(fix_punctuation, num_proc=num_workers)
    ds = ds.filter(
        partial(is_valid_length, min_length=min_length, max_length=max_length),
        num_proc=num_workers,
    )

    vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZ " + extra_chars
    tok_mapping = {char: idx for idx, char in enumerate(vocab)}

    if "test" not in ds.keys():
        ds = ds["train"].train_test_split(test_size=test_size, seed=seed)

    ds_train = ds["train"]
    ds_test = ds["test"]

    print("Number of training examples:", len(ds_train))
    print("Number of test examples:", len(ds_test))

    # apply the transforms on the fly to the dataset: cipher, tokenize, masking
    # seed goes in cipher_kwargs
    ds_train.set_transform(
        partial(
            transform_batch,
            cipher_kwargs=cipher_kwargs,
            cipher_type=cipher_type,
            vocab=vocab,
            tok_mapping=tok_mapping,
            max_length=max_length,
            loss_mask_punctuation=loss_mask_punctuation,
            loss_mask_spaces=loss_mask_spaces,
        )
    )

    if test_cipher_type is None:
        test_cipher_type = cipher_type
    if test_cipher_kwargs is None:
        test_cipher_kwargs = cipher_kwargs

    ds_test.set_transform(
        partial(
            transform_batch,
            cipher_kwargs=test_cipher_kwargs,
            cipher_type=test_cipher_type,
            vocab=vocab,
            tok_mapping=tok_mapping,
            max_length=max_length,
            loss_mask_punctuation=loss_mask_punctuation,
            loss_mask_spaces=loss_mask_spaces,
        )
    )

    # make a torch dataloader for the dataset
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )
    return dl_train, dl_test, tok_mapping
