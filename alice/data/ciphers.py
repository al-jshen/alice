import random
from functools import lru_cache
from typing import Optional

from scipy.stats import kendalltau


@lru_cache
def n_mappings(limit_n: int, seed: int, limit_kendall_tau: Optional[float] = None):
    """Generate a (cached) list of random letter mappings for ciphers.
    Optionally, limit the Kendall tau distance between mappings.

    Args:
        limit_n (int): Number of mappings to generate.
        seed (int): Seed for random number generation.
        limit_kendall_tau (float, optional): Minimum Kendall tau distance
            between mappings, as compared to the first mapping.
            Lower values indicate more diversity in the mappings.
    """

    if limit_kendall_tau is not None:
        assert -1 <= limit_kendall_tau <= 1, (
            "limit_kendall_tau must be between -1 and 1."
        )

    mappings = []
    reference_mapping = None

    i = 0
    while len(mappings) < limit_n:
        alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        shuffled_alphabet = alphabet[:]
        random.Random(seed + i).shuffle(shuffled_alphabet)
        current_mapping = {k: v for k, v in zip(alphabet, shuffled_alphabet)}

        # save first mapping as reference
        if reference_mapping is None:
            reference_mapping = current_mapping

        if limit_kendall_tau is not None:
            # Compute Kendall tau distance and filter
            tau, _ = kendalltau(
                list(reference_mapping.values()), list(current_mapping.values())
            )
            if tau > limit_kendall_tau:
                mappings.append(current_mapping)
        else:
            mappings.append(current_mapping)

        i += 1

    return mappings


class Cipher:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

        extra_chars = " .,!?-;:'\""
        for char in extra_chars:
            if char not in self.mapping:
                self.mapping[char] = char

        self.inverse_mapping = {v: k for k, v in mapping.items()}

    def encrypt(self, text):
        encrypted = "".join(self.mapping.get(char, char) for char in text)
        return encrypted

    def decrypt(self, encrypted_text):
        decrypted = "".join(
            self.inverse_mapping.get(char, char) for char in encrypted_text
        )
        return decrypted


class CryptogramCipher(Cipher):
    """A simple substitution cipher that randomly shuffles the alphabet."""

    def __init__(
        self,
        mapping: Optional[dict[str, str]] = None,
        seed: Optional[int] = None,
        limit_n: Optional[int] = None,
    ):
        if limit_n is not None:
            mappings = n_mappings(limit_n, seed)
            mapping = random.choice(mappings)
            super().__init__(mapping)

        else:
            if mapping is None:
                alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                shuffled_alphabet = alphabet[:]
                if seed is not None:
                    random.Random(seed).shuffle(shuffled_alphabet)
                else:
                    random.shuffle(shuffled_alphabet)
                mapping = {k: v for k, v in zip(alphabet, shuffled_alphabet)}
            super().__init__(mapping)

