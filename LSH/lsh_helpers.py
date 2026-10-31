import pickle
import re

from datasketch import LeanMinHash, MinHash, MinHashLSH
from num2words import num2words

from settings import LSH_SIM_THRESHOLD, PERMUTATIONS, SHINGLE_SIZE


class LshHelper:

    def __init__(self) -> None:
        self.num_permutations = PERMUTATIONS
        self.shingle_size = SHINGLE_SIZE
        self.lsh_sim_threshold = LSH_SIM_THRESHOLD

    def instantiate_lsh_object(self) -> MinHashLSH:
        """Instantiate the Locality Sensitive Hashing object."""
        lsh_object = MinHashLSH(threshold=self.lsh_sim_threshold,
                                num_perm=self.num_permutations)
        return lsh_object

    def preprocess(self, review: str) -> str:
        """
        Preprocesses the review:
            - replace all whitespace, newline, return, tab or form characters with a single space
            - pass numbers to words
            - exclude some special characters
            - convert all to lower case
            - get rid of multiple white spaces and trailing spaces.
        """
        tmp_string = re.sub(r"[\s\n\r\t\f]+", " ", review)
        tmp_string = re.sub(r"[$&+#|<>^*%]+", "", tmp_string).lower()
        tmp_string = re.sub(r"(\d+)",
                            lambda x: num2words(int(x.group(0))),
                            tmp_string)
        result = re.sub(r"\s+", " ", tmp_string).strip(' ')
        return result

    def to_shingles(self, review: str) -> set[str]:
        """
        Get k-shingle representation of each review.
        """
        k_shingles = {review[i:(i + self.shingle_size)]
                      for i in range(len(review) - self.shingle_size + 1)}
        return set(k_shingles)

    def take_minhash(self, review: str) -> bytes:
        """
        Get a minhash signature for a set of k-shingles.
        """
        review_shingles = self.to_shingles(review)

        review_minhash = MinHash(self.num_permutations)

        for shingle in review_shingles:
            review_minhash.update(shingle.encode("utf8"))

        # Lean MinHash: smaller memory footprint and faster deserialization,
        # negligible now but avoids scaling problems in the future.
        text_lean_minhash = LeanMinHash(review_minhash)
        return pickle.dumps(text_lean_minhash)