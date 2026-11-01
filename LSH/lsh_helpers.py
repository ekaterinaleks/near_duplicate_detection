import pickle
import re
from collections.abc import Hashable, Iterable, Mapping

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
        review_minhash.update_batch([shingle.encode("utf8")
                                     for shingle in review_shingles])

        # Lean MinHash: smaller memory footprint and faster deserialization,
        # negligible now but avoids scaling problems in the future.
        text_lean_minhash = LeanMinHash(review_minhash)
        return pickle.dumps(text_lean_minhash)

    def jaccard(self, shingles_a: set[str], shingles_b: set[str]) -> float:
        """Exact Jaccard similarity of two shingle sets (0.0 if both are empty)."""
        union = len(shingles_a | shingles_b)
        return len(shingles_a & shingles_b) / union if union else 0.0

    def score_candidates(self,
                         texts: Mapping[Hashable, str],
                         candidates: Mapping[Hashable, Iterable[Hashable]]
                         ) -> list[tuple[Hashable, Hashable, float]]:
        """Exact Jaccard similarity for each unique LSH candidate pair."""
        shingles: dict[Hashable, set[str]] = {}

        def get_shingles(_id: Hashable) -> set[str]:
            if _id not in shingles:
                shingles[_id] = self.to_shingles(texts[_id])
            return shingles[_id]

        scored = []
        checked: set[tuple[Hashable, Hashable]] = set()
        for _id, matches in candidates.items():
            # texts shorter than a shingle have no shingles and are matched exactly elsewhere
            if len(texts[_id]) < self.shingle_size:
                continue
            for match in matches:
                if match == _id or len(texts[match]) < self.shingle_size:
                    continue
                pair = (_id, match) if str(_id) < str(match) else (match, _id)
                if pair in checked:
                    continue
                checked.add(pair)
                scored.append((*pair, self.jaccard(get_shingles(_id), get_shingles(match))))
        return scored

    @staticmethod
    def connected_components(ids: Iterable[Hashable],
                             pairs: Iterable[tuple[Hashable, Hashable]]
                             ) -> dict[Hashable, int]:
        """Map each id to a cluster number; ids linked by pairs share a cluster (union-find)."""
        parent = {_id: _id for _id in ids}

        def find(_id: Hashable) -> Hashable:
            root = _id
            while parent[root] != root:
                root = parent[root]
            while parent[_id] != root:
                parent[_id], _id = root, parent[_id]
            return root

        for a, b in pairs:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        cluster_numbers: dict[Hashable, int] = {}
        return {_id: cluster_numbers.setdefault(find(_id), len(cluster_numbers))
                for _id in parent}