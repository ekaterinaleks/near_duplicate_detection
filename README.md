# Near-Duplicate Detection
Finding near-duplicate texts at scale with **MinHash + Locality-Sensitive Hashing**, applied to 1.38M Goodreads reviews in ![Static Python Badge](https://img.shields.io/badge/python-FBD343?style=plastic&logo=python&logoColor=234989bc).

## 🔎 Key Results
| Duplicated review texts | Exact matches | Exact + near-duplicates (LSH, checked with Jaccard) |
|---|---|---|
| Same user, different books | 13,788 | 16,298 |
| Different users, same book | 601 | 1,529 |

LSH surfaces reposts with typo fixes, reworded intros, edition-specific versions and blog-traffic spam that exact matching misses, while avoiding the ~950 billion pairwise comparisons of brute-force Jaccard similarity. Each LSH candidate pair (4,184 of them) is then checked with the exact Jaccard similarity (threshold 0.8), which confirms 84% of them.

## 🗃️ Directories:

### 📁LSH (Locality-sensitive hashing)
*Locality-sensitive hashing (LSH) allows us to preselect pairs that are likely to be near-duplicates by hashing the items and dividing them into buckets of potentially similar texts. This way we can use these probabilistic pairs, keeping in mind that there would be some false positives and false negatives. Or, if we need to maximize precision and get rid of false positives, we can calculate Jaccard similarity only for these candidate pairs and still substantially reduce the computation cost compared to brute-force Jaccard similarity calculation for all possible pairs.*

The theory behind the algorithm: [Chapter 3 from "Mining Massive Datasets"](http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf).

Library used: [datasketch](https://github.com/ekzhu/datasketch).

Main functions are in the [lsh_helpers.py](https://github.com/ekaterinaleks/near_duplicate_detection/blob/main/LSH/lsh_helpers.py) module.

To view the case study, open the [notebook](https://github.com/ekaterinaleks/near_duplicate_detection/blob/main/LSH/near_duplicate_detection_case.ipynb), its [.md version](https://github.com/ekaterinaleks/near_duplicate_detection/blob/main/LSH/near_duplicate_detection_case.md) or <a href="https://drive.google.com/file/d/1i09hAVCWH3UFpe-V_tptCSPDJV7Dh_QB/view?usp=sharing"> <img alt="Open in Google Colab" src="https://img.shields.io/badge/Open%20in%20Google%20Colab%20-%20%230e7fc0?style=plastic&logo=google%20colab&labelColor=grey"> </a>

Dataset: reviews from Goodreads 📚 taken from [Kaggle 2022 competition](https://www.kaggle.com/competitions/goodreads-books-reviews-290312), the original source is UCSD Book Graph.

## ▶️ How to Run
Requires Python 3.12+ and a [Kaggle API token](https://www.kaggle.com/docs/api). `pandarallel` needs Linux, macOS or WSL.
```bash
pip install -r LSH/requirements.txt
cd LSH
jupyter notebook near_duplicate_detection_case.ipynb
```
The notebook downloads the dataset via the Kaggle API (you must accept the competition rules first).

----