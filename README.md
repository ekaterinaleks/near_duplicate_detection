# Near-Duplicate Detection
This repository contains case studies that implement solutions for near-duplicate detection in ![Static Python Badge](https://img.shields.io/badge/python-FBD343?style=plastic&logo=python&logoColor=234989bc).

## 🗃️ Directories:

### 📁LSH (Locality-sensitive hashing)
    Locality-sensitive hashing (LSH) allows us to preselect pairs that are likely to be nead-duplicates by hashing the items and dividing them into buckets of potentially similar texts. This way we can use these probabilistic pairs, keeping in mind that there would be some false positives and false negatives. Or, if we need to maximize precision and get rid of false positives, we can calculate Jaccard similarity only for these candidate pairs and still substantially reduce the computation cost compared to brute-force Jaccard similarity calculation for all possible pais.

The theory behind the algorithm: [Chapter 3 from "Mining Massive Datasets"](http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf).

Library used: [datasketch](https://github.com/ekzhu/datasketch).

Main functions are in the [lsh_helpers.py](./LSH\lsh_helpers.py) module.

To view the case study either have a look at it in .md format (converted from .ipynb) available in the repo or ![Static Badge](https://img.shields.io/badge/Open%20in%20Google%20Colab%20-%20%230e7fc0?style=plastic&logo=google%20colab&labelColor=grey).

Dataset: reviews from Goodreads 📚 taken from [Kaggle 2022 competition](https://www.kaggle.com/competitions/goodreads-books-reviews-290312) originally downloaded from UCSD Book Graph.

----