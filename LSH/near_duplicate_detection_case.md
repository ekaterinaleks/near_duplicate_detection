---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Near-duplicate Detection

### Why Look for Near-duplicates

When talking about near-duplicate detection, the first things that come to mind are probably anti-plagiarism applications and deduplicating web pages. In my case, I first encountered this task when working on a recommendation system for online postings with a high proportion of items reposted with slight changes.

In general, detecting near duplicates can be very beneficial for optimization purposes. Firstly, in the case of compute-expensive NLP preprocessing, we could preprocess only one item and reuse the result for the whole group. Secondly, it could be used for merging duplicated items in recommendation systems to get smaller —memory is often an issue for RecSys MLops— and less sparse interaction matrix —thus addressing the second common pain point of recommenders where getting a higher density percentage is often key for getting better results with collaborative filtering.

Apart from deduplication purposes as an optimization problem, it could be used for anomaly detection as well. Detecting users that leave similar reviews for different items or different users leaving almost exactly the same comment about specific items could be useful for moderation purposes or excluding some interactions from recommendation system training as unreliable or uninformative (in case of suspicion of review or comment use as advertisement channel or fake info campaigns).

```python
#libraries
import pandas as pd
import pickle
import itertools
import multiprocessing as mp
from pandarallel import pandarallel
import zipfile

#modules
import lsh_helpers as lsh

#miscellanea
pandarallel.initialize(nb_workers=mp.cpu_count(), progress_bar=False)
pd.options.mode.chained_assignment = None
pd.set_option('max_colwidth', None)
```
        INFO: Pandarallel will run on 8 workers.
        INFO: Pandarallel will use Memory file system to transfer data between the main process and workers.

### Get the Data
We will be using a dataset of reviews from Goodreads 📚 taken from [Kaggle 2022 competition](https://www.kaggle.com/competitions/goodreads-books-reviews-290312) originally downloaded from UCSD Book Graph. This dataset wasn't deduplicated as its later version currently available at [Goodreads Book Graph Datasets](https://mengtingwan.github.io/data/goodreads).

```python
#download using Kaggle API
! kaggle competitions download -c goodreads-books-reviews-290312
```
        Downloading goodreads-books-reviews-290312.zip to /home/ekaterina_axolotl/near-duplicate-detection
        100%|███████████████████████████████████████▉| 634M/635M [00:17<00:00, 41.7MB/s]
        100%|████████████████████████████████████████| 635M/635M [00:17<00:00, 37.9MB/s]

```python
#read into df only two files from the zip 
with zipfile.ZipFile('goodreads-books-reviews-290312.zip', 'r') as folder:
    with folder.open('goodreads_test.csv') as file:
        df_goodreads_test = pd.read_csv(file)
    with folder.open('goodreads_train.csv') as file:
        df_goodreads_train = pd.read_csv(file)
```

```python
#modify and combine the files for the near-duplicate detection task
df_goodreads_train.drop('rating', axis=1, inplace=True)
df_goodreads = pd.concat([df_goodreads_test, df_goodreads_train])
```

### Explore the Data & Check for Exact Duplicates

```python
print(f'The dataframe has {df_goodreads.shape[0]} rows and {df_goodreads.shape[1]} columns.')
```
 ╰⪼ The dataframe has 1378033 rows and 10 columns.

```python
print(f'Are there any complete duplicates with identical data in all columns? - {df_goodreads.duplicated().any()}')
```
╰⪼ Are there any complete duplicates with identical data in all columns? - False

```python
print(f'Are there any duplicated review ids? - {df_goodreads.duplicated(['review_id']).any()}')
```
╰⪼ Are there any duplicated review ids? - False

```python
print(f'How many duplicated review texts are there? - {df_goodreads.duplicated(['review_text'], keep=False).sum()}') 
```
╰⪼ How many duplicated review texts are there? - 20539

```python
print(f'How many duplicated review texts are there from the same user about different books? - {df_goodreads.duplicated(['review_text', 'user_id'], keep=False).sum()}')
```
╰⪼ How many duplicated review texts are there from the same user about different books? - 13788

```python
print(f'How many duplicated review texts are there from different users about the same book? - {df_goodreads.duplicated(['review_text', 'book_id'], keep=False).sum()}')
```
╰⪼ How many duplicated review texts are there from different users about the same book? - 601

```python
print(f'Are there any duplicated review texts from the same user about the same book? - {df_goodreads.duplicated(['review_text', 'book_id','user_id']).any()}')
```
╰⪼ Are there any duplicated review texts from the same user about the same book? - False

### Detect near-Duplicates: Case Study and Algorithm
**Case Study Dataset**

Although there are no complete duplicates for reviews submitted by the same user about the same book, [Goodreads' users do complain about these cases](https://www.goodreads.com/topic/show/22591024-duplicate-reviews). This problem likely stems from reposted reviews with slight wording, spelling and/or punctuation changes that can't be detected as easily as complete duplicates.

Also, we detected only 13788 completely duplicated review texts from the same user about different books. However, reading the forum, we can deduce that there are more of these with slight changes.

**Textual Similarity: Lexical vs Semantic & Computational Cost**

Given the massive amounts of textual data used in ML, finding similar texts is a fairly common task. However, the approach used varies greatly based on the use case and the amount of data to be processed.

While Transformers revolutionized the semantic similarity search, this approach is hardly usable when detecting near-duplicates: semantically close texts can be rather different on a lexical level, and the cost of calculating embeddings is too high for such a seemingly trivial task.

A more basic and straightforward approach like Jaccard similarity that could be used for lexical similarity also has its drawbacks: it doesn't take into account word order and is computationally unfeasible for large datasets as it needs to compare each pair one by one. For example, for our dataset of 1,378,033 reviews that would result in 949,486,785,528 pairs.

**Locality-Sensitive Hashing Technique**

Locality-sensitive hashing (LSH) allows us to preselect pairs that are likely to be nead-duplicates by hashing the items and dividing them into buckets of potentially similar texts. This way we can use these probabilistic pairs, keeping in mind that there would be some false positives and false negatives. Or, if we need to maximize precision and get rid of false positives, we can calculate Jaccard similarity only for these candidate pairs and still substantially reduce the computation cost compared to brute-force Jaccard similarity calculation for all possible pais.

***Algorithm Used***

The theory behind the algorithm: [Chapter 3 from "Mining Massive Datasets"](http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf).

Library used: [datasketch](https://github.com/ekzhu/datasketch).

Main functions are in the [lsh_helpers.py](./LSH\lsh_helpers.py).

***Steps***

- Pre-process text and cut into shingles (overlapping strings of fixed size).
- Minhash a set of shingles: create multiple hashes to get a single vector of each piece of text, a.k.a a signature.
- Locality-Sensitive Hashing for signatures: reduce the dimension of the hash vector, creating a single hash from a number (band) of nearby elements in the hash vector, a.k.a. a bandhash signature.
- Get candidate pairs: all pairs of signatures where elements match at least in one position.
- (If needed, calculate the actual Jaccard similarity).

***Some Considerations***

- *Choosing the k-shingle size is crucial*

  If the k is too small, then most sequences of k-characters will appear in most documents. K should be large enough so that the probability of any given shingle appearing in any given document is low. A good rule of thumb is to imagine that there are only 20 frequent characters (the case of English) and estimate the number of k-shingles as 20**k. Since the median length of reviews is 600 characters, the shingle size of k=7 (1,280,000,000 possible shingles) should work perfectly fine.

- *Use of LSH for Minhash Signatures*

    One general approach to LSH is to “hash” items several times, in such a way that similar items are more likely to be hashed to the same bucket than dissimilar items are. Then we consider any pair that hashed to the same bucket for any of the hashings to be a candidate pair for similarity.

- *The trade-off of choosing the number of permutations*

    A higher number of permutations gives higher accuracy and higher processing times. The default value for the *datasketch* library is 128. According to the benchmark in the documentation of MinHash LSH of *datasketch*, 150 seems to give the best precision and recall.

```python
#instantiate the LSH class with algorithm implementation
lsh_instance = lsh.LshHelper()
```

```python
#preprocess reviews and prepare df for taking minhash
df_goodreads['clean_review_text'] = df_goodreads['review_text'].parallel_apply(lsh_instance.preprocess)
df_goodreads.drop(['read_at', 'started_at', 'n_votes', 'n_comments'], axis=1, inplace=True)
df_goodreads.index = df_goodreads['review_id']
```

```python
#take minhash of all reviews
it = iter(df_goodreads.index)
chunk = list(itertools.islice(it, 16384))
while chunk:
    df_goodreads.loc[chunk, 'minhash'] = df_goodreads.loc[chunk, 'clean_review_text'].parallel_apply(lsh_instance.take_minhash)
    chunk = list(itertools.islice(it, 16384))
    df_goodreads.to_pickle('goodreads_minhash.pkl')
```

```python editable=true slideshow={"slide_type": ""}
#instantiate Locality Sensitive Hashing object
lsh_object = lsh_instance.instantiate_lsh_object()
```

```python
#populate the LSH object
for _index, minhash in df_goodreads['minhash'].items():
    lsh_object.insert(_index, pickle.loads(minhash), check_duplication=False)
```

```python
#check for near-duplicates
near_duplicates_dict = {}
for _index, minhash in df_goodreads['minhash'].items():
    dups = lsh_object.query(pickle.loads(minhash))
    near_duplicates_dict[_index] = dups
```

```python
#transform to get a df that can be joined to the original df
df_doubles = pd.json_normalize(near_duplicates_dict).transpose().rename(columns={0:'set_of_doubles'})
df_doubles['review_id'] = df_doubles.index
df_doubles['set_of_doubles'] = df_doubles['set_of_doubles'].apply(sorted).apply(tuple)
```

```python
#join the original df and duplicate groups df to get the final df ready for further analysis and deduplication depending on a particular need
df_doubles.reset_index(drop=True, inplace=True)
df_goodreads.reset_index(drop=True, inplace=True)
df_goodreads = df_goodreads.merge(df_doubles, how='inner', on='review_id')
```

### Let's Have a Look at Some Results


**Speaking about with near-identical reviews from the same user but about different books, above we already detected 13788 exact duplicates. Has this number changed after including near-duplicates?**

```python
#check for near-duplicates posted about the same book by different users
df_goodreads['user_group'] = df_goodreads.groupby(['user_id', 'set_of_doubles']).ngroup()
df_goodreads['user_number'] = df_goodreads.groupby('user_group')['user_group'].transform(lambda x: len(x))
```

```python
print(f'How many duplicated and near-duplicated review texts are there from the same user about different books? - {df_goodreads.query('user_number > 1').shape[0]}')
```
╰⪼ How many duplicated and near-duplicated review texts are there from the same user about different books? - 16180

This makes it about 3K additional reviews detected as near-duplicates in addition to complete duplicates.

**In some cases the initial wording makes it seem different (and impossible to spot without near-duplicate detection), but the core review is still the same and can be processed as one piece.** ⬇️

```python
df_goodreads.loc[[1243209, 1243210, 1243211], ['user_id', 'book_id', 'review_text']]
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>book_id</th>
      <th>review_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1243209</th>
      <td>05d313f6ef1e4bde738d03fcdf8f215d</td>
      <td>21457243</td>
      <td>I'm upset. A good kind of upset. This is why I don't read stuff that I know I'll love and the next book will not be released until the next year. \n Damn. \n Initial Thoughts. I have seen this series around for the past months. Well, not around but for a few blogs which expressed their fervent love for the books. I have been meaning to read this for some time because... Anne. Bishop. \n I've read her Black Jewels series and have loved it so much that I count it as one of my absolute favorites and would be recommended to every human being that asked for a book recommendation from me. Actually, I stopped at Dreams Made Flesh because Jaenelle died and Daemon had a child with Surreal which is out of character for both of them. I mean really Daemon!?! This is why I'm wary of Simon and Meg's romance because I'm still traumatized with what I've learned of the characters' fates in Tangled Webs. That was one book I regret I've read. You can tell that I'm not over it even years after I've read it. \n Ahem. \n But I've been curious and didn't even read the first book's synopsis before I just dived in and began reading. I knew I was going to love it just 17% in. \n Full review at Whatever You Can Still Betray.</td>
    </tr>
    <tr>
      <th>1243210</th>
      <td>05d313f6ef1e4bde738d03fcdf8f215d</td>
      <td>17563080</td>
      <td>By now, would the human pack start having a betting pool about when Simon and Meg would get together? \n Initial Thoughts. I have seen this series around for the past months. Well, not around but for a few blogs which expressed their fervent love for the books. I have been meaning to read this for some time because... Anne. Bishop. \n I've read her Black Jewels series and have loved it so much that I count it as one of my absolute favorites and would be recommended to every human being that asked for a book recommendation from me. Actually, I stopped at Dreams Made Flesh because Jaenelle died and Daemon had a child with Surreal which is out of character for both of them. I mean really Daemon!?! This is why I'm wary of Simon and Meg's romance because I'm still traumatized with what I've learned of the characters' fates in Tangled Webs. That was one book I regret I've read. You can tell that I'm not over it even years after I've read it. \n Ahem. \n But I've been curious and didn't even read the first book's synopsis before I just dived in and began reading. I knew I was going to love it just 17% in. \n Full review at Whatever You Can Still Betray.</td>
    </tr>
    <tr>
      <th>1243211</th>
      <td>05d313f6ef1e4bde738d03fcdf8f215d</td>
      <td>15711341</td>
      <td>*howls* \n Initial Thoughts. I have seen this series around for the past months. Well, not around but for a few blogs which expressed their fervent love for the books. I have been meaning to read this for some time because... Anne. Bishop. \n I've read her Black Jewels series and have loved it so much that I count it as one of my absolute favorites and would be recommended to every human being that asked for a book recommendation from me. Actually, I stopped at Dreams Made Flesh because Jaenelle died and Daemon had a child with Surreal which is out of character for both of them. I mean really Daemon!?! This is why I'm wary of Simon and Meg's romance because I'm still traumatized with what I've learned of the characters' fates in Tangled Webs. That was one book I regret I've read. You can tell that I'm not over it even years after I've read it. \n Ahem. \n But I've been curious and didn't even read the first book's synopsis before I just dived in and began reading. I knew I was going to love it just 17% in. \n Full review at Whatever You Can Still Betray.</td>
    </tr>
  </tbody>
</table>
</div>

**Some reviews are just standard comments without any relevant information added by a specific user to their reviews** ⬇️

```python
df_goodreads.loc[[182963, 182965, 182966], ['user_id', 'book_id', 'review_text']]
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>book_id</th>
      <th>review_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>182963</th>
      <td>08cd4cfc474a90939847203da1ea2da3</td>
      <td>30635342</td>
      <td>Originally posted on I Smell Sheep Review to come soon. \n * A copy of this book was provided by the publisher via NetGalley for the purpose of an honest review. All conclusions are my own responsibility and I was not compensated for this review.</td>
    </tr>
    <tr>
      <th>182965</th>
      <td>08cd4cfc474a90939847203da1ea2da3</td>
      <td>21858682</td>
      <td>Originally posted on I Smell Sheep \n * A copy of this book was and provided by the publisher for the purpose of an honest review. All conclusions are my own responsibility and I was not compensated for this review.</td>
    </tr>
    <tr>
      <th>182966</th>
      <td>08cd4cfc474a90939847203da1ea2da3</td>
      <td>29875893</td>
      <td>Originally posted on I Smell Sheep \n * A copy of this book was and provided by the publisher via NetGalley for the purpose of an honest review. All conclusions are my own responsibility and I was not compensated for this review.</td>
    </tr>
  </tbody>
</table>
</div>

**Others are just people trying to send traffic to their blogs and don't contribute any meaningful information. However, the text inself varies, making it difficult to detect.** ⬇️

```python
df_goodreads.loc[[470436, 470437, 470444, 470450, 470483, 470484, 470485, 470486, 470487, 470490, 470495, 470509, 470510], ['user_id', 'book_id', 'review_text']]
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>book_id</th>
      <th>review_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>470436</th>
      <td>0d69df93e65da6f95232f075388dfe34</td>
      <td>7825557</td>
      <td>Read my review at http://bookwormblurbs.booklikes.com/p...</td>
    </tr>
    <tr>
      <th>470437</th>
      <td>0d69df93e65da6f95232f075388dfe34</td>
      <td>7686667</td>
      <td>3. 5 out of 5 stars. Read my review at http://bookwormblurbs.booklikes.com/p...</td>
    </tr>
    <tr>
      <th>470444</th>
      <td>0d69df93e65da6f95232f075388dfe34</td>
      <td>44184</td>
      <td>Read my review at http://bookwormblurbs.booklikes.com/p...</td>
    </tr>
    <tr>
      <th>470450</th>
      <td>0d69df93e65da6f95232f075388dfe34</td>
      <td>10766509</td>
      <td>Abandoned. Read my review at http://bookwormblurbs.booklikes.com/p...</td>
    </tr>
    <tr>
      <th>470483</th>
      <td>0d69df93e65da6f95232f075388dfe34</td>
      <td>22628</td>
      <td>Read my review at http://bookwormblurbs.booklikes.com/p...</td>
    </tr>
    <tr>
      <th>470484</th>
      <td>0d69df93e65da6f95232f075388dfe34</td>
      <td>231804</td>
      <td>3.5 stars. Read my review at http://bookwormblurbs.booklikes.com/p...</td>
    </tr>
    <tr>
      <th>470485</th>
      <td>0d69df93e65da6f95232f075388dfe34</td>
      <td>99561</td>
      <td>Read my review at http://bookwormblurbs.booklikes.com/p...</td>
    </tr>
    <tr>
      <th>470486</th>
      <td>0d69df93e65da6f95232f075388dfe34</td>
      <td>5107</td>
      <td>Read my review at http://bookwormblurbs.booklikes.com/p...</td>
    </tr>
    <tr>
      <th>470487</th>
      <td>0d69df93e65da6f95232f075388dfe34</td>
      <td>15753740</td>
      <td>Read my review at http://bookwormblurbs.booklikes.com/p...</td>
    </tr>
    <tr>
      <th>470490</th>
      <td>0d69df93e65da6f95232f075388dfe34</td>
      <td>15818107</td>
      <td>Read my review at http://bookwormblurbs.booklikes.com/p...</td>
    </tr>
    <tr>
      <th>470495</th>
      <td>0d69df93e65da6f95232f075388dfe34</td>
      <td>9375</td>
      <td>Read my review at http://bookwormblurbs.booklikes.com/p...</td>
    </tr>
    <tr>
      <th>470509</th>
      <td>0d69df93e65da6f95232f075388dfe34</td>
      <td>3636</td>
      <td>Read my review at http://bookwormblurbs.booklikes.com/p...</td>
    </tr>
    <tr>
      <th>470510</th>
      <td>0d69df93e65da6f95232f075388dfe34</td>
      <td>227443</td>
      <td>Read my review at http://bookwormblurbs.booklikes.com/p...</td>
    </tr>
  </tbody>
</table>
</div>

**The following two sets reviews are original and later versions of the same piece of writing, most probably posted for different editions. The introductions in both cases are a bit different, making it impossible to detect them as complete duplicates** ⬇️

```python
df_goodreads.loc[[209316, 209365], ['user_id', 'book_id', 'review_text']]
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>book_id</th>
      <th>review_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>209316</th>
      <td>06ca020a59e70d460b5644588fdb2c20</td>
      <td>22819449</td>
      <td>4.5* :) \n *Re-read but original review* \n Upon receiving this book, I was so incredibly excited to start it! There is so much positivity floating around this book and I really could not wait to start. I felt a little anticipatory and really hoped that I wouldn't be one of the awkward ones that just 'didn't get it'. The beginning of the book had me conflicted. I found myself wondering when things were going to get going and I started to worry that I wasn't going to like it. This soon changed! \n This book felt like two things rolled into one. At first we were with Alina on her journey to cross The Shadow Fold - the dark and dangerous expanse which secludes the Ravka. Towards the end it felt like a massive battle of good vs. evil and I absolutely loved it. It's a book similar to the likes of Throne of Glass by Sarah J. Maas and I love that it reminded me of that. It's a book full of adventure and I adore fantasy books which follow characters journeys to various places. This book definitely had all of the characteristics of the genre that I love; the journey, the enticing villain, romance, a strong heroine and plenty of action. \n "The thought filled me with grief, grief for the dreams we'd shared, for the love I'd felt, for the hopeful girl I would never be again." \n The transformation of Alina was fascinating. She went from being an orphan with no place in the world to the Sun Summoner, one of a kind and completely kick ass. Alina was never a push over, and like most young girls, susceptible to the charms of The Darkling. The thing I loved about this was that it wasn't just Alina that was susceptible. Every female Grisha felt the pull towards him, as if it was a built in mechanism that no one can help. I loved watching her develop her control over her power and seeing her take on the responsibility of being the one that can finally attempt to rise the darkness bestowed by the Shadow Fold. \n I loved Mal, Alina's best friend and fellow orphan, a top class tracker and a really amazing character. But I also loved The Darkling. The seductive, all powerful villain. It isn't often that I feel conflicted about a villain but am really looking forward to seeing how it all unfolds in the next book of the series. The relationships developed are wonderful and the romance within the book isn't overpowering. It doesn't command centre stage and allows Alina's development and her task to take precedence. Exactly how I like my books! \n I highly recommend this book, it certainly lives up to the hype created by the massive Grisha fandom. If you haven't read this book yet, what are you waiting for? I am so disappointed that it took me so long to get hooked on this series!</td>
    </tr>
    <tr>
      <th>209365</th>
      <td>06ca020a59e70d460b5644588fdb2c20</td>
      <td>18001355</td>
      <td>*Received in exchange for an honest review* \n *Thank you to Gollancz for providing a copy* \n Upon receiving this book, I was so incredibly excited to start it! There is so much positivity floating around this book and I really could not wait to start. I felt a little anticipatory and really hoped that I wouldn't be one of the awkward ones that just 'didn't get it'. The beginning of the book had me conflicted. I found myself wondering when things were going to get going and I started to worry that I wasn't going to like it. This soon changed! \n This book felt like two things rolled into one. At first we were with Alina on her journey to cross The Shadow Fold - the dark and dangerous expanse which secludes the Ravka. Towards the end it felt like a massive battle of good vs. evil and I absolutely loved it. It's a book similar to the likes of Throne of Glass by Sarah J. Maas and I love that it reminded me of that. It's a book full of adventure and I adore fantasy books which follow characters journeys to various places. This book definitely had all of the characteristics of the genre that I love; the journey, the enticing villain, romance, a strong heroine and plenty of action. \n "The thought filled me with grief, grief for the dreams we'd shared, for the love I'd felt, for the hopeful girl I would never be again." \n The transformation of Alina was fascinating. She went from being an orphan with no place in the world to the Sun Summoner, one of a kind and completely kick ass. Alina was never a push over, and like most young girls, susceptible to the charms of The Darkling. The thing I loved about this was that it wasn't just Alina that was susceptible. Every female Grisha felt the pull towards him, as if it was a built in mechanism that no one can help. I loved watching her develop her control over her power and seeing her take on the responsibility of being the one that can finally attempt to rise the darkness bestowed by the Shadow Fold. \n I loved Mal, Alina's best friend and fellow orphan, a top class tracker and a really amazing character. But I also loved The Darkling. The seductive, all powerful villain. It isn't often that I feel conflicted about a villain but am really looking forward to seeing how it all unfolds in the next book of the series. The relationships developed are wonderful and the romance within the book isn't overpowering. It doesn't command centre stage and allows Alina's development and her task to take precedence. Exactly how I like my books! \n I highly recommend this book, it certainly lives up to the hype created by the massive Grisha fandom. If you haven't read this book yet, what are you waiting for? I am so disappointed that it took me so long to get hooked on this series!</td>
    </tr>
  </tbody>
</table>
</div>

```python
df_goodreads.loc[[1215804, 1215808], ['user_id', 'book_id', 'review_text']]
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>book_id</th>
      <th>review_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1215804</th>
      <td>017619f3daea3c2382652fc71cc3b3d7</td>
      <td>11519267</td>
      <td>This review first appeared on I Heart Romance &amp; YA \n "A surprising read!" \n I really didn't have any expectations when I started reading this book. In fact, I thought it was going to be another Divergent/Hunger Games dystopia book and thought it would be too cookie-cutter for my taste. BUT I was really surprised that I enjoyed it! Sure, there were a few flaws that made me rate it 4 stars but..... Day! \n Legend is really full of kickass, badass, fighter FEMALES! In this book alone, 3 were prominent fighters. Not to mention that the heroine, June IS a freaking child fighting prodigy! At this point, Legend diverted from the usual cookie cutter dystopia - girl not fighter ready but then turns into an awesome fighter - but she was already someone who WAS brought up to be a soldier - a prodigy, if you will. \n Legend is about a dystopian Los Angeles (another surreal thing because I live in this area) in the future Republic of America. In this Republic, there is civil war between the states and the United States is a thing of the past. June grew up in a rich neighborhood, raised by her brother after their parents' death (father was a scientist for the Republic) and is a prodigy in the Trials. She is the only reported person to even ace her trials and at 15, already about to graduate her senior year at Drake University. Our hero Day, grew up in the slum - the Lake District of Los Angeles and is the number 1 most wanted criminal in the city. Believed to be dead by his family, he has survived on the streets and be like a modern day Robin Hood. Both weren't supposed to meet UNTIL someone killed June's brother. And there is a plague. Well, there is always a plague in these YA dystopians. \n I really like this story a lot and as I've said before, I was really surprised. I LOVED that June was badass and I loved her intelligence. It might be a tad bit unbelievable for a 15 year old but it wasn't TOO unbelievable that I did not enjoy the story. PLUS she had a dog, Ali that she DID NOT forget when she escaped Los Angeles. Well, she had to leave Ali but she it was mentioned in the book that she planned to hide him close (view spoiler)[when she left the city with Day. (hide spoiler)] \n I also liked Day. I loved that HE did not head a rebellion even though what he was doing did not really have any purpose other than helping out his family (in secret). Oh and he had an imperfection. AND he was of Asian decent! Yay for diversity. Okay, so he is HALF Asian but still, it is still refreshing to read about diversity in YA. \n There was a semi insta-love thing going on which I didn't really care for but it wasn't really the main focus of the story - which was a delightful surprise! \n Legend is a fast paced story and while listening to it on audio, I was surprised that it was already over! It is a short YA dystopian book at less than 30 chapters but it was jam packed full of action and drama. There was a little twist in the end that I was expecting but it did not affect my enjoyment to the book as a whole. \n I am definitely recommending this book, if you haven't already read it. I am currently listening to the second book in the series: Prodigy. \n THOUGHTS ON THE AUDIO \n Narrator: Steven Kaplan &amp; Mariel Stern \n Two awesome narrators made this book an exciting listen! Legend is told in Day and June's alternating POV with the two narrators alternating. I love that I get to listen to a different voice every other chapter and both really performed well and made the characters feel real to me. Legend is an awesome read and it is even more amazing when listened to. \n Read the full review on I Heart Romance &amp; YA</td>
    </tr>
    <tr>
      <th>1215808</th>
      <td>017619f3daea3c2382652fc71cc3b3d7</td>
      <td>9275658</td>
      <td>"A surprising read!" \n I really didn't have any expectations when I started reading this book. In fact, I thought it was going to be another Divergent/Hunger Games dystopia book and thought it would be too cookie-cutter for my taste. BUT I was really surprised that I enjoyed it! Sure, there were a few flaws that made me rate it 4 stars but..... Day! \n Legend is really full of kickass, badass, fighter FEMALES! In this book alone, 3 were prominent fighters. Not to mention that the heroine, June IS a freaking child fighting prodigy! At this point, Legend diverted from the usual cookie cutter dystopia - girl not fighter ready but then turns into an awesome fighter - but she was already someone who WAS brought up to be a soldier - a prodigy, if you will. \n Legend is about a dystopian Los Angeles (another surreal thing because I live in this area) in the future Republic of America. In this Republic, there is civil war between the states and the United States is a thing of the past. June grew up in a rich neighborhood, raised by her brother after their parents' death (father was a scientist for the Republic) and is a prodigy in the Trials. She is the only reported person to even ace her trials and at 15, already about to graduate her senior year at Drake University. Our hero Day, grew up in the slum - the Lake District of Los Angeles and is the number 1 most wanted criminal in the city. Believed to be dead by his family, he has survived on the streets and be like a modern day Robin Hood. Both weren't supposed to meet UNTIL someone killed June's brother. And there is a plague. Well, there is always a plague in these YA dystopians. \n I really like this story a lot and as I've said before, I was really surprised. I LOVED that June was badass and I loved her intelligence. It might be a tad bit unbelievable for a 15 year old but it wasn't TOO unbelievable that I did not enjoy the story. PLUS she had a dog, Ali that she DID NOT forget when she escaped Los Angeles. Well, she had to leave Ali but she it was mentioned in the book that she planned to hide him close (view spoiler)[when she left the city with Day. (hide spoiler)] \n I also liked Day. I loved that HE did not head a rebellion even though what he was doing did not really have any purpose other than helping out his family (in secret). Oh and he had an imperfection. AND he was of Asian decent! Yay for diversity. Okay, so he is HALF Asian but still, it is still refreshing to read about diversity in YA. \n There was a semi insta-love thing going on which I didn't really care for but it wasn't really the main focus of the story - which was a delightful surprise! \n Legend is a fast paced story and while listening to it on audio, I was surprised that it was already over! It is a short YA dystopian book at less than 30 chapters but it was jam packed full of action and drama. There was a little twist in the end that I was expecting but it did not affect my enjoyment to the book as a whole. \n I am definitely recommending this book, if you haven't already read it. I am currently listening to the second book in the series: Prodigy. \n THOUGHTS ON THE AUDIO \n Narrator: Steven Kaplan &amp; Mariel Stern \n Two awesome narrators made this book an exciting listen! Legend is told in Day and June's alternating POV with the two narrators alternating. I love that I get to listen to a different voice every other chapter and both really performed well and made the characters feel real to me. Legend is an awesome read and it is even more amazing when listened to.</td>
    </tr>
  </tbody>
</table>
</div>

**Speaking about with near-identical reviews from different users but about the same book, above we already detected 601 exact duplicates. Has this changed after including near-duplicates?**

```python
#check for near-duplicates posted about the same book by different users
df_goodreads['book_group'] = df_goodreads.groupby(['book_id', 'set_of_doubles']).ngroup()
df_goodreads['book_number'] = df_goodreads.groupby('book_group')['book_group'].transform(lambda x: len(x))
print(f'How many duplicated and near-duplicated review texts are there from different users about the same book? - {df_goodreads.query('book_number > 1').shape[0]}')
```
╰⪼ How many near-duplicated review texts are there from different users about the same book? - 1347

This makes it 746 additional near-duplicates.

```python
#let's check some longer reviews that are unlikely tobe a typical short "review to come" placeholder
df_goodreads['review_length'] = df_goodreads['review_text'].apply(len)
df_goodreads.query('(book_number > 1) & (review_length > 200)').sort_values('book_group')[['user_id', 'book_id', 'review_text']]
```

**This one is clearly written by the same person and isn't an exact duplicate only due to some typos:**
⬇️

```python
df_goodreads.loc[[234437,1143023], ['user_id', 'book_id', 'review_text']]
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>book_id</th>
      <th>review_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>234437</th>
      <td>92ec07eb6116e956d36d28a6a63a1eea</td>
      <td>2175</td>
      <td>This novel had been on my TBR list for years before I finally got around to reading it while on one of my rare "classics" kicks. \n Unfortunately, for me, it was hardly worth the wait. The tale was far too introspective and dragging for my taste, and ultimately depressing. \n Although I confess a penchant for happy endings, an emotion filled tragedy can be satisfying in its own way. (Wuthering Heights comes immediately to mind). \n Unfortunately even with its tragic ending, I was unmoved, finding that I could neither like, nor pity any of the characters. \n Madame Bovary left me craving a good dose of Emily Bront.</td>
    </tr>
    <tr>
      <th>1143023</th>
      <td>6730d0d062328bbfb874ad9c5ce02f33</td>
      <td>2175</td>
      <td>This novel had been on my TBR list for years before I finally got around to reading it while on one of my rare "classics" kicks. \n Unfortunately, for me, it was hardly worth the wait. The tale was far too introspective and dragging for my taste, and ultimately depressing. \n Although I confess a penchant for happy endings, an emotion filled tragedy can be satisfying in its own way. (Wuthering Heights comes immediately to mind). \n Unfortunately even with its tragic ending, I was unmoved, finding that I could neither like, nor pity any of the characters. \n Madame Bovary left me craving a good dose of Emily Bronte.</td>
    </tr>
  </tbody>
</table>
</div>

**The following two reviews might seem different at first glance, but after closer inspection turn out to be two versions of the same review as revealed by large chunks with the exact same wording and author capitalization for emphasis.** ⬇️

```python
df_goodreads.loc[[932841,853033], ['user_id', 'book_id', 'review_text']]
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>book_id</th>
      <th>review_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>932841</th>
      <td>99386ea50984a4ae3a9af9f4ffabb193</td>
      <td>73100</td>
      <td>Once upon a time, I read a bunch of SEP (I was going through a phase). And this is more of an attempt to ward off such a phase, at least not to repeat the same mistakes! \n Here's Krista's review in case I ever think of picking up SEP EVER AGAIN: \n At first, I thought I would enjoy this book. I should have known better, because this is Susan Elizabeth Phillips, whose work ALWAYS annoys me because of the sexist tendencies all of her women seem to exhibit. This one was no different. \n When I was first introduced to Molly, I found her likable, funny, kind of weird, but not in a bad way--and then she goes and does something unthinkable and after that point, I just couldn't like her at all. So, Kevin's boss had told him to take some time off and stay in this cabin for awhile. Kevin and Molly had met a few times before, but didn't really know each other. So Molly shows up at this cabin, uninvited, then refuses to leave once she finds Kevin there, because she's too petty. THEN, when Kevin is asleep in his bed, she snoops through his stuff, finds a condom in his shaving kit, goes into his room, climbs into his bed and starts touching him while he is asleep. Kevin, still mostly asleep, and thinking the whole thing a dream in which he's making love with his ex-girlfriend, starts to respond to her touches. She then proceeds to rape him--and it is rape, because he would not have consented to have sex with her if he actually were aware of what was happening (he establishes this). So after she rapes him, he wakes up, freaks out, and announces that the condom had broken because it was about a million years old. \n After this point, Molly had ruined herself in my mind. I cannot respect a woman who does something so devious and pathetic. \n But it gets worse. \n Molly gets pregnant with Kevin's baby. When Kevin later calls to check up on her, she blatantly lies to him and says she is not pregnant. Yeah. First she rapes him, then she plans on stealing his baby. Okay. And in case that weren't enough, her sister and brother-in-law, Phoebe and Dan Calebow from some other book in the series I haven't read because of its ridiculous cover, show up, figure out that it was Kevin who got her pregnant. So then Dan goes and finds Kevin and punches him and accuses him of being the scum of the earth and a seducer of innocents and a low-life jerk trying to run away from his duty blah blah blah. Kevin, being a nice guy, lets this happen, not revealing Molly's psychotics. Molly, being too ashamed and selfish, lets her brother-in-law and sister treat Kevin like crap and insult him while she holds on to her secret about raping him. \n Kevin agrees to marry her, like I said, because he's a good guy. Molly has a miscarriage, blames Kevin, insults him, and proceeds to become depressed. Kevin, being a freaking good guy, feels sorry for her so he takes her on a trip to this campground that he owns to see if he can get her to perk up again. Molly is awful to him the whole time. Once there, she refuses to help with anything. Eventually she comes around, after Kevin practically begs her, and then she's really proud of herself for the monumental task of cooking breakfast. She deserves a freakin' medal. \n What follows is a series of Molly's antics. She tricks Kevin at every turn, pretending to drown, purposely tipping over a canoe to get his attention, forcing a cat up into a tree and then making him go after the cat. She's a very mature woman, obviously. \n Then the freaking Calebows show up to tell Kevin he's a jerk and to demand to know what his intentions are. Then Dan accuses Kevin of using Molly for a quick roll in the hay, because, apparently, football players (ahem, Dan, you freakin' hypocrite) cannot actually love a woman because they are too shallow and don't care about women or love. Of course, Molly believes this load of garbage, so she gets all pissed and claims she's using Kevin because he's available and she deserves to be "naughty" after so many years of being "good." Nice. \n Then Molly goes back to her schemes. When Kevin tries to sell HIS campground, Molly goes to the buyer and tells him a bunch of lies about the place and that Kevin is mentally insane and needs help. The buyer believes her and leaves. Controlling much, Molly? How about living your own life? \n After breaking up for a period, Kevin realizes he can't live without a psychotic, controlling b*atch running his life, so he goes to Molly's sister to ask for HER blessing to marry Molly--as if he needs it. Phoebe refuses, accuses Kevin of using Molly to guarantee his continued career with his football team, owned by Phoebe. She then says if he so much as goes near Molly, he will be off the team. Are. You. Freaking. Kidding. Me??? So Kevin decides Molly is more important and goes to marry her anyway. \n Molly tells him he's an idiot and that men should give up everything in their lives for the women they love, but women should give up nothing. So, what do we learn? That men should give up their careers and everything else they like to please a woman, but a woman can do whatever she pleases. We learn that Kevin is willing to give up everything, including his career as a professional football player, which is basically his whole life, in order to be with Molly, the woman who raped him and controlled his life and gave up nothing for him. \n This is not a book. It's a sexist peice of poo. \n Yes. Women can be sexist, too. SEP is proof of this. \n Why the 2 stars? I liked the H. The story was terrible, but the H wasn't.</td>
    </tr>
    <tr>
      <th>853033</th>
      <td>bed0a93b9c63b3619929f2557cd9bea1</td>
      <td>73100</td>
      <td>** spoiler alert ** \n Okay, so this is going to be a rant, not a review. \n At first, I thought I would enjoy this book. I should have known better, because this is Susan Elizabeth Phillips, whose work ALWAYS annoys me because of the sexist tendencies all of her women seem to exhibit. This one was no different. \n When I was first introduced to Molly, I found her likable, funny, kind of weird, but not in a bad way--and then she goes and does something unthinkable and after that point, I just couldn't like her at all. So, Kevin's boss had told him to take some time off and stay in this cabin for awhile. Kevin and Molly had met a few times before, but didn't really know each other. So Molly shows up at this cabin, uninvited, then refuses to leave once she finds Kevin there, because she's too petty. THEN, when Kevin is asleep in his bed, she snoops through his stuff, finds a condom in his shaving kit, goes into his room, climbs into his bed and starts touching him while he is asleep. Kevin, still mostly asleep, and thinking the whole thing a dream in which he's making love with his ex-girlfriend, starts to respond to her touches. She then proceeds to rape him--and it is rape, because he would not have consented to have sex with her if he actually were aware of what was happening (he establishes this). So after she rapes him, he wakes up, freaks out, and announces that the condom had broken because it was about a million years old. \n After this point, Molly had ruined herself in my mind. I cannot respect a woman who does something so devious and pathetic. \n But it gets worse. \n Molly gets pregnant with Kevin's baby. When Kevin later calls to check up on her, she blatantly lies to him and says she is not pregnant. Yeah. First she rapes him, then she plans on stealing his baby. Okay. And in case that weren't enough, her sister and brother-in-law, Phoebe and Dan Calebow from some other book in the series I haven't read because of its ridiculous cover, show up, figure out that it was Kevin who got her pregnant. So then Dan goes and finds Kevin and punches him and accuses him of being the scum of the earth and a seducer of innocents and a low-life jerk trying to run away from his duty blah blah blah. Kevin, being a nice guy, lets this happen, not revealing Molly's psychotics. Molly, being too ashamed and selfish, lets her brother-in-law and sister treat Kevin like crap and insult him while she holds on to her secret about raping him. \n Kevin agrees to marry her, like I said, because he's a good guy. Molly has a miscarriage, blames Kevin, insults him, and proceeds to become depressed. Kevin, being a freaking good guy, feels sorry for her so he takes her on a trip to this campground that he owns to see if he can get her to perk up again. Molly is awful to him the whole time. Once there, she refuses to help with anything. Eventually she comes around, after Kevin practically begs her, and then she's really proud of herself for the monumental task of cooking breakfast. She deserves a freakin' medal. \n What follows is a series of Molly's antics. She tricks Kevin at every turn, pretending to drown, purposely tipping over a canoe to get his attention, forcing a cat up into a tree and then making him go after the cat. She's a very mature woman, obviously. \n Then the freaking Calebows show up to tell Kevin he's a jerk and to demand to know what his intentions are. Then Dan accuses Kevin of using Molly for a quick roll in the hay, because, apparently, football players (ahem, Dan, you freakin' hypocrite) cannot actually love a woman because they are too shallow and don't care about women or love. Of course, Molly believes this load of garbage, so she gets all pissed and claims she's using Kevin because he's available and she deserves to be "naughty" after so many years of being "good." Nice. \n Then Molly goes back to her schemes. When Kevin tries to sell HIS campground, Molly goes to the buyer and tells him a bunch of lies about the place and that Kevin is mentally insane and needs help. The buyer believes her and leaves. Controlling much, Molly? How about living your own life? \n After breaking up for a period, Kevin realizes he can't live without a psychotic, controlling b*atch running his life, so he goes to Molly's sister to ask for HER blessing to marry Molly--as if he needs it. Phoebe refuses, accuses Kevin of using Molly to guarantee his continued career with his football team, owned by Phoebe. She then says if he so much as goes near Molly, he will be \n off the team. \n Are. You. Freaking. Kidding. Me??? So Kevin decides Molly is more important and goes to marry her anyway. \n Molly tells him he's an idiot and that men should give up everything in their lives for the women they love, but women should give up nothing. So, what do we learn? That men should give up their careers and everything else they like to please a woman, but a woman can do whatever she pleases. We learn that Kevin is willing to give up everything, including his career as a professional football player, which is basically his whole life, in order to be with Molly, the woman who raped him and controlled his life and gave up nothing for him. \n This is not a book. It's a sexist peice of poo. \n Yes. Women can be sexist, too. SEP is proof of this.</td>
    </tr>
  </tbody>
</table>
</div>

**The two reviews below are clearly by the same person (with different user_ids), but have some small differences (punctuation, small additions like "Review by Toni", etc.)** ⬇️

```python
df_goodreads.loc[[732165,893342], ['user_id', 'book_id', 'review_text']]
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>book_id</th>
      <th>review_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>732165</th>
      <td>f50ff6d1d6395aecfe7ae2bc3e51e8bb</td>
      <td>18689657</td>
      <td>Stock up on tissues, wine and turn your air condition on for this 6 Star read!! \n The end of The Thrill of It, I shouted NOOO!! So loud my neighbors rushed over about to bang down my door in fear of something seriously wrong occurred and boy did it Lauren Blakely and her evil genius mind. The ending left me heartbrokenly scared for the course ahead for Harley and Trey. After all the turmoil these two experienced throughout The Thrill of It and to have found faith and rediscovery in themselves through one another was uplifting to read. Then Ms. Blakely has your tires screeching and the direction of these two characters' story becomes fuzzy. \n I was beyond anxious to get ahold of Every Second With You; then once I got it all I wanted to do was call out of work. Nothing was going to come in between me and Trey Westin. This man stole my heart in TTOI a man so broken, flawed, yet intelligent and passionate. However, the new obstacles Ms. Blakely presents in Every Second With You will have you on the edge of your seat, biting your freshly manicured nails not giving a flying fudge because you are so nervous for Trey. At the same time you feel and ache for Harley. Talk about a girl not being able to catch a break :/ \n Although constantly tested, Harley demonstrates early on in ESWY how far and strong she has become. My admiration and respect for her continued to grow in this story. On the other hand, it was Trey that struggles when triggers off set his vulnerabilities \n The drama Ms. Blakely incorporates in this story makes the events of TTOI look like highschool play. The storyline is brilliant. Both characters must work hard to accept and or redefine their perception of themselves if they want to grow as a couple. The support one another show each other will make you tear up because it is true love at its finest (romcom movies take notes from Lauren Blakely!) \n I thought my emotions were all over the place in the first book but wow was I proven wrong. As I sit here and watch luging on the Olypmics, the head rush I'm sure these athletes feel following one helluva fast swerving and whipflash like turns is how I felt after reading this book and like that athlete the adrenaline blissful after effects were so worth the road bumps and anxiety spikes. In addition, holy grail of sex scenes. Lauren Blakely I do not know how you do it; every book you write the scenes get hotter. I'm talking don't bother wearing panties, don't read in public, boy/girlfriends, husbands, wives better stock on their thank you cards because woman your intimate scenes leave me pantingly speechless! The intimate actions and Trey Westin's dirty (No soap could ever clean this man's mouth, nor do you want it too haha) mouth are definitely a treat however, there is more to Trey than being sexifyingly delicious; he is also sensually compassionate and loving towards Harley. \n A lot of sequels will start off strong and fizzle out or they hold your interest but you can't stop comparing it to the first book, have no fear neither of those things occur with Every Second With You. This sequel is by far one of my top 3 favorite sequels. The drama is new not repetitive, the sex is hotter, the romance is heartwarming and the aspect that I applaud Ms. Blakely on the most is how realistic these individuals develop their sense of self-identity and as a couple! \n Be Warned--Lauren Blakely does not take it easy on you she holds the controller on your emotions until the very end. Nothing about Harley and Trey's ride is smooth but dammit give me Dramamine I will take my chances with motion sickness every time in order to get to that final destination that is the end of Every Second With You. Amazing, mindblown, heart swelling, 300% satisfaction Lauren Blakely your writing never ceases to make me more and more of a crazed fan for you \n *So thankful received ARC in return for honest review*</td>
    </tr>
    <tr>
      <th>893342</th>
      <td>88e4407451355899b43ed6f6c3014623</td>
      <td>18689657</td>
      <td>Stock up on tissues, wine and turn your air conditioner on for this 6 Star read!! \n At the end of The Thrill of It, I shouted NOOO!! So loud, my neighbors rushed over about to bang down my door in fear of something seriously wrong having occurred and boy did it..... Lauren Blakely and her evil genius mind. The ending left me heartbrokenly scared for the course ahead for Harley and Trey. After all the turmoil these two experienced throughout The Thrill of It and to have found faith and rediscovery in themselves through one another was uplifting to read. Then Ms. Blakely has your tires screeching and the direction of these two characters' story becomes fuzzy. \n I was beyond anxious to get a hold of Every Second With You; then once I got it, all I wanted to do was call out of work. Nothing was going to come in between me and Trey Westin. This man stole my heart in TTOI a man so broken, flawed, yet intelligent and passionate. However, the new obstacles Ms. Blakely presents in Every Second With You will have you on the edge of your seat, biting your freshly manicured nails not giving a flying fudge because you are so nervous for Trey. At the same time you feel and ache for Harley. Talk about a girl not being able to catch a break :/ \n Although constantly tested, Harley demonstrates early on in ESWY how far and strong she has become. My admiration and respect for her continued to grow in this story. On the other hand, it was Trey that struggles when triggers off set his vulnerabilities \n The drama Ms. Blakely incorporates in this story makes the events of TTOI look like high school play. The storyline is brilliant. Both characters must work hard to accept and or redefine their perception of themselves if they want to grow as a couple. The support they show each other will make you tear up because it is true love at its finest (rom com movies take notes from Lauren Blakely!) \n I thought my emotions were all over the place in the first book but wow was I proven wrong. As I sit here and watch luging on the Olympics, the head rush I'm sure these athletes feel following one helluva fast swerving and whip lash like turns, is how I felt after reading this book. Like that athlete the adrenaline blissful after effects were so worth the road bumps and anxiety spikes. In addition to this we have the holy grail of sex scenes. Lauren Blakely I do not know how you do it; every book you write the scenes get hotter. I'm talking don't bother wearing panties, don't read in public, boy/girlfriends, husbands, wives better stock up on their thank you cards because woman your intimate scenes leave me pantingly speechless! The intimate actions and Trey Westin's dirty (No soap could ever clean this man's mouth, nor do you want it too haha) mouth are definitely a treat. However, there is more to Trey than being sexifyingly delicious; he is also sensually compassionate and loving towards Harley. \n A lot of sequels will start off strong and fizzle out or they hold your interest but you can't stop comparing it to the first book. Have no fear neither of those things occur with Every Second With You. This sequel is by far one of my top 3 favorite sequels. The drama is new not repetitive, the sex is hotter, the romance is heart warming and the aspect that I applaud Ms. Blakely on the most is how realistic these individuals develop their sense of self-identity and as a couple! \n Be Warned--Lauren Blakely does not take it easy on you and she holds the controller on your emotions until the very end. Nothing about Harley and Trey's ride is smooth but dammit, give me Dramamine I will take my chances with motion sickness every time in order to get to that final destination that is the end of Every Second With You. Amazing, mind blowing, heart swelling, 300% satisfaction. Lauren Blakely your writing never ceases to make me more and more of a crazed fan for you. \n ~ Review by Tori</td>
    </tr>
  </tbody>
</table>
</div>

### Conclusion
**Locality-Sensitive Hashing** is an effective technique for detecting near-duplicates, providing a balance between reliability and computational efficiency, although as any probabilistic approach it may produce some false positives and false negatives.

It can be adjusted based on specific tasks, hardware resources, and precision needs. In recommendation systems, it is particularly useful for merging duplicate items, which helps create a more compact interaction matrix. This addresses memory constraints and can be used to reduce sparsity, ultimately leading to better performance.

Additionally, Locality-Sensitive Hashing could be used in anomaly detection or to help exclude certain interactions from training recommendation systems, particularly when there are suspicions of fraudulent reviews or comment campaigns used for advertising or spreading misinformation.
