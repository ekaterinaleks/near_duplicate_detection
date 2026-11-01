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

### Setup (Colab)
Skip these two cells when running locally. On Colab, add `KAGGLE_API_TOKEN` to Secrets (key icon) and put this folder's files in Google Drive.

```python
#Kaggle API token from Colab Secrets and access to Google Drive
import os
from google.colab import drive, userdata

os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
drive.mount('/content/drive')
```

```text
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
```

```python
#install dependencies
%cd "/content/drive/MyDrive/Colab Notebooks/Shared GitHub Colabs"
!pip install -q -r requirements.txt
```

```text
/content/drive/MyDrive/Colab Notebooks/Shared GitHub Colabs
```

```python
#libraries
import pandas as pd
import pickle
import itertools
import multiprocessing as mp
from pandarallel import pandarallel
from tqdm.auto import tqdm
import zipfile
import gc

#modules
import lsh_helpers as lsh
from settings import CHUNK_SIZE, JACCARD_THRESHOLD

#miscellanea
pandarallel.initialize(nb_workers=mp.cpu_count(), progress_bar=False)
pd.options.mode.chained_assignment = None
pd.set_option('max_colwidth', None)
```

```text
INFO: Pandarallel will run on 8 workers.
INFO: Pandarallel will use Memory file system to transfer data between the main process and workers.
```

### Get the Data
We will be using a dataset of reviews from Goodreads 📚 taken from [Kaggle 2022 competition](https://www.kaggle.com/competitions/goodreads-books-reviews-290312) originally downloaded from UCSD Book Graph. This dataset wasn't deduplicated as its later version currently available at [Goodreads Book Graph Datasets](https://mengtingwan.github.io/data/goodreads).

```python
#download using Kaggle API
! kaggle competitions download -c goodreads-books-reviews-290312
```

```text
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.0.2), please consider upgrading to the latest version (2.2.5)
goodreads-books-reviews-290312.zip: Skipping, found more recently modified local copy (use --force to force download)
```

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

```text
The dataframe has 1378033 rows and 10 columns.
```

```python
print(f'Are there any complete duplicates with identical data in all columns?\n{df_goodreads.duplicated().any()}')
```

```text
Are there any complete duplicates with identical data in all columns?
False
```

```python
print(f'Are there any duplicated review ids?\n{df_goodreads.duplicated(['review_id']).any()}')
```

```text
Are there any duplicated review ids?
False
```

```python
print(f'How many duplicated review texts are there?\n{df_goodreads.duplicated(['review_text'], keep=False).sum()}')
```

```text
How many duplicated review texts are there?
20539
```

```python
print(f'How many duplicated review texts are there from the same user about different books?\n{df_goodreads.duplicated(['review_text', 'user_id'], keep=False).sum()}')
```

```text
How many duplicated review texts are there from the same user about different books?
13788
```

```python
print(f'How many duplicated review texts are there from different users about the same book?\n{df_goodreads.duplicated(['review_text', 'book_id'], keep=False).sum()}')
```

```text
How many duplicated review texts are there from different users about the same book?
601
```

```python
print(f'Are there any duplicated review texts from the same user about the same book?\n{df_goodreads.duplicated(['review_text', 'book_id','user_id']).any()}')
```

```text
Are there any duplicated review texts from the same user about the same book?
False
```

### Detect near-Duplicates: Case Study and Algorithm
**Case Study Dataset**

Although there are no complete duplicates for reviews submitted by the same user about the same book, [Goodreads' users do complain about these cases](https://www.goodreads.com/topic/show/22591024-duplicate-reviews). This problem likely stems from reposted reviews with slight wording, spelling and/or punctuation changes that can't be detected as easily as complete duplicates.

Also, we detected only 13788 completely duplicated review texts from the same user about different books. However, reading the forum, we can deduce that there are more of these with slight changes.

**Textual Similarity: Lexical vs Semantic & Computational Cost**

Given the massive amounts of textual data used in ML, finding similar texts is a fairly common task. However, the approach used varies greatly based on the use case and the amount of data to be processed.

While Transformers revolutionized the semantic similarity search, this approach is hardly usable when detecting near-duplicates: semantically close texts can be rather different on a lexical level, and the cost of calculating embeddings is too high for such a seemingly trivial task.

A more basic and straightforward approach like Jaccard similarity that could be used for lexical similarity also has its drawbacks: it doesn't take into account word order and is computationally unfeasible for large datasets as it needs to compare each pair one by one. For example, for our dataset of 1,378,033 reviews that would result in 949,486,785,528 pairs.

**Locality-Sensitive Hashing Technique**

Locality-sensitive hashing (LSH) allows us to preselect pairs that are likely to be near-duplicates by hashing the items and dividing them into buckets of potentially similar texts. This way we can use these probabilistic pairs, keeping in mind that there would be some false positives and false negatives. Or, if we need to maximize precision and get rid of false positives, we can calculate Jaccard similarity only for these candidate pairs and still substantially reduce the computation cost compared to brute-force Jaccard similarity calculation for all possible pairs.

***Algorithm Used***

The theory behind the algorithm: [Chapter 3 from "Mining Massive Datasets"](http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf).

Library used: [datasketch](https://github.com/ekzhu/datasketch).

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
with tqdm(total=len(df_goodreads), desc='minhash', unit='review') as progress:
    chunk = list(itertools.islice(it, CHUNK_SIZE))
    while chunk:
        df_goodreads.loc[chunk, 'minhash'] = df_goodreads.loc[chunk, 'clean_review_text'].parallel_apply(lsh_instance.take_minhash)
        progress.update(len(chunk))
        chunk = list(itertools.islice(it, CHUNK_SIZE))
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

**Speaking of near-identical reviews from the same user but about different books, above we already detected 13788 exact duplicates. Has this number changed after including near-duplicates?**

```python
#check for near-duplicates posted about the same book by different users
df_goodreads['user_group'] = df_goodreads.groupby(['user_id', 'set_of_doubles']).ngroup()
df_goodreads['user_number'] = df_goodreads.groupby('user_group')['user_group'].transform(lambda x: len(x))
```

```python
print(f'How many near-duplicated review texts are there from the same user about different books?\n{df_goodreads.query('user_number > 1').shape[0]}')
```

```text
How many near-duplicated review texts are there from the same user about different books?
15959
```

**In some cases the initial wording makes it seem different (and impossible to spot without near-duplicate detection), but the core review is still the same and can be processed as one piece.** ⬇️

```python
df_goodreads.loc[[1243209, 1243210, 1243211], ['user_id', 'book_id', 'review_text']]
```

<div>
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

**Others are just people trying to send traffic to their blogs and don't contribute any meaningful information. However, the text itself varies, making it difficult to detect.** ⬇️

```python
df_goodreads.loc[[470436, 470437, 470444, 470450, 470483, 470484, 470485, 470486, 470487, 470490, 470495, 470509, 470510], ['user_id', 'book_id', 'review_text']]
```

<div>
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

**The following two sets of reviews are original and later versions of the same piece of writing, most probably posted for different editions. The introductions in both cases are a bit different, making it impossible to detect them as complete duplicates** ⬇️

```python
df_goodreads.loc[[209316, 209365], ['user_id', 'book_id', 'review_text']]
```

<div>
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

**Speaking of near-identical reviews from different users but about the same book, above we already detected 601 exact duplicates. Has this changed after including near-duplicates?**

```python
#check for near-duplicates posted about the same book by different users
df_goodreads['book_group'] = df_goodreads.groupby(['book_id', 'set_of_doubles']).ngroup()
df_goodreads['book_number'] = df_goodreads.groupby('book_group')['book_group'].transform(lambda x: len(x))
print(f'How many near-duplicated review texts are there from different users about the same book?\n{df_goodreads.query('book_number > 1').shape[0]}')
```

```text
How many near-duplicated review texts are there from different users about the same book?
1098
```

```python
#let's check some longer reviews that are unlikely to be a typical short "review to come" placeholder
df_goodreads['review_length'] = df_goodreads['review_text'].apply(len)
df_goodreads.query('(book_number > 1) & (review_length > 200)').sort_values('book_group')[['user_id', 'book_id', 'review_text']]
```

<div>
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
    <tr>
      <th>1205978</th>
      <td>8d296efe4f23f9d2747ef37e1c38b173</td>
      <td>14383</td>
      <td>Kaderin the cold-hearted has spent the last thousand years not feeling. She lost them when she watched her sisters die by the same Horde vampire she showed mercy to in battle mere seconds before. Believing her lack of feelings are a blessing she hasn't felt anything since. No happiness, joy, fear, or sadness, just nothing. Now finding herself in Sebastian Wroth's home she can't seem to bring herself to kill him. She see's something in him. Something she just can't bring herself to end. \n This is the third book in the Immortals After Dark series and I'm really enjoying them I just have one small complaint and that's the fact that the last three books seem to have the same formula. All of the girls have had the same excuse time and time again. They push and push and push the men away and the men are relentless in their fight, never giving up. It's endearing and irritating at the same time. \n This one did have me thinking and thinking about time travel and how it just doesn't make sense no matter what way you look at it. I'm glad that Cole dealt with that and I feel like it was handled in the best possible way. \n However, I obviously have to continue on because I can't stop thinking about poor Bowen. He too, was competing in the Hie for a chance to go back and save his mate. I couldn't stop thinking about how much he's suffered.</td>
    </tr>
    <tr>
      <th>1363866</th>
      <td>3693bb4f1062b659a354848cf11ca313</td>
      <td>14383</td>
      <td>Kaderin the Cold-Hearted, has spent the last thousand years not feeling. She lost them when she watched her sisters die by the same Horde vampire she showed mercy to in battle mere seconds before. Believing her lack of feelings are a blessing she hasn't felt anything since. No happiness, joy, fear, or sadness, just nothing. Now finding herself in Sebastian Wroth's home, a vampire, she can't seem to bring herself to kill him. She see's something in him. Something she just can't bring herself to end. \n This is the third book in the Immortals After Dark series and I'm really enjoying them I just have one small complaint and that's the fact that the last three books seem to have the same formula. All of the girls have had the same excuse time and time again. They push and push and push the men away and the men are relentless in their fight, never giving up. It's endearing and irritating at the same time. \n This one did have me thinking about time travel and how it just doesn't make sense no matter what way you look at it. I'm glad that Cole dealt with that and I feel like it was handled in the best possible way. I must hand it to her though, even after I put the book down I was contemplating and coming up with how it couldn't possible. \n However, I obviously have to continue on in this series because I can't stop thinking about poor Bowen. He too, was competing in the Hie for a chance to go back and save a loved one. I couldn't stop thinking about how much he's suffered.</td>
    </tr>
    <tr>
      <th>702117</th>
      <td>b6aceda54f217ed2f7148239fef3fca5</td>
      <td>14384</td>
      <td>I purchased both book 1 (The Warlord Wants Forever) and book 2 A Hunger Like No Other (which is labeled book 1, but really isn't) at reduced prices to give IAD a try. I haven't read allot of series PNR so I wasn't really expecting to like it; and what a pleasant surprise it has been. \n Getting into Cole's world took a little while (at first I found the whole idea of "fated" mates" a little hokey) but it works well with what seems to be her basic love/hate trope (which I personally like). I thought there was a nice balance between romance and plot (sort of 50/50). I am also enjoying the way characters and story lines from the previous book are carried forwarded (although I guess all series are like that). It was nice to see Myst and Wroth from TWWF making an appearance in the story and to see the continued development of all of Emma's Valkyrie aunts. \n Emma turned out to be a strong character and I enjoyed her transformation from "Emma the Timid" to (SPOILER) "Emma the King Killer" (END "SPOILER"). I wasn't particularity taken with Lachlain initially, he rather came across as a blundering oaf...but then considering he was coming off 150 years of hideous torture and was temporarily insane...I accepted he was not at his best. He and Emma ended up being pretty sweet together and it's hard to deny that a Scottish accent isn't hot; but I would still take Wroth any day of the week. \n AHLNO was fun and I already ordered the next book in the series.</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1051539</th>
      <td>edce295372f8b4a773a1c74a24bc7268</td>
      <td>24895218</td>
      <td>4 LOVING STARS!!!! \n What to do when the stranger you met and spent the most amazing weekend with ever turns out to be your stepbrother? \n Aria met a Mystery man a year ago and spent a weekend with him, only for her Prince Charming to up and leave. Some time after that, her mother meets the man of her dreams and weds. A year later, it is finally time for Aria to meet her stepbrothers, and when she does, she realizes the eldest is her Mystery Man...the Mystery Man who took her virginity a year earlier...her mystery man! And Mystery Man has a name...BLADE. \n Trying to fight what was never done between them proves hard. Aria has experienced devastating pain in her life, pain that has her screaming in the mornings. She won't be able to deal if he leaves again, but she doesn't know how to not want him, and frankly, neither does he. So, they both fight the intense attraction, knowing that being with each other is wrong.......right? \n ******* \n These stepbrother books are KILLING me! Lol!!! Step-lover was such a beautiful, heartbreaking and intense love story! I am so happy that I read this! I enjoyed watching things develop between Aria and Blade, I could feel their chemistry with every turn of the page. This was an intense build between the two and whenever they gave in to what they were feeling, it was dynamite! \n As much as the book speaks of insta-love, it didn't feel like that to me. It took a year for them to see each other again, so it was like a slow burn for both of them. It was obvious that neither of them got the other out of their head or heart. Both Aria and Blade were strong characters. I never felt like they were stupid or petty. I actually felt sorry for Aria quite a bit, only because I could feel her pain from not getting the man she loves. The fight they had to go through to get where they got at the end was so sad for me to read. I wish I could have waved a magic wand &amp; everything was right for them. Sadness was palpable, desperation was so tangible, and the love.......jeez, their love and passion: enviable. \n The secondary characters added the next bit of amazing this book needed. It would be nothing without these characters: the best friend (Mel), the other brothers/stepbrothers (Brody &amp; Lee), and the parents/stepdad (Nancy &amp; Jack). Hell, even Bryce, Yates &amp; Peyton, minor characters, had major roles to play in the development of the story, as well as the outcome! I look forward to Mel and Brody's story...Miss Jewel can't hurry fast enough with that one! ^_^ \n I was very impressed with the writing. So easy to read, coherent, well-worded with probably two or three "mistakes". I wasn't even looking for them, I was so engrossed in the story! Great character development, great back story for both (that you have to read to understand). My only qualm is that I NEEDED more. Not just more story since I didn't want it to end, but MORE of a conclusion/resolution. The ending felt abrupt, no buildup. I felt like there was more that could be said, more that could have happened, more intricacies to their relationship that could have been revealed to give the story that much more of an edge....which would explain my 4 stars. I guess I will have to wait for Mel and Brody's book to find out how things further go with Ari and Blade, but I do hope they get a conclusion novella for themselves because I really think they need a proper happy ending. :) \n The quote at the top represents all this story is. So go in with an open heart. One-click IMMEDIATELY, you won't regret it, unless you're an asshole/troll. -_-</td>
    </tr>
    <tr>
      <th>558462</th>
      <td>1a24bac964b67d5afe7175b79573d1a6</td>
      <td>30818614</td>
      <td>The Room Mate (Room Mates #1) by Kendall Ryan \n 4 stars!!! \n "He might be young, twenty-four years to my twenty-eight, but his eyes spoke of wisdom and maturity." \n Kendall Ryan delivers a fun, flirty read that ticks all those boxes for a hopeless romantic like me. While, on first impressions, Cannon should be someone you dislike for his ego alone, you cannot help but fall in love with our doctor in the making and it does seem that it isn't his ego talking, but it is his reality. \n Cannon is in his last year of residency before he becomes a fully qualified doctor. Cannon is extremely professional and motivated and his quest is only fuelled by his inherent belief that being the man of the family, he has to be there to support his mom and sister...cue ovaries melting. Cannon tries his hardest to stay away from the opposite sex, it seems that every time he sleeps with one, they instantly fall in love and become some crazed stalker. Even one night stands are off-limits for him. While this did give me the proverbial eye roll, it is a novel spin and led to some hilarious scenes as well as some not. \n "Growing up the way I did, I guess it shaped my goals. Now I'm just perfecting the art of making lemonade." \n Paige is Cannon's sisters best friend...older best friend. Cannon has lusted over Paige for over a decade, she has been the object of his teenage fantasies for years but they haven't met in the flesh for about five years. A lot changes in five years. After another stalker issue, Cannon finds himself homeless and only needing a place to stay for a couple of months, Cannon's sister sets him up to stay with Paige. \n When these two meet, you can tell that any suppressed lust Cannon had for Paige soon resurfaced and it seems that Paige wasn't immune either. This was a forbidden romance of sorts, not only because Paige was a few years older than Cannon but the fact that Cannon was her best friends little brother, a little brother that she was extremely protective over. With careers and friendships telling them that this was definitely a bad idea...what happens when chemistry tells you it isn't? \n "...he was the realest, best, brightest thing in my world, and I hated that we would never be more." \n This was one really enjoyable read that seemed to end as quick as it began. Totally immersed in their story the pages flowed effortlessly and with only a little drama the pacing was perfect. I loved their relationship, the dialogue and how they seamlessly converged into each other's lives. This was a cute read, relatable, believable and heart-warming. \n "Nothing about us was a mistake." \n www.theromancecover.com</td>
    </tr>
    <tr>
      <th>1217197</th>
      <td>fccc5f50c862761742285b249b45bc38</td>
      <td>30818614</td>
      <td>Thank you Donna C from Amazon \n "He might be young, twenty-four years to my twenty-eight, but his eyes spoke of wisdom and maturity." \n Kendall Ryan delivers a fun, flirty read that ticks all those boxes for a hopeless romantic like me. While, on first impressions, Cannon should be someone you dislike for his ego alone, you cannot help but fall in love with our doctor in the making and it does seem that it isn't his ego talking, but it is his reality. \n Cannon is in his last year of residency before he becomes a fully qualified doctor. Cannon is extremely professional and motivated and his quest is only fueled by his inherent belief that being the man of the family, he has to be there to support his mom and sister...cue ovaries melting. Cannon tries his hardest to stay away from the opposite sex, it seems that every time he sleeps with one, they instantly fall in love and become some crazed stalker. Even one night stands are off-limits for him. While this did give me the proverbial eye roll, it is a novel spin and led to some hilarious scenes as well as some not. \n "Growing up the way I did, I guess it shaped my goals. Now I'm just perfecting the art of making lemonade." \n Paige is Cannon's sisters best friend...older best friend. Cannon has lusted over Paige for over a decade, she has been the object of his teenage fantasies for years but they haven't met in the flesh for about five years. A lot changes in five years. After another stalker issue, Cannon finds himself homeless and only needing a place to stay for a couple of months, Cannon's sister sets him up to stay with Paige. \n When these two meet, you can tell that any suppressed lust Cannon had for Paige soon resurfaced and it seems that Paige wasn't immune either. This was a forbidden romance of sorts, not only because Paige was a few years older than Cannon but the fact that Cannon was her best friends little brother, a little brother that she was extremely protective over. With careers and friendships telling them that this was definitely a bad idea...what happens when chemistry tells you it isn't? \n "...he was the realest, best, brightest thing in my world, and I hated that we would never be more." \n This was one really enjoyable read that seemed to end as quick as it began. Totally immersed in their story the pages flowed effortlessly and with only a little drama the pacing was perfect. I loved their relationship, the dialogue and how they seamlessly converged into each other's lives. This was a cute read, relatable, believable and heart-warming. \n "Nothing about us was a mistake."</td>
    </tr>
    <tr>
      <th>1004333</th>
      <td>874507ab349b2258eaecd967f4573c00</td>
      <td>36252773</td>
      <td>Joint review with my blog partner, Tori: \n Tori: I have never been one for the daddy kink. Most erotic romances I've encountered portray it as 24/7 acting like a baby and that's just not for me. However, I have been a fan of Eve Dangerfield's racy contemps since reading her debut, Degrees of Control, and more than willing to give this one a shot. And I'm so glad I did. Act Your Age is a sexy, humorous, and bittersweet opposite attraction erotic romance that places its focus more on the characters and their issues rather than the kink the storyline is built around. Dangerfield has a fantastic voice that creates a sense of comraderie between the reader and her characters. She writes about everyday people, with regular lives and problems who like kinky sex. The angst levels are fairly low with some drama but nothing over the top \n Heavily character driven, Dangerfield introduces us to our hero and heroine, Ty Henderson and Kate McGrath, setting up the storyline while allowing us ample time to get to know them. I found both of them to be delightfully messy contradictions. I laughed while cringing at the two of them more often than not at their awkward and HAWT scenes. Ty works at a large engineering firm where Kate was recently hired on as an engineer. Though he isn't her immediate boss, he is higher up the chain of command than she is. Still recovering from a bad break up, Ty stomps around like a grumpy butthead with one leg in the grave. Drinking too much and avoiding the home he shared with his cheating ex fiancee, Ty's attraction to Kate only makes him angry and he directs all that anger towards her. \n Ty didn't think Middleton was a great girl. He thought Middleton was a pain in the ass. Waltzing around with her shiny hair and long legs and her throaty voice, being cuter than a fistful of buttons. Where did she get off? \n Kate is young, energetic, and flints around like Pollyanna on acid. She indulges in juvenile clothing and brings homemade baked good to the office on a daily basis. Suffering from ADHD that resulted in some bad childhood behavior and teasing, Kate now takes Ritalin and quietly pines for some positive attention from Ty. \n That he didn't like her always would have hurt. That he didn't like her and she was so hopelessly attracted to him burned. In the kind of all-encompassing obsession she'd only ever read about in books, but unlike books, her obsession never went anywhere or did anything. She just got sweaty palms whenever she saw him and it made her work-life very awkward. Her obsession wasn't even original. Everyone had a crush on Tyler Henderson. Women, gay men, straight men, more perceptive animals. It was like having a crush on a Hemsworth brother--it made you basic as hell \n What did you think of Kate and Ty in the beginning, Mandi? \n Mandi: Before I get into our joint review, I wanted to take a quick minute to highlight what this author and Kate, our heroine, means when she says she likes daddy kink. Maybe you have never read this kink, or you have a preconceived notion as to what it is. So in Kate's words: \n When I imagine daddy-daughter play, I imagine entering a state of mind in which I'm entirely innocent and the man I'm with corrupts and disciplines me. To me, this is power play, not a desire to experience incest or genuine trauma. I'm NOT into the extreme aspects of age-play or daddy-daughter kink (infantilization, pacifiers, baby-talk, or nappies). I consider myself more of a submissive with a twist. \n Let me note, their main form of role-playing is Ty is the "step dad" and Kate is the "teenage daughter home alone while mom is at work." There are times Kate pretends she doesn't want her "step dad" to have sex with her and he forces himself. It's all 100% consensual and they are both totally into it - but I wanted to make sure you know that, in case that is not your jam. \n I ended up liking this book a lot. It's over the top sometimes. The book got nicknamed, "The baseball bat" book on Twitter because I tweeted this quote: \n Erect, his penis was approximately the width of a baseball bat, properly and ridiculously huge. "What the heck...?" \n Yes, that is some girth but don't let this over the top description deter you. This is a very erotic, and explicit book but it's not all just sex - the relationship is built up very well. But first, the sex. They go heavy into role-playing and Kate calling Ty daddy. Remember they are in role play here.... \n "Say that again." Kate's heart leapt into her mouth. Was he angry? Was he going to yell at her? "I-I like sucking your cock." \n "Little girl." His voice was icy with disdain. "Say what you just called me again." \n But she couldn't. Instead she asked, "Are you mad at me?" \n There was a silence more loaded than all the guns in a gun-nut's gun locker. Then he spoke. \n "No, but I want you to say it again." \n "But I--" \n "Not what I want to hear, sweetheart." He gently forced her head back, not hard, but firm. His manhandling and the sense of powerlessness it arose in her made Kate quiver. How did you know? she wanted to ask. I never told you, how did you know? But, she supposed in the scheme of things that didn't matter as much as following orders. She swallowed. \n "I like sucking your cock, Daddy." \n I could go on and on with sex quotes - there are a lot! But there is more to this story. The author takes her time letting us get to know Kate. From initial reaction, I thought disney princess. But as the author peels back her layers - we learn she was diagnosed with severe ADHD when she was seventeen and has been on Ritalin ever since. She didn't have that great of a childhood. She struggles with friendships. She struggles with being assertive and doesn't really know who she is. \n On the other hand, Ty was dumped by his fiancee two years prior for not wanting to have kids. He has had a hard time getting over her. He likes to be assertive and in control and when he does finally go in the bedroom with Kate, he can't believe he actually found someone who can fulfill his desires - same with her. \n Kate eventually finds female friendships, and loses some too - for so many pages devoted to spanking, we dive into complicated, personal issues as well. I appreciated that. \n Tori, aside from the erotic scenes, did you find their relationship solid and well done? \n Tori: Yes. I loved their relationship. The imperfections, the miscommunications, and all the jagged little edges only made it more real and endearing to me. Dangerfield doesn't pull any punches with this couple as she guides us through their relationship using dual narrative. We see them at the best and worst. I enjoy when an author tries to convince me two people who seem completely wrong for one another are actually perfect together. Dangerfield gently cuts through all the artifice and layers, letting us see what she's always known-that our protagonists are indeed perfect for one another. \n Ty and Kate grow so much during this story. Both of them have problems with communication and their sexuality. Kate is already working towards inner acceptance while Ty is just beginning. Their age difference and working relationship is a huge stumbling block for Ty. He's already lost one woman he loved and he is scared to take another risk. \n Middleton crawled into my hotel bed because she's attracted to old men and I was in the right place at the right time. We start dating. The sex is good. Her family think she's insane for fucking someone old enough to be her dad. All my friends think I'm having a mid-life crisis. People whisper behind our backs at parties. The age gap sounds bad and she's so baby-faced it looks worse. The fact I used to be her boss is the cherry on top of the shit sundae. Doesn't stop us from fucking one another, but the smell of it taints everything like when there's a dead rat in your ventilation. A few years pass, I get even fucking older, my hair starts to fall out, and my dick doesn't work. Middleton meets a guy her own age. She leaves me. I die alone. \n The erotic sexual components are exquisitely written and blend perfectly with the building romance and main storyline. Dangerfield hits all the emotional tones in here. While their role playing starts out as just a way to scratch each others itches, they soon they begin to take on the personalities in those roles. Ty's protective instincts roar to the front as he becomes more invested in Kate's happiness while Kate flourishes by all the attention and comfort Ty lavishes on her. \n "When she pressed her face against his chest he wanted to give her everything. He'd never, ever felt that before." \n What did you think of their relationship and the path Dangerfield choose to take them down, Mandi? \n Mandi: I thought their relationship was so well-developed. I love the balance of erotic scenes with courtship and learning about Ty and Kate's background and establishing a foundation for their relationship. I've read a few other Eve Dangerfield books and she is an author that has now made my "anticipation list." \n Grade: B+ \n Tori: A personable cast of secondary characters are used to push Kate and Ty forward as individuals and as a couple. The Derby team not only gives Kate a much-needed boost of self-confidence to grab for what she wants, but also the courage to walk away. I loved their banter , energy, and the hidden dynamics of Kate's relationship with them that Eve teases out. \n Eve Dangerfield is definitely on my list of authors to watch for \n Tori's Favorite Quotes: "If I fuck you, you'll think all your prayers have been answered. I'll make you see god, Middleton. I'll make you think I am god." \n "You're so behind on the oral sex ratio it's not even funny." \n Tori's Grade: B+</td>
    </tr>
    <tr>
      <th>1090859</th>
      <td>c0e0fda388f87af0deffad748c9c8b67</td>
      <td>36252773</td>
      <td>Reviewed by Mandi and Tori \n Tyler Henderson, a 45 year old former golden boy with the personality of the grinch, just wants to be called daddy but isn't sure how to ask without sounding like a pervert. \n Kate 'Middleton' McGrath, a 25 year old engineer who looks and dresses like Lolita just happens to be looking for a daddy. \n Kate knows Ty would be the perfect daddy for her fantasies, but she's pretty sure he can't stand her. That is until a business trip, one too many drinks, and a case of mistaken identity shows them just how good it can be between them. \n As Ty and Kate explore their mutual kinks, acting out all their fantasies, their feelings for one another begin to deepen. When Ty has a chance to prove he's all in, he fails. Now he has to convince Kate that not only can he be her daddy...he can also be her man. \n Tori: I have never been one for the daddy kink. Most erotic romances I've encountered portray it as 24/7 acting like a baby and that's just not for me. However, I have been a fan of Eve Dangerfield's racy contemps since reading her debut, Degrees of Control, and more than willing to give this one a shot. And I'm so glad I did. Act Your Age is a sexy, humorous, and bittersweet opposite attraction erotic romance that places its focus more on the characters and their issues rather than the kink the storyline is built around. Dangerfield has a fantastic voice that creates a sense of comraderie between the reader and her characters. She writes about everyday people, with regular lives and problems who like kinky sex. The angst levels are fairly low with some drama but nothing over the top \n Heavily character driven, Dangerfield introduces us to our hero and heroine, Ty Henderson and Kate McGrath, setting up the storyline while allowing us ample time to get to know them. I found both of them to be delightfully messy contradictions. I laughed while cringing at the two of them more often than not at their awkward and HAWT scenes. Ty works at a large engineering firm where Kate was recently hired on as an engineer. Though he isn't her immediate boss, he is higher up the chain of command than she is. Still recovering from a bad break up, Ty stomps around like a grumpy butthead with one leg in the grave. Drinking too much and avoiding the home he shared with his cheating ex fiancee, Ty's attraction to Kate only makes him angry and he directs all that anger towards her. \n Ty didn't think Middleton was a great girl. He thought Middleton was a pain in the ass. Waltzing around with her shiny hair and long legs and her throaty voice, being cuter than a fistful of buttons. Where did she get off? \n Kate is young, energetic, and flints around like Pollyanna on acid. She indulges in juvenile clothing and brings homemade baked good to the office on a daily basis. Suffering from ADHD that resulted in some bad childhood behavior and teasing, Kate now takes Ritalin and quietly pines for some positive attention from Ty. \n That he didn't like her always would have hurt. That he didn't like her and she was so hopelessly attracted to him burned. In the kind of all-encompassing obsession she'd only ever read about in books, but unlike books, her obsession never went anywhere or did anything. She just got sweaty palms whenever she saw him and it made her work-life very awkward. Her obsession wasn't even original. Everyone had a crush on Tyler Henderson. Women, gay men, straight men, more perceptive animals. It was like having a crush on a Hemsworth brother--it made you basic as hell \n What did you think of Kate and Ty in the beginning, Mandi? \n Mandi: Before I get into our joint review, I wanted to take a quick minute to highlight what this author and Kate, our heroine, means when she says she likes daddy kink. Maybe you have never read this kink, or you have a preconceived notion as to what it is. So in Kate's words: \n When I imagine daddy-daughter play, I imagine entering a state of mind in which I'm entirely innocent and the man I'm with corrupts and disciplines me. To me, this is power play, not a desire to experience incest or genuine trauma. I'm NOT into the extreme aspects of age-play or daddy-daughter kink (infantilization, pacifiers, baby-talk, or nappies). I consider myself more of a submissive with a twist. \n Let me note, their main form of role-playing is Ty is the "step dad" and Kate is the "teenage daughter home alone while mom is at work." There are times Kate pretends she doesn't want her "step dad" to have sex with her and he forces himself. It's all 100% consensual and they are both totally into it - but I wanted to make sure you know that, in case that is not your jam. \n I ended up liking this book a lot. It's over the top sometimes. The book got nicknamed, "The baseball bat" book on Twitter because I tweeted this quote: \n Erect, his penis was approximately the width of a baseball bat, properly and ridiculously huge. "What the heck...?" \n Yes, that is some girth but don't let this over the top description deter you. This is a very erotic, and explicit book but it's not all just sex - the relationship is built up very well. But first, the sex. They go heavy into role-playing and Kate calling Ty daddy. Remember they are in role play here.... \n "Say that again." Kate's heart leapt into her mouth. Was he angry? Was he going to yell at her? "I-I like sucking your cock." \n "Little girl." His voice was icy with disdain. "Say what you just called me again." \n But she couldn't. Instead she asked, "Are you mad at me?" \n There was a silence more loaded than all the guns in a gun-nut's gun locker. Then he spoke. \n "No, but I want you to say it again." \n "But I--" \n "Not what I want to hear, sweetheart." He gently forced her head back, not hard, but firm. His manhandling and the sense of powerlessness it arose in her made Kate quiver. How did you know? she wanted to ask. I never told you, how did you know? But, she supposed in the scheme of things that didn't matter as much as following orders. She swallowed. \n "I like sucking your cock, Daddy." \n I could go on and on with sex quotes - there are a lot! But there is more to this story. The author takes her time letting us get to know Kate. From initial reaction, I thought disney princess. But as the author peels back her layers - we learn she was diagnosed with severe ADHD when she was seventeen and has been on Ritalin ever since. She didn't have that great of a childhood. She struggles with friendships. She struggles with being assertive and doesn't really know who she is. \n On the other hand, Ty was dumped by his fiancee two years prior for not wanting to have kids. He has had a hard time getting over her. He likes to be assertive and in control and when he does finally go in the bedroom with Kate, he can't believe he actually found someone who can fulfill his desires - same with her. \n Kate eventually finds female friendships, and loses some too - for so many pages devoted to spanking, we dive into complicated, personal issues as well. I appreciated that. \n Tori, aside from the erotic scenes, did you find their relationship solid and well done? \n Tori: Yes. I loved their relationship. The imperfections, the miscommunications, and all the jagged little edges only made it more real and endearing to me. Dangerfield doesn't pull any punches with this couple as she guides us through their relationship using dual narrative. We see them at the best and worst. I enjoy when an author tries to convince me two people who seem completely wrong for one another are actually perfect together. Dangerfield gently cuts through all the artifice and layers, letting us see what she's always known-that our protagonists are indeed perfect for one another. \n Ty and Kate grow so much during this story. Both of them have problems with communication and their sexuality. Kate is already working towards inner acceptance while Ty is just beginning. Their age difference and working relationship is a huge stumbling block for Ty. He's already lost one woman he loved and he is scared to take another risk. \n Middleton crawled into my hotel bed because she's attracted to old men and I was in the right place at the right time. We start dating. The sex is good. Her family think she's insane for fucking someone old enough to be her dad. All my friends think I'm having a mid-life crisis. People whisper behind our backs at parties. The age gap sounds bad and she's so baby-faced it looks worse. The fact I used to be her boss is the cherry on top of the shit sundae. Doesn't stop us from fucking one another, but the smell of it taints everything like when there's a dead rat in your ventilation. A few years pass, I get even fucking older, my hair starts to fall out, and my dick doesn't work. Middleton meets a guy her own age. She leaves me. I die alone. \n The erotic sexual components are exquisitely written and blend perfectly with the building romance and main storyline. Dangerfield hits all the emotional tones in here. While their role playing starts out as just a way to scratch each others itches, they soon they begin to take on the personalities in those roles. Ty's protective instincts roar to the front as he becomes more invested in Kate's happiness while Kate flourishes by all the attention and comfort Ty lavishes on her. \n "When she pressed her face against his chest he wanted to give her everything. He'd never, ever felt that before." \n What did you think of their relationship and the path Dangerfield choose to take them down, Mandi? \n Mandi: I thought their relationship was so well-developed. I love the balance of erotic scenes with courtship and learning about Ty and Kate's background and establishing a foundation for their relationship. I've read a few other Eve Dangerfield books and she is an author that has now made my "anticipation list." \n Grade: B+ \n Tori: A personable cast of secondary characters are used to push Kate and Ty forward as individuals and as a couple. The Derby team not only gives Kate a much-needed boost of self-confidence to grab for what she wants, but also the courage to walk away. I loved their banter , energy, and the hidden dynamics of Kate's relationship with them that Eve teases out. \n Eve Dangerfield is definitely on my list of authors to watch for. \n Tori's Grade: B+</td>
    </tr>
  </tbody>
</table>
</div>

**This one is clearly written by the same person and isn't an exact duplicate only due to some typos:**
⬇️

```python
df_goodreads.loc[[234437,1143023], ['user_id', 'book_id', 'review_text']]
```

<div>
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

```python
#free memory: the LSH index and the signatures are no longer needed
del lsh_object
df_goodreads.drop(columns='minhash', inplace=True)
gc.collect()
```

```text
12254806
```

### Checking the Matches

LSH only proposes candidate pairs, so some of them are false positives.

To get the final counts, we can check each pair with the exact Jaccard similarity and keep it only if it reaches the threshold.
The true positive pairs confirmed by Jaccard are then merged into clusters, so a review similar to several others ends up in a single cluster.

Note that this check only removes false positives. Near-duplicate pairs that LSH missed (false negatives) cannot be recovered at this step.

```python
#get the cleaned text of each review by its id
texts = dict(zip(df_goodreads['review_id'], df_goodreads['clean_review_text']))
```

```python
#choose one representative review for each distinct cleaned text, so identical texts are scored only once
df_goodreads['representative_id'] = df_goodreads.groupby('clean_review_text')['review_id'].transform('first')
representative_dict = dict(zip(df_goodreads['review_id'], df_goodreads['representative_id']))
representative_ids = set(df_goodreads['representative_id'])
```

```python
#get the LSH candidates of a representative of each match
representative_candidates = {_id: {representative_dict[match] for match in near_duplicates_dict[_id]} for _id in representative_ids}
```

```python
#calculate the exact Jaccard similarity for each candidate pair of representative documents
scored_pairs = lsh_instance.score_candidates(texts=texts, candidates=representative_candidates)
print(f'How many candidate pairs are there to check?\n{len(scored_pairs)}')
```

```text
How many candidate pairs are there to check?
4184
```

```python
#keep only the pairs that reach the similarity threshold
verified_pairs = [(a, b) for a, b, similarity in scored_pairs if similarity >= JACCARD_THRESHOLD]
print(f'How many of them are confirmed by the exact Jaccard similarity of at least {JACCARD_THRESHOLD}?\n{len(verified_pairs)} ({len(verified_pairs) / max(len(scored_pairs), 1):.1%})')
```

```text
How many of them are confirmed by the exact Jaccard similarity of at least 0.8?
3520 (84.1%)
```

```python
#merge the confirmed pairs and the links to identical texts into groups, so every review id gets a group id
identical_text_pairs = [(_id, rep) for _id, rep in representative_dict.items() if _id != rep]
final_groups = lsh_instance.connected_components(ids=texts.keys(), pairs=verified_pairs + identical_text_pairs)
df_goodreads['verified_group'] = df_goodreads['review_id'].map(final_groups)
```

**Let's go back to the two questions from above. Do the numbers change once only the confirmed matches are kept?**

```python
#count again by user and by book, now using the confirmed groups
verified_user_counts = df_goodreads.groupby(['user_id', 'verified_group'])['review_id'].transform('size')
verified_book_counts = df_goodreads.groupby(['book_id', 'verified_group'])['review_id'].transform('size')
print(f'After checking: how many duplicated and near-duplicated review texts are there from the same user about different books?\n{(verified_user_counts > 1).sum()}')
print(f'After checking: how many duplicated and near-duplicated review texts are there from different users about the same book?\n{(verified_book_counts > 1).sum()}')
```

```text
After checking: how many duplicated and near-duplicated review texts are there from the same user about different books?
16298
After checking: how many duplicated and near-duplicated review texts are there from different users about the same book?
1529
```

**Here is one of the verified groups. The texts are not identical, but they were put in the same group:** ⬇️

```python
#pick a small verified group that contains different texts, not only identical ones
group_size = df_goodreads.groupby('verified_group')['review_id'].transform('size')
text_variants = df_goodreads.groupby('verified_group')['clean_review_text'].transform('nunique')
sample_group = df_goodreads.loc[(text_variants > 1) & (group_size == 5), 'verified_group'].iloc[5]
df_goodreads.loc[df_goodreads['verified_group'] == sample_group, ['user_id', 'book_id', 'review_text']]
```

<div>
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
      <th>3134</th>
      <td>0d344a261de9ab42e62cf9b3b7c52cc4</td>
      <td>18170039</td>
      <td>This novella did no favors for my opinion toward Aspen. I feel a mighty need to punch someone in the face now. \n ______________________________ \n This review can also be found at The Starry-Eyed Revue. \n **I'm reviewing this series as a whole and there are likely to be spoilers below. If you haven't started the series yet, and you actually care about the outcome, I suggest you don't read any further.** \n I never had any intention of reading this series. After the fiasco with Wendy Darling and the author and her agent -- publicist? Idk...it's been so long that I can't remember -- I immediately shelved all the books by this author as "not in this lifetime". But I was perusing my library's audiobook selection, and I was bored and couldn't decide what to listen to next when I came across The Selection. And from there, a very bad idea was born. \n I didn't even know the final book was releasing soon when I picked up the first two books plus the novellas...that's how little attention I've paid to this series. And yet, I still embarked upon my mission to read the series in its entirety, knowing what a train wreck it would be. And it was: you know it's terrible, you're cringing the entire time, and yet you just. Can't. Look. Away. \n I don't enjoy reality TV...at all. I don't find it even mildly entertaining usually, but if I discover it already on the television when I sit down, I admit that it's very addictive and hard to look away from. Which is one of the reasons I detest it so much...what a time suck! Such was my experience with this series. I didn't want to keep reading about these girls and all their drama or that love triangle of doom. After all, I despise The Bachelor and this story is that show's premise in book form. But morbid curiosity just wouldn't let me pass up this series. \n I think one of the things that helped me maintain my sanity was the fact that I was listening to the audio instead of wasting my precious reading time. I've listened to several audiobooks narrated by Amy Rubinate, so I know that I like her narration style. I can't say that I was impressed with her performance this time around, though I think she did portray America as the whiney, sniveling mess of a girl she is with precision. However, every time she voiced Maxon's dialogue, I was reminded of how she made Cole sound in the Everneath series, and it made me sad that I wasn't re-listening to that series instead of this one. \n Things I loathed about this story: \n 1. America's immaturity when it came to love and other people's feelings. This girl was already planning to marry her first love, so one would expect her to have some sense of what is expected in an adult relationship. Instead, she strings her first love along while trying to decipher her feelings for the Crown Prince...while also stringing him along. America's constant back-and-forth -- during the ENTIRE series -- drove me nuts. I wanted both guys to find out about each other and say to hell with her. \n 2. The fact that America was essentially set up to become a martyr, that everything just perfectly aligned for her to be the perfect choice. Of course the King would be an evil blow-hard who would make things difficult for America at every turn. Of course he didn't want her to succeed. Of course because of her willingness to stand up to him, she became the people's favorite. Of course her own father would be a rebel himself, as would Caesar Flickerman, er, sorry...I mean Gavril whatever his name was. \n 3. All the drama and misunderstandings. Miscommunication in a story is one thing, but when you let every situation get out of hand because no one can apologize or admit fault, it becomes beyond annoying. Create tension in the story some other way because you aren't doing your characters any favors with this type of behavior. \n 4. How the story wrapped up so perfectly and yet left so much unsaid. First, so a life-or-death situation negates any betrayal felt five minutes before? Really, Maxon? Also, maybe I missed it in my haste to be done with this series for good, but what happened to the rebels? Was there even a funeral for Maxon's parents before the wedding? How did Aspen and Lucy come to be so close so quickly? So many unanswered questions! Grrr. \n 5. The fact that this contemporary story about a love triangle was trying to masquerade as a dystopian novel. Yeah, no. Throwing in a tyrant of a king and some rebels does not a dystopian story make. \n There's more but if I list all of my complaints, we'll be here all day. \n Redeeming factors: \n 1. I actually did appreciate that this story showcased that first love =/= true love. (Not always, anyway.) It may have taken the characters forever to get to this point, but they did...eventually. \n 2. The competition between the girls was fleshed out and eventually bordered on friendship, not just a group of girls all vying for one guy's attention. This, too, took awhile, but they got there. \n 3. Pretty covers with pretty girls in dresses that actually do have significance to the story. I have to admit, this series would look gorgeous on my shelf. If I had room for a series that I knew I'd never read again. \n I obviously didn't love these books, but I did find them at least somewhat entertaining, hence my ability to actually finish the series. I can understand the appeal of these books, but it's clear they were not written with someone like me in mind. I knew that from the get-go, and yet I still endeavored to make the most of this series. I have no one to blame but myself. \n GIF it to me straight: \n Yep. I couldn't have said it better myself.</td>
    </tr>
    <tr>
      <th>3138</th>
      <td>0d344a261de9ab42e62cf9b3b7c52cc4</td>
      <td>15820748</td>
      <td>This novella added nothing to the storyline, except to display just how clueless Maxon is when it comes to love and emotional ties. \n ______________________________ \n This review can also be found at The Starry-Eyed Revue. \n **I'm reviewing this series as a whole and there are likely to be spoilers below. If you haven't started the series yet, and you actually care about the outcome, I suggest you don't read any further.** \n I never had any intention of reading this series. After the fiasco with Wendy Darling and the author and her agent -- publicist? Idk...it's been so long that I can't remember -- I immediately shelved all the books by this author as "not in this lifetime". But I was perusing my library's audiobook selection, and I was bored and couldn't decide what to listen to next when I came across The Selection. And from there, a very bad idea was born. \n I didn't even know the final book was releasing soon when I picked up the first two books plus the novellas...that's how little attention I've paid to this series. And yet, I still embarked upon my mission to read the series in its entirety, knowing what a train wreck it would be. And it was: you know it's terrible, you're cringing the entire time, and yet you just. Can't. Look. Away. \n I don't enjoy reality TV...at all. I don't find it even mildly entertaining usually, but if I discover it already on the television when I sit down, I admit that it's very addictive and hard to look away from. Which is one of the reasons I detest it so much...what a time suck! Such was my experience with this series. I didn't want to keep reading about these girls and all their drama or that love triangle of doom. After all, I despise The Bachelor and this story is that show's premise in book form. But morbid curiosity just wouldn't let me pass up this series. \n I think one of the things that helped me maintain my sanity was the fact that I was listening to the audio instead of wasting my precious reading time. I've listened to several audiobooks narrated by Amy Rubinate, so I know that I like her narration style. I can't say that I was impressed with her performance this time around, though I think she did portray America as the whiney, sniveling mess of a girl she is with precision. However, every time she voiced Maxon's dialogue, I was reminded of how she made Cole sound in the Everneath series, and it made me sad that I wasn't re-listening to that series instead of this one. \n Things I loathed about this story: \n 1. America's immaturity when it came to love and other people's feelings. This girl was already planning to marry her first love, so one would expect her to have some sense of what is expected in an adult relationship. Instead, she strings her first love along while trying to decipher her feelings for the Crown Prince...while also stringing him along. America's constant back-and-forth -- during the ENTIRE series -- drove me nuts. I wanted both guys to find out about each other and say to hell with her. \n 2. The fact that America was essentially set up to become a martyr, that everything just perfectly aligned for her to be the perfect choice. Of course the King would be an evil blow-hard who would make things difficult for America at every turn. Of course he didn't want her to succeed. Of course because of her willingness to stand up to him, she became the people's favorite. Of course her own father would be a rebel himself, as would Caesar Flickerman, er, sorry...I mean Gavril whatever his name was. \n 3. All the drama and misunderstandings. Miscommunication in a story is one thing, but when you let every situation get out of hand because no one can apologize or admit fault, it becomes beyond annoying. Create tension in the story some other way because you aren't doing your characters any favors with this type of behavior. \n 4. How the story wrapped up so perfectly and yet left so much unsaid. First, so a life-or-death situation negates any betrayal felt five minutes before? Really, Maxon? Also, maybe I missed it in my haste to be done with this series for good, but what happened to the rebels? Was there even a funeral for Maxon's parents before the wedding? How did Aspen and Lucy come to be so close so quickly? So many unanswered questions! Grrr. \n 5. The fact that this contemporary story about a love triangle was trying to masquerade as a dystopian novel. Yeah, no. Throwing in a tyrant of a king and some rebels does not a dystopian story make. \n There's more but if I list all of my complaints, we'll be here all day. \n Redeeming factors: \n 1. I actually did appreciate that this story showcased that first love =/= true love. (Not always, anyway.) It may have taken the characters forever to get to this point, but they did...eventually. \n 2. The competition between the girls was fleshed out and eventually bordered on friendship, not just a group of girls all vying for one guy's attention. This, too, took awhile, but they got there. \n 3. Pretty covers with pretty girls in dresses that actually do have significance to the story. I have to admit, this series would look gorgeous on my shelf. If I had room for a series that I knew I'd never read again. \n I obviously didn't love these books, but I did find them at least somewhat entertaining, hence my ability to actually finish the series. I can understand the appeal of these books, but it's clear they were not written with someone like me in mind. I knew that from the get-go, and yet I still endeavored to make the most of this series. I have no one to blame but myself. \n GIF it to me straight: \n Yep. I couldn't have said it better myself.</td>
    </tr>
    <tr>
      <th>3139</th>
      <td>0d344a261de9ab42e62cf9b3b7c52cc4</td>
      <td>15844362</td>
      <td>This review can also be found at The Starry-Eyed Revue. \n **I'm reviewing this series as a whole and there are likely to be spoilers below. If you haven't started the series yet, and you actually care about the outcome, I suggest you don't read any further.** \n I never had any intention of reading this series. After the fiasco with Wendy Darling and the author and her agent -- publicist? Idk...it's been so long that I can't remember -- I immediately shelved all the books by this author as "not in this lifetime". But I was perusing my library's audiobook selection, and I was bored and couldn't decide what to listen to next when I came across The Selection. And from there, a very bad idea was born. \n I didn't even know the final book was releasing soon when I picked up the first two books plus the novellas...that's how little attention I've paid to this series. And yet, I still embarked upon my mission to read the series in its entirety, knowing what a train wreck it would be. And it was: you know it's terrible, you're cringing the entire time, and yet you just. Can't. Look. Away. \n I don't enjoy reality TV...at all. I don't find it even mildly entertaining usually, but if I discover it already on the television when I sit down, I admit that it's very addictive and hard to look away from. Which is one of the reasons I detest it so much...what a time suck! Such was my experience with this series. I didn't want to keep reading about these girls and all their drama or that love triangle of doom. After all, I despise The Bachelor and this story is that show's premise in book form. But morbid curiosity just wouldn't let me pass up this series. \n I think one of the things that helped me maintain my sanity was the fact that I was listening to the audio instead of wasting my precious reading time. I've listened to several audiobooks narrated by Amy Rubinate, so I know that I like her narration style. I can't say that I was impressed with her performance this time around, though I think she did portray America as the whiney, sniveling mess of a girl she is with precision. However, every time she voiced Maxon's dialogue, I was reminded of how she made Cole sound in the Everneath series, and it made me sad that I wasn't re-listening to that series instead of this one. \n Things I loathed about this story: \n 1. America's immaturity when it came to love and other people's feelings. This girl was already planning to marry her first love, so one would expect her to have some sense of what is expected in an adult relationship. Instead, she strings her first love along while trying to decipher her feelings for the Crown Prince...while also stringing him along. America's constant back-and-forth -- during the ENTIRE series -- drove me nuts. I wanted both guys to find out about each other and say to hell with her. \n 2. The fact that America was essentially set up to become a martyr, that everything just perfectly aligned for her to be the perfect choice. Of course the King would be an evil blow-hard who would make things difficult for America at every turn. Of course he didn't want her to succeed. Of course because of her willingness to stand up to him, she became the people's favorite. Of course her own father would be a rebel himself, as would Caesar Flickerman, er, sorry...I mean Gavril whatever his name was. \n 3. All the drama and misunderstandings. Miscommunication in a story is one thing, but when you let every situation get out of hand because no one can apologize or admit fault, it becomes beyond annoying. Create tension in the story some other way because you aren't doing your characters any favors with this type of behavior. \n 4. How the story wrapped up so perfectly and yet left so much unsaid. First, so a life-or-death situation negates any betrayal felt five minutes before? Really, Maxon? Also, maybe I missed it in my haste to be done with this series for good, but what happened to the rebels? Was there even a funeral for Maxon's parents before the wedding? How did Aspen and Lucy come to be so close so quickly? So many unanswered questions! Grrr. \n 5. The fact that this contemporary story about a love triangle was trying to masquerade as a dystopian novel. Yeah, no. Throwing in a tyrant of a king and some rebels does not a dystopian story make. \n There's more but if I list all of my complaints, we'll be here all day. \n Redeeming factors: \n 1. I actually did appreciate that this story showcased that first love =/= true love. (Not always, anyway.) It may have taken the characters forever to get to this point, but they did...eventually. \n 2. The competition between the girls was fleshed out and eventually bordered on friendship, not just a group of girls all vying for one guy's attention. This, too, took awhile, but they got there. \n 3. Pretty covers with pretty girls in dresses that actually do have significance to the story. I have to admit, this series would look gorgeous on my shelf. If I had room for a series that I knew I'd never read again. \n I obviously didn't love these books, but I did find them at least somewhat entertaining, hence my ability to actually finish the series. I can understand the appeal of these books, but it's clear they were not written with someone like me in mind. I knew that from the get-go, and yet I still endeavored to make the most of this series. I have no one to blame but myself. \n GIF it to me straight: \n Yep. I couldn't have said it better myself.</td>
    </tr>
    <tr>
      <th>3402</th>
      <td>0d344a261de9ab42e62cf9b3b7c52cc4</td>
      <td>14748094</td>
      <td>Why am I doing this to myself? \n ----------- \n Listened to a library audio version. Horrible but addictive. Like a terrible, terrible soap opera. The kind even my own mother wouldn't watch. \n The plan is to see it through to the end and then review the trilogy together. I'd rather stab myself in the eye, but for some reason, I feel the urge to rant about these, even though I had plenty of warning beforehand. *le sigh* \n ______________________________ \n This review can also be found at The Starry-Eyed Revue. \n **I'm reviewing this series as a whole and there are likely to be spoilers below. If you haven't started the series yet, and you actually care about the outcome, I suggest you don't read any further.** \n I never had any intention of reading this series. After the fiasco with Wendy Darling and the author and her agent -- publicist? Idk...it's been so long that I can't remember -- I immediately shelved all the books by this author as "not in this lifetime". But I was perusing my library's audiobook selection, and I was bored and couldn't decide what to listen to next when I came across The Selection. And from there, a very bad idea was born. \n I didn't even know the final book was releasing soon when I picked up the first two books plus the novellas...that's how little attention I've paid to this series. And yet, I still embarked upon my mission to read the series in its entirety, knowing what a train wreck it would be. And it was: you know it's terrible, you're cringing the entire time, and yet you just. Can't. Look. Away. \n I don't enjoy reality TV...at all. I don't find it even mildly entertaining usually, but if I discover it already on the television when I sit down, I admit that it's very addictive and hard to look away from. Which is one of the reasons I detest it so much...what a time suck! Such was my experience with this series. I didn't want to keep reading about these girls and all their drama or that love triangle of doom. After all, I despise The Bachelor and this story is that show's premise in book form. But morbid curiosity just wouldn't let me pass up this series. \n I think one of the things that helped me maintain my sanity was the fact that I was listening to the audio instead of wasting my precious reading time. I've listened to several audiobooks narrated by Amy Rubinate, so I know that I like her narration style. I can't say that I was impressed with her performance this time around, though I think she did portray America as the whiney, sniveling mess of a girl she is with precision. However, every time she voiced Maxon's dialogue, I was reminded of how she made Cole sound in the Everneath series, and it made me sad that I wasn't re-listening to that series instead of this one. \n Things I loathed about this story: \n 1. America's immaturity when it came to love and other people's feelings. This girl was already planning to marry her first love, so one would expect her to have some sense of what is expected in an adult relationship. Instead, she strings her first love along while trying to decipher her feelings for the Crown Prince...while also stringing him along. America's constant back-and-forth -- during the ENTIRE series -- drove me nuts. I wanted both guys to find out about each other and say to hell with her. \n 2. The fact that America was essentially set up to become a martyr, that everything just perfectly aligned for her to be the perfect choice. Of course the King would be an evil blow-hard who would make things difficult for America at every turn. Of course he didn't want her to succeed. Of course because of her willingness to stand up to him, she became the people's favorite. Of course her own father would be a rebel himself, as would Caesar Flickerman, er, sorry...I mean Gavril whatever his name was. \n 3. All the drama and misunderstandings. Miscommunication in a story is one thing, but when you let every situation get out of hand because no one can apologize or admit fault, it becomes beyond annoying. Create tension in the story some other way because you aren't doing your characters any favors with this type of behavior. \n 4. How the story wrapped up so perfectly and yet left so much unsaid. First, so a life-or-death situation negates any betrayal felt five minutes before? Really, Maxon? Also, maybe I missed it in my haste to be done with this series for good, but what happened to the rebels? Was there even a funeral for Maxon's parents before the wedding? How did Aspen and Lucy come to be so close so quickly? So many unanswered questions! Grrr. \n 5. The fact that this contemporary story about a love triangle was trying to masquerade as a dystopian novel. Yeah, no. Throwing in a tyrant of a king and some rebels does not a dystopian story make. \n There's more but if I list all of my complaints, we'll be here all day. \n Redeeming factors: \n 1. I actually did appreciate that this story showcased that first love =/= true love. (Not always, anyway.) It may have taken the characters forever to get to this point, but they did...eventually. \n 2. The competition between the girls was fleshed out and eventually bordered on friendship, not just a group of girls all vying for one guy's attention. This, too, took awhile, but they got there. \n 3. Pretty covers with pretty girls in dresses that actually do have significance to the story. I have to admit, this series would look gorgeous on my shelf. If I had room for a series that I knew I'd never read again. \n I obviously didn't love these books, but I did find them at least somewhat entertaining, hence my ability to actually finish the series. I can understand the appeal of these books, but it's clear they were not written with someone like me in mind. I knew that from the get-go, and yet I still endeavored to make the most of this series. I have no one to blame but myself. \n GIF it to me straight: \n Yep. I couldn't have said it better myself.</td>
    </tr>
    <tr>
      <th>3463</th>
      <td>0d344a261de9ab42e62cf9b3b7c52cc4</td>
      <td>10507293</td>
      <td>Listened to a library audio version. Horrible but addictive. Like a terrible, terrible soap opera. The kind even my own mother wouldn't watch. \n The plan is to see it through to the end and then review the trilogy together. I'd rather stab myself in the eye, but for some reason, I feel the urge to rant about these, even though I had plenty of warning beforehand. *le sigh* \n ______________________________ \n This review can also be found at The Starry-Eyed Revue. \n **I'm reviewing this series as a whole and there are likely to be spoilers below. If you haven't started the series yet, and you actually care about the outcome, I suggest you don't read any further.** \n I never had any intention of reading this series. After the fiasco with Wendy Darling and the author and her agent -- publicist? Idk...it's been so long that I can't remember -- I immediately shelved all the books by this author as "not in this lifetime". But I was perusing my library's audiobook selection, and I was bored and couldn't decide what to listen to next when I came across The Selection. And from there, a very bad idea was born. \n I didn't even know the final book was releasing soon when I picked up the first two books plus the novellas...that's how little attention I've paid to this series. And yet, I still embarked upon my mission to read the series in its entirety, knowing what a train wreck it would be. And it was: you know it's terrible, you're cringing the entire time, and yet you just. Can't. Look. Away. \n I don't enjoy reality TV...at all. I don't find it even mildly entertaining usually, but if I discover it already on the television when I sit down, I admit that it's very addictive and hard to look away from. Which is one of the reasons I detest it so much...what a time suck! Such was my experience with this series. I didn't want to keep reading about these girls and all their drama or that love triangle of doom. After all, I despise The Bachelor and this story is that show's premise in book form. But morbid curiosity just wouldn't let me pass up this series. \n I think one of the things that helped me maintain my sanity was the fact that I was listening to the audio instead of wasting my precious reading time. I've listened to several audiobooks narrated by Amy Rubinate, so I know that I like her narration style. I can't say that I was impressed with her performance this time around, though I think she did portray America as the whiney, sniveling mess of a girl she is with precision. However, every time she voiced Maxon's dialogue, I was reminded of how she made Cole sound in the Everneath series, and it made me sad that I wasn't re-listening to that series instead of this one. \n Things I loathed about this story: \n 1. America's immaturity when it came to love and other people's feelings. This girl was already planning to marry her first love, so one would expect her to have some sense of what is expected in an adult relationship. Instead, she strings her first love along while trying to decipher her feelings for the Crown Prince...while also stringing him along. America's constant back-and-forth -- during the ENTIRE series -- drove me nuts. I wanted both guys to find out about each other and say to hell with her. \n 2. The fact that America was essentially set up to become a martyr, that everything just perfectly aligned for her to be the perfect choice. Of course the King would be an evil blow-hard who would make things difficult for America at every turn. Of course he didn't want her to succeed. Of course because of her willingness to stand up to him, she became the people's favorite. Of course her own father would be a rebel himself, as would Caesar Flickerman, er, sorry...I mean Gavril whatever his name was. \n 3. All the drama and misunderstandings. Miscommunication in a story is one thing, but when you let every situation get out of hand because no one can apologize or admit fault, it becomes beyond annoying. Create tension in the story some other way because you aren't doing your characters any favors with this type of behavior. \n 4. How the story wrapped up so perfectly and yet left so much unsaid. First, so a life-or-death situation negates any betrayal felt five minutes before? Really, Maxon? Also, maybe I missed it in my haste to be done with this series for good, but what happened to the rebels? Was there even a funeral for Maxon's parents before the wedding? How did Aspen and Lucy come to be so close so quickly? So many unanswered questions! Grrr. \n 5. The fact that this contemporary story about a love triangle was trying to masquerade as a dystopian novel. Yeah, no. Throwing in a tyrant of a king and some rebels does not a dystopian story make. \n There's more but if I list all of my complaints, we'll be here all day. \n Redeeming factors: \n 1. I actually did appreciate that this story showcased that first love =/= true love. (Not always, anyway.) It may have taken the characters forever to get to this point, but they did...eventually. \n 2. The competition between the girls was fleshed out and eventually bordered on friendship, not just a group of girls all vying for one guy's attention. This, too, took awhile, but they got there. \n 3. Pretty covers with pretty girls in dresses that actually do have significance to the story. I have to admit, this series would look gorgeous on my shelf. If I had room for a series that I knew I'd never read again. \n I obviously didn't love these books, but I did find them at least somewhat entertaining, hence my ability to actually finish the series. I can understand the appeal of these books, but it's clear they were not written with someone like me in mind. I knew that from the get-go, and yet I still endeavored to make the most of this series. I have no one to blame but myself. \n GIF it to me straight: \n Yep. I couldn't have said it better myself.</td>
    </tr>
  </tbody>
</table>
</div>

```python
print(f'How many distinct reviews are left after merging identical and near-duplicated texts?\n{df_goodreads['verified_group'].nunique()} out of {df_goodreads.shape[0]}')
```

```text
How many distinct reviews are left after merging identical and near-duplicated texts?
1357868 out of 1378033
```

### Conclusion
**Locality-Sensitive Hashing** is an effective technique for detecting near-duplicates, providing a balance between reliability and computational efficiency, although as any probabilistic approach it may produce some false positives and false negatives.

It can be adjusted based on specific tasks, hardware resources, and precision needs. In recommendation systems, it is particularly useful for merging duplicate items, which helps create a more compact interaction matrix. This addresses memory constraints and can be used to reduce sparsity, ultimately leading to better performance.

Additionally, Locality-Sensitive Hashing could be used in anomaly detection or to help exclude certain interactions from training recommendation systems, particularly when there are suspicions of fraudulent reviews or comment campaigns used for advertising or spreading misinformation.
