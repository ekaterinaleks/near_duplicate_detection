# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: duplicates-lsh
#     language: python
#     name: python3
# ---

# # Near-duplicate detection
# ### Why look for near-duplicates
#
# In case of recommendation system, the same algorithm could be applied to the recommended items themselves to detect near-duplicated posts or items. On one hand, it would allow to optimize the amount of memory needed for processing.

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

# ### Get the data
# I will be using a dataset of reviews from Goodreads 📚 taken from [Kaggle 2022 competition](https://www.kaggle.com/competitions/goodreads-books-reviews-290312) originally downloaded from UCSD Book Graph. This dataset wasn't deduplicated as its later version currently available at [Goodreads Book Graph Datasets](https://mengtingwan.github.io/data/goodreads).

#download with Kaggle API
# ! kaggle competitions download -c goodreads-books-reviews-290312

#read into df only two files from the zip 
with zipfile.ZipFile('goodreads-books-reviews-290312.zip', 'r') as folder:
    with folder.open('goodreads_test.csv') as file:
        df_goodreads_test = pd.read_csv(file)
    with folder.open('goodreads_train.csv') as file:
        df_goodreads_train = pd.read_csv(file)

#modify and combine the files for the near-duplicate detection task
df_goodreads_train.drop('rating', axis=1, inplace=True)
df_goodreads = pd.concat([df_goodreads_test, df_goodreads_train])

# ### Check for exact duplicates

print(f'Are there any complete duplicates with identical data in all columns?\n{df_goodreads.duplicated().any()}')

print(f'Are there any duplicated review ids?\n{df_goodreads.duplicated(['review_id']).any()}')

print(f'How many duplicated review texts are there?\n{df_goodreads.duplicated(['review_text'], keep=False).sum()}') 

print(f'Are there any duplicated review texts from the same user about the same book?\n{df_goodreads.duplicated(['review_text', 'book_id','user_id']).any()}')

# ### Detect Near-Duplicates
# Although there are no complete duplicates for reviews submitted by the same user about the same book, [Goodreads' users do complain about these cases](https://www.goodreads.com/topic/show/22591024-duplicate-reviews). It is very probably that this problem stems from reposted reviews with slight wording, spelling and/or punctuation changes that can't be detected as easily as complete duplicates.
#
# Apart from deduplication purposes as optimization problem, it could be used as anomaly detection as well. It could be useful to be able to detect users that leave similar reviews for different items or different users leaving exactly the same comment about specific items for moderation purposes or consider excluding corresponding interactions from recommendation system training as unreliable.
#

lsh_instance = lsh.LshHelper()

#preprocess reviews to prepare for taking minhash
df_goodreads['clean_review_text'] = df_goodreads['review_text'].parallel_apply(lsh_instance.preprocess)

df_goodreads.index = df_goodreads['review_id']

try:
    df_goodreads = pd.read_pickle('goodreads_minhash.pkl')
except:
    df_goodreads['minhash'] = None

df = pd.read_pickle('goodreads_minhash.pkl')

df_goodreads = df.query('minhash.notna()')

it = iter(df_goodreads.query('minhash.isna()').index)
chunk = list(itertools.islice(it, 4096))

# %%time
while chunk:
    df_goodreads.loc[chunk, 'minhash'] = df_goodreads.loc[chunk, 'review_text'].parallel_apply(lsh_instance.take_minhash)
    chunk = list(itertools.islice(it, 4096))
    df_goodreads.to_pickle('goodreads_minhash.pkl')
    

#instantiate Locality Sensitive Hashing object
lsh_object = lsh_instance.instantiate_lsh_object()

#populate the LSH object
for _index, minhash in df_goodreads['minhash'].items():
    lsh_object.insert(_index, pickle.loads(minhash), check_duplication=False)

#check for near-duplicates
near_duplicates_dict = {}
for _index, minhash in df_goodreads['minhash'].items():
    dups = lsh_object.query(pickle.loads(minhash))
    near_duplicates_dict[_index] = dups

df_doubles = pd.json_normalize(near_duplicates_dict).transpose().rename(columns={0:'set_of_doubles'})

df_doubles['review_id'] = df_doubles.index

df_doubles['set_of_doubles'] = df_doubles['set_of_doubles'].apply(sorted).apply(tuple)

df_doubles.reset_index(drop=True, inplace=True)

df_goodreads.reset_index(drop=True, inplace=True)

df_goodreads = df_goodreads.merge(df_doubles, how='inner', on='review_id')

df_goodreads['group'] = df_goodreads.groupby(['user_id', 'set_of_doubles']).ngroup()

df_goodreads['number'] = df_goodreads.groupby('group')['group'].transform(lambda x: len(x))

df_goodreads.query('number > 1').sort_values('group').head(10)
