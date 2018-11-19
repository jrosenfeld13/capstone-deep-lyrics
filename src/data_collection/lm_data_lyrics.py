import pandas as pd
import numpy as np
import nltk.tokenize
import itertools
from pathlib import Path

from sklearn.model_selection import train_test_split

from fastai import *
from fastai.text import *

import argparse

def tokenize_lyrics(lyrics):
    tk = nltk.tokenize.LineTokenizer(blanklines='keep')
    tokd = tk.tokenize(lyrics)
    
    re_tk = nltk.tokenize.RegexpTokenizer(r'\[[^\]]+\]|\w+|[\d\.,]+|\S+',
                                          discard_empty=False)
    re_tokd = re_tk.tokenize_sents(tokd)
    
<<<<<<< HEAD
    [s.insert(0, f'xBOL {line_num+1}') for line_num, s in enumerate(re_tokd)] # insert start token for each line
=======
    [s.insert(0, f'xBOL {line_num + 1}') for line_num, s in enumerate(re_tokd)] # insert start token for each line
>>>>>>> 7506a16b7e7bb732d299999841a82baae623c87e
    [s.append('xEOL') for s in re_tokd] # append end token for each line
    
    flat = list(itertools.chain(*re_tokd))
    flat.insert(0, 'xBOS')
    flat.append('xEOS')
    # lower case and de-space
    flat = [w.lower().replace(' ', '-') for w in flat]
    return flat

def main(data_out='../../data/models/default_model'):
    
    data_out = Path(data_out)
    data_out.mkdir(exist_ok=True)
    
    # load scraped data
    df = pd.read_csv('https://storage.googleapis.com/w210-capstone/data/lyrics-valid.csv',
                     header=None, escapechar='\\',
                     names=['msd_id', 'lyrics'])

    # only keep lyrics with length < 5000
    df = df[df.lyrics.str.len() < 5000]
    df['tokd'] = df.lyrics.apply(tokenize_lyrics)
    df['tokd_len'] = df.tokd.apply(len)

    # split train/test
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=2018)
    
    # tokens
    tokens = np.array(df_train.tokd)
    np.save(data_out/'train_tok.npy', tokens)
    
    tokens = np.array(df_test.tokd)
    np.save(data_out/'valid_tok.npy', tokens)
    
    # create data bunch and save "tmp" files
    data_lm = TextLMDataBunch.from_tokens(data_out,
                                          bs=128,
                                          max_vocab=10000)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                    help='out path to store train test splits',
                    default='../../data/models/default_model')
    
    args = parser.parse_args()
    
    main(args.o)



