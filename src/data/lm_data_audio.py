import pandas as pd
import numpy as np
import nltk.tokenize
import itertools
from pathlib import Path

from sklearn.model_selection import train_test_split

import argparse

def main(data_out='../../data/interim/msd-aggregate'):
    
    data_out = Path(data_out)
    data_out.mkdir(exist_ok=True)
    
    # need lyrics data to match keys
    df_lyrics = pd.read_csv('https://storage.googleapis.com/w210-capstone/data/lyrics-valid.csv',
                             header=None, escapechar='\\',
                             names=['msd_id', 'lyrics'])
    # only keep lyrics < 5000
    df_lyrics = df_lyrics[df_lyrics.lyrics.str.len() < 5000]
    
    # take keys from lyrics
    df_key = pd.DataFrame(df_lyrics.msd_id)
    df_key.rename(columns={'msd_id': 'track_id'}, inplace=True)
    
    # bring in audio data
    msd_aggregate = 'https://storage.googleapis.com/w210-capstone/data/msd-aggregate.csv'
    df = pd.read_csv(msd_aggregate)
    
    # match audio to keys from lyrics
    df_audio = pd.merge(df_key, df, how='inner', on='track_id')
    
    # split data using same seed as lm_data_lyrics.py
    df_train, df_test = train_test_split(df_audio,
                                         test_size=0.2,
                                         random_state=2018)
    
    df_train.to_csv(data_out/'msd-aggregate-train.csv', index=False)
    df_test.to_csv(data_out/'msd-aggregate-valid.csv', index=False)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                    help='out path to store train test splits',
                    default='../../data/interim/msd-aggregate')
    
    args = parser.parse_args()
    
    main(args.o)