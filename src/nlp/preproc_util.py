import pandas as pd
import numpy as np

def log_features(X):
    return np.log(X)

def bin_tempo(X):
    '''
    ref: https://en.wikipedia.org/wiki/Tempo#Italian_tempo_markings
    These are rough loosely based on tempo markings above
    Have considered both classical forms of music and popular
    '''
    assert X.shape[1] == 1, "Only 1 column can be binned"
    bins = [0, 60, 76, 108, 120, 156, 176, 200, 500]
    return pd.DataFrame(pd.cut(X.iloc[:,0], bins=bins))

def bin_time_signature(X):
    assert X.shape[1] == 1, "Only 1 column can be binned"
    X['time_signature_bin'] = "Other Signature"
    X.loc[X['time_signature'] == 4, 'time_signature_bin'] = '4/4 Signature'
    X.loc[X['time_signature'] == 3, 'time_signature_bin'] = '3/4 Signature'
    return X[['time_signature_bin']]

def to_string(X):
    return X.astype('str')