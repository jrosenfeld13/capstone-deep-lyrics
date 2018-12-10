import connexion
import six
import json
from pathlib import Path
import pickle
import os
from pandas.io.json import json_normalize
import pandas as pd
import numpy as np


from swagger_server.models.lyrics_request import LyricsRequest  # noqa: E501
from swagger_server import util

from ..src.nlp.generate_lyrics import DeepLyric
from ..src.nlp.evaluate_lyrics import Evaluator
from .src.nlp.preproc_util import log_features, bin_tempo, bin_time_signature, to_string

import sys
sys.path.insert(0, '/home/ubuntu/capstone-deep-lyrics')
import src
#from .src.data_collection.multimodal_data import AudioDataset, MultimodalDataLoader, map_weights
#from .src.nlp.neural_model import MultiLinearDecoder, MultiModalPostRNN

from pandas.io.json import json_normalize

def generate_lyrics(body):  # noqa: E501
    """Request lyric generation.

    Request lyric generation. # noqa: E501

    :param body: LyricsRequest object that defines parameters of the lyrics desired.
    :type body: dict | bytes

    :rtype: str
    """
    if connexion.request.is_json:
        body = connexion.request.get_json()  # noqa: E501

    # Settings to load DeepLyric Instance
    FILE_DIR = Path(os.path.dirname(__file__))
    MODEL_ROOT = Path('../../../../models')
    MODEL_ROOT = FILE_DIR/MODEL_ROOT
    
    assert body['model_type'] in ['language', 'multimodal'], "Model does not exist."
    if body['model_type'] == 'multimodal':
        MODEL_NAME = '4.3-MM-108k-post-genre-song_title'
        
        # file load (preprocessor for audio)
        preproc_file = MODEL_ROOT/f'{MODEL_NAME}_preprocessor.pkl'
        with open(preproc_file, 'rb') as f:
            preproc = pickle.load(f)
    else:
        MODEL_NAME = '4.2-LM-108k-lines-genre-song_title'
        preproc=None
        
    model_file = MODEL_ROOT/f'{MODEL_NAME}_architecture.pkl'
    itos_file = MODEL_ROOT/f'{MODEL_NAME}_itos.pkl'
    # preproc_file = Path('../../../../models/'/f'{model_name}_preprocessor.pkl')

    with open(model_file, 'rb') as f:
        architecture = pickle.load(f)
    with open(itos_file, 'rb') as f:
        itos = pickle.load(f)
        
    deep_lyric = DeepLyric(model=architecture,
                           itos=itos,
                           model_type=body['model_type'],
                           preprocessor=preproc,
                           model_name=MODEL_NAME)
    
    ### PARSE PARAMS FROM WEB
    genre = body['genres'][0]
    
    # audio features
    df_audio = json_normalize(body)
    df_audio = df_audio[['loudness', 'duration', 'key',
                         'mode', 'time_signature', 'tempo']]
    df_audio['duration'] = df_audio['duration'] / 1000
    
    # convert to np to make serializable
    df_audio = list(df_audio.iloc[0,:])
    deep_lyric.set_config('audio', df_audio)
    
    # other features
    for k in ['seed_text', 'title', 'beam_width',
              'top_k', 'context_length', 'max_len',
              'multinomial', 'temperature']:
        try:
            deep_lyric.set_config(k, body[k])
        except:
            continue
            
    if genre:
        deep_lyric.set_config('genre', genre)

    
    # generate and evaluate
    evaluator = Evaluator(deep_lyric)
    evaluator.get_lyric()
    evaluator.evaluate()
    out = evaluator.save_json(out=True, format_lyrics=True)
    
    #out = deep_lyric.save_json(out=True)
    # out = deep_lyric.save_json(out=True, format_lyrics=True)
    #return json.dumps(out)
    return json.dumps(out)
