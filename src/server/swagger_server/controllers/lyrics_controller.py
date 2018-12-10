import connexion
import six
import json
from pathlib import Path
import pickle
import os

from swagger_server.models.lyrics_request import LyricsRequest  # noqa: E501
from swagger_server import util

from ..src.nlp.generate_lyrics import DeepLyric
from ..src.nlp.evaluate_lyrics import Evaluator

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
    MODEL_ROOT = Path('../../../../models')
    
    assert body['model_type'] in ['language', 'multimodal'], "Model does not exist."
    if body['model_type'] == 'multimodal':
        MODEL_NAME = '4.3-MM-108k-post-genre-song_title'
        
        # file load (preprocessor for audio)
        preproc_file = MODEL_ROOT/f'{MODEL_NAME}_preprocessor.pkl'
        with open(preproc_file, 'rb') as f:
            preproc = pickle.load(f)
    else:
        MODEL_NAME = '4.2-LM-108k-lines-genre-song_title'
        
    model_file = MODEL_ROOT/f'{MODEL_NAME}_architecture.pkl'
    itos_file = MODEL_ROOT/f'{MODEL_NAME}_itos.pkl'
    # preproc_file = Path('../../../../models/'/f'{model_name}_preprocessor.pkl')

    with open(model_file, 'rb') as f:
        architecture = pickle.load(f)
    with open(itos_file, 'rb') as f:
        itos = pickle.load(f)
        
    deep_lyric = DeepLyric(architecture=architecture,
                           itos=itos,
                           model_type=body['model_type'],
                           preprocessor=None,
                           model_name=MODEL_NAME)
    
    ### PARSE PARAMS FROM WEB
    genre = body['genres']
    if len(genre) > 1:
        genre = genre[0]
    
    # #title
    # title = body['title']
    #
    # deep_lyric.set_config('seed_text', f'xbos xgenre {genre} xtitle {title}')
    # deep_lyric.set_config('max_len', 130)
    # deep_lyric.set_config('context_length', 70)
    # deep_lyric.set_config('beam_width', 3)
    # deep_lyric.set_config('top_k', 5)
    # deep_lyric.set_config('temperature', 1.45)
    
    deep_lyric.generate_text()
    
    #out = deep_lyric.save_json(out=True)
    out = deep_lyric.save_json(format_lyrics=True)
    return json.dump(out)
