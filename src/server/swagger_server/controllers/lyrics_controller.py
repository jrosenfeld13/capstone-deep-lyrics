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
from ..src.nlp.preprocessor_util import log_features, bin_tempo, bin_time_signature, to_string


def generate_lyrics(body):  # noqa: E501
    """Request lyric generation.

    Request lyric generation. # noqa: E501

    :param body: LyricsRequest object that defines parameters of the lyrics desired.
    :type body: dict | bytes

    :rtype: str
    """
    if connexion.request.is_json:
        body = connexion.request.get_json()  # noqa: E501

    ###ESTABLISH MODEL TYPE
    body['model_type'] = 'language' ##HARDCODE LANGUAGE MODEL FOR NOW
    MODEL = '4.2-LM-108k-lines-genre-song_title'

    if body['model_type'] == 'multimodal':
        MODEL = '4.3-MM-108k-post-genre-song_title'
    
    ### LOAD MODEL AND CREATE DEEPLYRIC OBJECT  
    #load from local (needs this relative path workaround)
    ARCHITCTURE_FILE = os.path.join(os.path.dirname(__file__), f'../../../../data/models/{MODEL}/models/{MODEL}_architecture.pkl')
    ITOS_FILE = os.path.join(os.path.dirname(__file__), f'../../../../data/models/{MODEL}/tmp/itos.pkl')
    PREPROCESSOR_FILE = os.path.join(os.path.dirname(__file__), f'../../../../data/models/{MODEL}/{MODEL}_preprocessor.pkl')

    with open(ARCHITCTURE_FILE, 'rb') as f:
        architecture = pickle.load(f)
    with open(ITOS_FILE, 'rb') as f:
        itos = pickle.load(f)
    
    preprocessor=None
    if body['model_type'] == 'multimodal':
        with open(PREPROCESSOR_FILE, 'rb') as f:
            preprocessor = pickle.load(f)
        
    deep_lyric = DeepLyric(architecture, itos, model_type=body['model_type'], preprocessor=preprocessor)
    
    ### PARSE PARAMS FROM WEB
    #genre
    genre = body['genres']
    if len(genre) > 1:
        genre = genre[0]      
    
    #title
    title = body['title']
    
    deep_lyric.set_config('seed_text', f'xbos xgenre {genre} xtitle {title}')
    deep_lyric.set_config('max_len', 130)
    deep_lyric.set_config('context_length', 70)
    deep_lyric.set_config('beam_width', 3)
    deep_lyric.set_config('top_k', 5)
    deep_lyric.set_config('temperature', 1.45)
    
    deep_lyric.generate_text()
    
    #out = deep_lyric.save_json(out=True)
    out = deep_lyric.pretty_format()
    return json.dumps(out)
