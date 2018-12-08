import connexion
import six
import json
from pathlib import Path
import pickle

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

    MODEL = '4.2-LM-108k-lines-genre-song_title'

    ### LOAD MODEL AND CREATE DEEPLYRIC OBJECT
    #load from google cloudstore
    deep_lyric = DeepLyric(MODEL, model_type='language', model_name=MODEL)
    
    #load from local (needs relative path workaround)
    #MODEL_PATH = Path(f'../../../../data/models/{MODEL}')
    #with open(MODEL_PATH/f'models/{MODEL}_architecture.pkl') as f:
    #    architecture = pickle.loads(f)
    #with open(MODEL_PATH/f'tmp/itos.pkl') as f:
    #    itos = pickle.loads(f)
    #deep_lyric = DeepLyric(architecture, itos, model_type='language')
    
    ### PARSE PARAMS FROM WEB
    #genre
    genre = body['genres']
    if len(genre) > 1:
        genre = genre[0] # first for now      
    
    #title
    title = body['title']
    
    deep_lyric.set_config('seed_text', f'xbos xgenre {genre} xtitle {title} xbol-1')
    deep_lyric.generate_text()
    
    #out = deep_lyric.save_json(out=True)
    out = deep_lyric.pretty_format()
    return json.dumps(out)
