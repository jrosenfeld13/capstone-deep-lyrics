import connexion
import six
import json

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
   
    #body = {
    #  "danceability": 0.6027456183070403,
    #  "duration_ms": 0.5962133916683182,
    #  "energy": 0.14658129805029452,
    #  "genres": [
    #    "folk",
    #    "gipsy"
    #  ],
    #  "mode": 0,
    #  "tags": [
    #    "tags",
    #    "tags"
    #  ],
    #  "tempo": 0.5637376656633328,
    #  "title": "title",
    #  "year": 1465
    #}
    
    # bring in model -- this should be moved outside of this function eventually
    # and we should load from disk
    MODEL = '4.2-LM-108k-lines-genre-song_title'
    deep_lyric = DeepLyric(MODEL, model_type='language', model_name=MODEL)
    
    # params from web
    genre = body['genres']
    if len(genre) > 1:
        genre = genre[0] # first for now
    #title = body['title']

    deep_lyric.set_config('seed_text', f'xbos xgenre {genre} xtitle')
    deep_lyric.generate_text()
    
    #out = deep_lyric.save_json(out=True)
    out = deep_lyric.pretty_format()
    return json.dumps(out)
