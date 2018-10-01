import connexion
import six

from swagger_server.models.lyrics_request import LyricsRequest  # noqa: E501
from swagger_server import util

import sys
sys.path.append('../../src') #allow relative import of nlp packages
from nlp.generate_text import generate_text

def generate_lyrics(body):  # noqa: E501
    """Request lyric generation.

    Request lyric generation. # noqa: E501

    :param body: LyricsRequest object that defines parameters of the lyrics desired.
    :type body: dict | bytes

    :rtype: str
    """
    if connexion.request.is_json:
        body = LyricsRequest.from_dict(connexion.request.get_json())  # noqa: E501
    
    lyrics = generate_text(infile='../../data/models/trigram-weights.pkl') #infile eventually needs to connect to a cloud store url
    return lyrics
