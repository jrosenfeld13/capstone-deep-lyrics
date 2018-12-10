import connexion
import six

from swagger_server.models.lyrics_request import LyricsRequest  # noqa: E501
from swagger_server import util


def generate_lyrics(body):  # noqa: E501
    """Request lyric generation.

    Request lyric generation. # noqa: E501

    :param body: LyricsRequest object that defines parameters of the lyrics desired.
    :type body: dict | bytes

    :rtype: str
    """
    if connexion.request.is_json:
        body = LyricsRequest.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'
