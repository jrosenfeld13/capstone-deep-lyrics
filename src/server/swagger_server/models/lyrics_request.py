# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server import util


class LyricsRequest(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, mode: int=None, key: int=None, time_signature: int=None, danceability: float=None, energy: float=None, duration_ms: float=None, loudness: float=None, tempo: float=None, title: str=None, genres: List[str]=None, beam_width: int=None, top_k: int=None, context_length: int=None, max_len: int=None, context: str=None):  # noqa: E501
        """LyricsRequest - a model defined in Swagger

        :param mode: The mode of this LyricsRequest.  # noqa: E501
        :type mode: int
        :param key: The key of this LyricsRequest.  # noqa: E501
        :type key: int
        :param time_signature: The time_signature of this LyricsRequest.  # noqa: E501
        :type time_signature: int
        :param danceability: The danceability of this LyricsRequest.  # noqa: E501
        :type danceability: float
        :param energy: The energy of this LyricsRequest.  # noqa: E501
        :type energy: float
        :param duration_ms: The duration_ms of this LyricsRequest.  # noqa: E501
        :type duration_ms: float
        :param loudness: The loudness of this LyricsRequest.  # noqa: E501
        :type loudness: float
        :param tempo: The tempo of this LyricsRequest.  # noqa: E501
        :type tempo: float
        :param title: The title of this LyricsRequest.  # noqa: E501
        :type title: str
        :param genres: The genres of this LyricsRequest.  # noqa: E501
        :type genres: List[str]
        :param beam_width: The beam_width of this LyricsRequest.  # noqa: E501
        :type beam_width: int
        :param top_k: The top_k of this LyricsRequest.  # noqa: E501
        :type top_k: int
        :param context_length: The context_length of this LyricsRequest.  # noqa: E501
        :type context_length: int
        :param max_len: The max_len of this LyricsRequest.  # noqa: E501
        :type max_len: int
        :param context: The context of this LyricsRequest.  # noqa: E501
        :type context: str
        """
        self.swagger_types = {
            'mode': int,
            'key': int,
            'time_signature': int,
            'danceability': float,
            'energy': float,
            'duration_ms': float,
            'loudness': float,
            'tempo': float,
            'title': str,
            'genres': List[str],
            'beam_width': int,
            'top_k': int,
            'context_length': int,
            'max_len': int,
            'context': str
        }

        self.attribute_map = {
            'mode': 'mode',
            'key': 'key',
            'time_signature': 'time_signature',
            'danceability': 'danceability',
            'energy': 'energy',
            'duration_ms': 'duration_ms',
            'loudness': 'loudness',
            'tempo': 'tempo',
            'title': 'title',
            'genres': 'genres',
            'beam_width': 'beam_width',
            'top_k': 'top_k',
            'context_length': 'context_length',
            'max_len': 'max_len',
            'context': 'context'
        }

        self._mode = mode
        self._key = key
        self._time_signature = time_signature
        self._danceability = danceability
        self._energy = energy
        self._duration_ms = duration_ms
        self._loudness = loudness
        self._tempo = tempo
        self._title = title
        self._genres = genres
        self._beam_width = beam_width
        self._top_k = top_k
        self._context_length = context_length
        self._max_len = max_len
        self._context = context

    @classmethod
    def from_dict(cls, dikt) -> 'LyricsRequest':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The LyricsRequest of this LyricsRequest.  # noqa: E501
        :rtype: LyricsRequest
        """
        return util.deserialize_model(dikt, cls)

    @property
    def mode(self) -> int:
        """Gets the mode of this LyricsRequest.

        Song mode. 0 if minor, 1 if major.  # noqa: E501

        :return: The mode of this LyricsRequest.
        :rtype: int
        """
        return self._mode

    @mode.setter
    def mode(self, mode: int):
        """Sets the mode of this LyricsRequest.

        Song mode. 0 if minor, 1 if major.  # noqa: E501

        :param mode: The mode of this LyricsRequest.
        :type mode: int
        """

        self._mode = mode

    @property
    def key(self) -> int:
        """Gets the key of this LyricsRequest.


        :return: The key of this LyricsRequest.
        :rtype: int
        """
        return self._key

    @key.setter
    def key(self, key: int):
        """Sets the key of this LyricsRequest.


        :param key: The key of this LyricsRequest.
        :type key: int
        """

        self._key = key

    @property
    def time_signature(self) -> int:
        """Gets the time_signature of this LyricsRequest.


        :return: The time_signature of this LyricsRequest.
        :rtype: int
        """
        return self._time_signature

    @time_signature.setter
    def time_signature(self, time_signature: int):
        """Sets the time_signature of this LyricsRequest.


        :param time_signature: The time_signature of this LyricsRequest.
        :type time_signature: int
        """

        self._time_signature = time_signature

    @property
    def danceability(self) -> float:
        """Gets the danceability of this LyricsRequest.


        :return: The danceability of this LyricsRequest.
        :rtype: float
        """
        return self._danceability

    @danceability.setter
    def danceability(self, danceability: float):
        """Sets the danceability of this LyricsRequest.


        :param danceability: The danceability of this LyricsRequest.
        :type danceability: float
        """

        self._danceability = danceability

    @property
    def energy(self) -> float:
        """Gets the energy of this LyricsRequest.


        :return: The energy of this LyricsRequest.
        :rtype: float
        """
        return self._energy

    @energy.setter
    def energy(self, energy: float):
        """Sets the energy of this LyricsRequest.


        :param energy: The energy of this LyricsRequest.
        :type energy: float
        """

        self._energy = energy

    @property
    def duration_ms(self) -> float:
        """Gets the duration_ms of this LyricsRequest.


        :return: The duration_ms of this LyricsRequest.
        :rtype: float
        """
        return self._duration_ms

    @duration_ms.setter
    def duration_ms(self, duration_ms: float):
        """Sets the duration_ms of this LyricsRequest.


        :param duration_ms: The duration_ms of this LyricsRequest.
        :type duration_ms: float
        """
        if duration_ms is not None and duration_ms < 0:  # noqa: E501
            raise ValueError("Invalid value for `duration_ms`, must be a value greater than or equal to `0`")  # noqa: E501

        self._duration_ms = duration_ms

    @property
    def loudness(self) -> float:
        """Gets the loudness of this LyricsRequest.


        :return: The loudness of this LyricsRequest.
        :rtype: float
        """
        return self._loudness

    @loudness.setter
    def loudness(self, loudness: float):
        """Sets the loudness of this LyricsRequest.


        :param loudness: The loudness of this LyricsRequest.
        :type loudness: float
        """

        self._loudness = loudness

    @property
    def tempo(self) -> float:
        """Gets the tempo of this LyricsRequest.


        :return: The tempo of this LyricsRequest.
        :rtype: float
        """
        return self._tempo

    @tempo.setter
    def tempo(self, tempo: float):
        """Sets the tempo of this LyricsRequest.


        :param tempo: The tempo of this LyricsRequest.
        :type tempo: float
        """
        if tempo is not None and tempo < 0:  # noqa: E501
            raise ValueError("Invalid value for `tempo`, must be a value greater than or equal to `0`")  # noqa: E501

        self._tempo = tempo

    @property
    def title(self) -> str:
        """Gets the title of this LyricsRequest.

        Song title.  # noqa: E501

        :return: The title of this LyricsRequest.
        :rtype: str
        """
        return self._title

    @title.setter
    def title(self, title: str):
        """Sets the title of this LyricsRequest.

        Song title.  # noqa: E501

        :param title: The title of this LyricsRequest.
        :type title: str
        """

        self._title = title

    @property
    def genres(self) -> List[str]:
        """Gets the genres of this LyricsRequest.

        comma-separated list of genres  # noqa: E501

        :return: The genres of this LyricsRequest.
        :rtype: List[str]
        """
        return self._genres

    @genres.setter
    def genres(self, genres: List[str]):
        """Sets the genres of this LyricsRequest.

        comma-separated list of genres  # noqa: E501

        :param genres: The genres of this LyricsRequest.
        :type genres: List[str]
        """

        self._genres = genres

    @property
    def beam_width(self) -> int:
        """Gets the beam_width of this LyricsRequest.

        Advanced parameter for the beam width of the beam search algorithm used for lyrics generation.  # noqa: E501

        :return: The beam_width of this LyricsRequest.
        :rtype: int
        """
        return self._beam_width

    @beam_width.setter
    def beam_width(self, beam_width: int):
        """Sets the beam_width of this LyricsRequest.

        Advanced parameter for the beam width of the beam search algorithm used for lyrics generation.  # noqa: E501

        :param beam_width: The beam_width of this LyricsRequest.
        :type beam_width: int
        """

        self._beam_width = beam_width

    @property
    def top_k(self) -> int:
        """Gets the top_k of this LyricsRequest.

        Advanced parameter for the search algorithm used for lyrics generation, indicating how many of the top sequences we keep in memory for each word added.  # noqa: E501

        :return: The top_k of this LyricsRequest.
        :rtype: int
        """
        return self._top_k

    @top_k.setter
    def top_k(self, top_k: int):
        """Sets the top_k of this LyricsRequest.

        Advanced parameter for the search algorithm used for lyrics generation, indicating how many of the top sequences we keep in memory for each word added.  # noqa: E501

        :param top_k: The top_k of this LyricsRequest.
        :type top_k: int
        """

        self._top_k = top_k

    @property
    def context_length(self) -> int:
        """Gets the context_length of this LyricsRequest.

        Length of the context we keep for lyrics generation  # noqa: E501

        :return: The context_length of this LyricsRequest.
        :rtype: int
        """
        return self._context_length

    @context_length.setter
    def context_length(self, context_length: int):
        """Sets the context_length of this LyricsRequest.

        Length of the context we keep for lyrics generation  # noqa: E501

        :param context_length: The context_length of this LyricsRequest.
        :type context_length: int
        """

        self._context_length = context_length

    @property
    def max_len(self) -> int:
        """Gets the max_len of this LyricsRequest.

        Max length in words for the lyric to be generated.  # noqa: E501

        :return: The max_len of this LyricsRequest.
        :rtype: int
        """
        return self._max_len

    @max_len.setter
    def max_len(self, max_len: int):
        """Sets the max_len of this LyricsRequest.

        Max length in words for the lyric to be generated.  # noqa: E501

        :param max_len: The max_len of this LyricsRequest.
        :type max_len: int
        """

        self._max_len = max_len

    @property
    def context(self) -> str:
        """Gets the context of this LyricsRequest.

        Advanced parameter for language model context.  # noqa: E501

        :return: The context of this LyricsRequest.
        :rtype: str
        """
        return self._context

    @context.setter
    def context(self, context: str):
        """Sets the context of this LyricsRequest.

        Advanced parameter for language model context.  # noqa: E501

        :param context: The context of this LyricsRequest.
        :type context: str
        """

        self._context = context
