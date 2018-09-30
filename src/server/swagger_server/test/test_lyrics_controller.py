# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.lyrics_request import LyricsRequest  # noqa: E501
from swagger_server.test import BaseTestCase


class TestLyricsController(BaseTestCase):
    """LyricsController integration test stubs"""

    def test_generate_lyrics(self):
        """Test case for generate_lyrics

        Request lyric generation.
        """
        body = LyricsRequest()
        response = self.client.open(
            '/lyrics',
            method='POST',
            data=json.dumps(body),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
