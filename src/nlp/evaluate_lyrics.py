import numpy as np
import nltk.tokenize
from fastai import *
from fastai.text import *

from copy import copy, deepcopy
from enum import Enum

class Evaluator():
    """
    Evaluates model and generated lyrics of DeepLyric instance
    
    There are three general types of evaluations:
        - model evaluation - e.g. weights analysis, perplexity, etc.
        - lyric evaluation - e.g. rhyme density, POS, etc.
        - lyric comparison - e.g. BLEU, etc.
        
    """
    
    def __init__(self, deep_lyric):
        """`DeepLyric` object stores all hyperparameters and configs"""
        self.deep_lyric = deep_lyric
        
    def get_lyric(self):
        """
        Generates one song with given hyperparamters
        
        Returns:
        --------
        song : list (`str`)
            string sentence tokens
        """
        self.deep_lyric.generate_text()
        song_idx = self.deep_lyric.best_song
        song = [self.deep_lyric.get_word_from_index(w) for w in song_idx]
        return song
        
    def get_lyrics(self, n):
        """
        Generates a batch of songs of size `n`
        
        Returns:
        list (self.best_song) : list( list (`str`) )
        """
        songs = []
        for i in range(n):
            songs.append(self.get_lyrics())
            
        return songs
        
    def get_rhyme_density(self, ):
        """
        Calculates Rhyme Density for given tokens
        
        Parameters
        ----------
        
        
        """
        
    
