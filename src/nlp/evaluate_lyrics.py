import numpy as np
import nltk
from nltk.translate import bleu_score
import nltk.tokenize
from fastai import *
from fastai.text import *

from collections import Counter
from collections import defaultdict
import string
import pronouncing
import copy

import json
from datetime import datetime
import requests

from copy import copy, deepcopy
from enum import Enum

from .generate_lyrics import DeepLyric
from .evaluation_methods import parse_tokens, calculate_rhyme_density

def get_bleu_reference():
    """
    Retrieve preprocessor from google cloud storage
    """
    REFERENCE_URL = 'https://storage.googleapis.com/w210-capstone/lyrics/reference/4.1-LM-108k-lines-validation-tokens_100.pkl'
    ref = requests.get(REFERENCE_URL)
    ref = ref.content
    ref = pickle.load(ref)
    return ref

class Evaluator(DeepLyric):
    """
    Evaluates model and generated lyrics of DeepLyric instance
    
    There are three general types of evaluations:
        - model evaluation - e.g. weights analysis, perplexity, etc.
        - lyric evaluation - e.g. rhyme density, POS, etc.
        - lyric comparison - e.g. BLEU, etc.
        
    """
    INIT_METRICS = {
        'rhymeDensityAP': None,
        'rhymeDensityAV': None,
        'rhymeDensityAS': None,
        'rhymeDensityEP': None,
        'rhymeDensityEV': None,
        'rhymeDensityES': None,
        'BLEU_1_excl_Unsmoothed': None,
        'BLEU_2_excl_Unsmoothed': None,
        'BLEU_3_excl_Unsmoothed': None,
        'BLEU_4_excl_Unsmoothed': None,
        'BLEU_3_cumul_Smoothed': None,
        'BLEU_4_cumul_Smoothed': None,
        'closestMeters': None,
        'editsPerLine': None,
        'POS_conformity': None
    }
        
    # AVAILABLE_METRICS = ['list', 'of', 'metrics']
    
    def __init__(self, deep_lyric, set_lyric_state=True):
        """`DeepLyric` object stores all hyperparameters and configs"""
        self.deep_lyric = deep_lyric
        self.set_metric(metrics_dict=copy(self.INIT_METRICS))
        
        self.bleu_ref = get_bleu_reference()
        
        if set_lyric_state:
            self.get_lyric()
    
    @property
    def metrics(self):
        return self._metrics
        
    def set_metric(self, key=None, value=None, metrics_dict=None):
        """
        Set evaluation metric and its corresponding value
        
        Parameters:
        -----------
        key : str
            metric key, e.g. rhyme_density, pos, etc.
        value : str
            metric value
        config_dict : dict
            dictionary of {`key`: `value`} for passing in default metrics
        """
        if not metrics_dict:
            self._metrics[key] = value
        else:
            self._metrics = metrics_dict
    
    def get_lyric(self):
        """
        Generates one song with given hyperparamters and updates state
        `self.generated_song`
        
        Returns:
        --------
        song : list (`str`)
            string sentence tokens
        """
        self.deep_lyric.generate_text()
        song_idx = self.deep_lyric.best_song
        self.generated_song = [self.deep_lyric.get_word_from_index(w) for w in song_idx]

    def evaluate(self, out=False):
        """
        Runs all available metrics and updates `self.metrics` state
        and returns state dictionary
        
        Updates:
        --------
        'rhymeDensityAP': rhyme density using all words, perfect rhymes only
        'rhymeDensityAV': rhyme density using all words, all vowel rhymes
        'rhymeDensityAS': rhyme density using all words, all stressed rhymes
        'rhymeDensityEP': rhyme density using end words, perfect rhymes only
        'rhymeDensityEV': rhyme density using end words, all vowel rhymes
        'rhymeDensityES': rhyme density using end words, all vowel rhymes
        'BLEU_1_excl_Unsmoothed'
        'BLEU_2_excl_Unsmoothed'
        'BLEU_3_excl_Unsmoothed'
        'BLEU_4_excl_Unsmoothed'
        'BLEU_3_cumul_Smoothed'
        'BLEU_4_cumul_Smoothed'
        """
        try:
            self.generated_song
        except AttributeError as e:
            print(f"{e} : first generate song using `set_lyric_state=True`")
            raise
        
        # rhyme density
        rhymeDensityAP = calculate_rhyme_density(self.generated_song,
                                                 rhymeType='perfect',
                                                 rhymeLocation='all')
        rhymeDensityAV = calculate_rhyme_density(self.generated_song,
                                                 rhymeType='allVowels',
                                                 rhymeLocation='all')
        rhymeDensityAS = calculate_rhyme_density(self.generated_song,
                                                 rhymeType='stressed',
                                                 rhymeLocation='all')
        rhymeDensityEP = calculate_rhyme_density(self.generated_song,
                                                 rhymeType='perfect',
                                                 rhymeLocation='end')
        rhymeDensityEV = calculate_rhyme_density(self.generated_song,
                                                 rhymeType='allVowels',
                                                 rhymeLocation='end')
        rhymeDensityES = calculate_rhyme_density(self.generated_song,
                                                 rhymeType='stressed',
                                                 rhymeLocation='end')
                                                 
        self.set_metric('rhymeDensityAP', rhymeDensityAP)
        self.set_metric('rhymeDensityAV', rhymeDensityAV)
        self.set_metric('rhymeDensityAS', rhymeDensityAS)
        self.set_metric('rhymeDensityEP', rhymeDensityEP)
        self.set_metric('rhymeDensityEV', rhymeDensityEV)
        self.set_metric('rhymeDensityES', rhymeDensityES)
        
        # BLEU
        # use set_metric
        self.set_metric('BLEU_1_excl_Unsmoothed',
                        bleu(self.generated_song, self.bleu_ref,
                        nGram=1, nGramType='exclusive', shouldSmooth=False))
        self.set_metric('BLEU_2_excl_Unsmoothed',
                        bleu(self.generated_song, self.bleu_ref,
                        nGram=2, nGramType='exclusive', shouldSmooth=False))
        self.set_metric('BLEU_3_excl_Unsmoothed',
                        bleu(self.generated_song, self.bleu_ref,
                        nGram=3, nGramType='exclusive', shouldSmooth=False))
        self.set_metric('BLEU_4_excl_Unsmoothed',
                        bleu(self.generated_song, self.bleu_ref,
                        nGram=4, nGramType='exclusive', shouldSmooth=False))
        self.set_metric('BLEU_3_cumul_Smoothed',
                        bleu(self.generated_song, self.bleu_ref,
                        nGram=3, nGramType='cumulative', shouldSmooth=True))
        self.set_metric('BLEU_4_cumul_Smoothed',
                        bleu(self.generated_song, self.bleu_ref,
                        nGram=4, nGramType='cumulative', shouldSmooth=True))
        
        # Meter
        closestMeters, editsPerLine = findMeter(self.generated_song)
        self.set_metric('closestMeters', closestMeters)
        self.set_metric('editsPerLine', editsPerLine)
        
        # POS conformity
        self.set_metric('POS_conformity',
                        get_POS_conformity(self.generated_song))
                
        if out:
            return self.metrics

    def save_json(self, dir=None, name=None, out=False):
        """
        Saves `self.deep_lyric.cofig`, `self.metrics`, and `self.generated_song`
        to JSON
        
        Parameters
        ----------
        dir : str
            directory to store json output
        name : str
            If none, utc timestamp will be used
            
        Returns
        -------
        Saves to file json of the following schema
        
        {
            meta : `self.deep_lyrics.config`,
            lyric : ['these', 'are', 'lyric', 'tokens'],
            metrics : `self.metrics`
        }
        """
        
        if not name:
            name = str(round(datetime.timestamp(datetime.utcnow())))
            
        try:
            self.generated_song
        except AttributeError as e:
            print(f"{e} : first generate song using `set_lyric_state=True`")
            raise
        
        payload = {'meta': self.deep_lyric.config,
                   'lyric': self.generated_song,
                   'metrics': self.metrics}
        
        if dir:
            full_path = f"{dir}/{name}"
            with open(full_path, "w") as f:
                json.dump(payload, f, indent=4)
                
        if out:
            return payload
            
    def batch_analysis(n):
        """
        Iterates lyrics and metrics and exports to desired `out` type.
        We don't update states with this function
        
        Parameters:
        -----------
        n : int
            number of examples to run evaluation on
            
        Returns:
        --------
        csv :
        """
        pass
