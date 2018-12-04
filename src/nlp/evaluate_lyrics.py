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
        self.deep_lyric = deep_lyric
        
    def 
    
