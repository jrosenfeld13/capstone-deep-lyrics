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

def parse_tokens(tokens, lines=True, tags=False):
    """
    Parses tokens with various options for evaluation methods.
    Assumes `xbol-1` tag as first line of actual lyrics.
    
    Input Example:
    --------------
    ['xbos', 'xgenre', 'death', 'metal', 'xtitle', 'and', 'i', 'don', "'t",
     'think', 'that', 'xbol-1', 'today', 'is', 'the', 'greatest', 'day', 'ever', 'xeol',
     'xbol-2', 'so', 'what', 'never', 'xeol', 'xeos']
          
    Output Example:
    ---------------
    `lines=True` and `tags=False`:
        [['today', 'is', 'the', 'greatest', 'day', 'ever'],
         ['so', 'what', 'never']]
         
          
    """
    # lines and no tags
    if lines and not tags:
        reached_bol = False
        parsed_tokens = []
        for w in tokens:
            if w == 'xbol-1':
                reached_bol = True
                current_line = []
                continue

            if not reached_bol:
                continue

            if 'xbol' in w:
                parsed_tokens.append(current_line)
                current_line = []
                continue

            elif w not in ['xeol', 'xeos']:
                current_line.append(w)

            elif w == 'xeos':
                parsed_tokens.append(current_line)
                break
                
    # no lines and no tags:
    if not lines and not tags:
        reached_bol = False
        parsed_tokens = []
        for w in tokens:
            if w == 'xbol-1':
                reached_bol = True
                continue
                
            if not reached_bol:
                continue
                
            if 'xbol' in w:
                continue
                
            elif w not in ['xbol', 'xeol', 'xeos']:
                parsed_tokens.append(w)
                
            elif w == 'xeos':
                break
                
    # no lines and tags:
    if not lines and tags:
        reached_bol = False
        parsed_tokens = []
        for w in tokens:
            if w == 'xbol-1':
                reached_bol = True
                parsed_tokens.append('xbol')
                continue
                
            if not reached_bol:
                continue
                
            if 'xbol' in w:
                parsed_tokens.append('xbol')
                continue
                
            elif w == 'xeos':
                break
                
            else:
                parsed_tokens.append(w)
                
    # lines and tags (not necessary):
    if lines and tags:
        raise Exception(f'Combination of lines=True and tags=True is not implemented')
        
    return parsed_tokens

def calculate_rhyme_density(tokens, rhymeType='perfect', rhymeLocation='all'):
    """
    Computes rhyme density for a list of tokens
    
    Parameters:
    -----------
    rhymeType : str
        - 'perfect' is a perfect rhyme
        - 'stressed' is a rhyming in the vowel sound + stress only
        - 'allVowels' is a rhyming at all vowel syllables
        
    rhymeLocation : str
        choose to look at 'all' text or 'end' (last word in each line)
    """
    
    assert rhymeType in ['perfect', 'stressed', 'allVowels'], "Unexpected value for rhymeType"
    assert rhymeLocation in ['all', 'end'], "Unexpected value for rhymeLocation"
    
    rhymePart_cnt = Counter()
    rhyme_cnt = 0
    distinct_rhyme_cnt = 0
    
    if rhymeLocation == 'all':
        tokens = parse_tokens(tokens, lines=False, tags=False)
        
    elif rhymeLocation == 'end':
        tokens = [line[-1] for line in parse_tokens(tokens, lines=True, tags=False)]
        
    # only retrieve first pronunciation from `phones_for_words`
    # we can enhance here by doing permutations of pronunciations
    pros = [pronouncing.phones_for_word(token)[0] for token in tokens\
            if pronouncing.phones_for_word(token)]
    for pro in pros:
        if rhymeType == 'perfect':
            rhymePart_cnt[pronouncing.rhyming_part(pro)] += 1
        elif rhymeType == 'stressed':
            # look at only stressed syllables
            # slightly modified logic from JP implementation
            rhyming_parts = pronouncing.rhyming_part(pro).split()
            if rhyming_parts:
                rhyming_parts = [part for part in rhyming_parts if part[-1] in ['1', '2']]
            if rhyming_parts:
                rhyming_parts = rhyming_parts[0]
            else:
                continue
            rhymePart_cnt[rhyming_parts] += 1
        elif rhymeType == 'allVowels':
            # look at all vowel parts - new method
            rhyming_parts = pronouncing.rhyming_part(pro).split()
            rhyming_parts = [part for part in rhyming_parts if part[-1].isdigit()]
            for rhyme in rhyming_parts:
                rhymePart_cnt[rhyme] += 1

    for v in rhymePart_cnt.values():
        rhyme_cnt += v-1
            
    # denominator - word for 'perfect'; vowel syllables for 'vowel'
    # denominator = sum(rhymePart_cnt.values())-1
    denominator = len(tokens)-1
    
    if denominator > 0:
        rhymeDensity = rhyme_cnt / denominator
    else:
        rhymeDensity = None
            
        
#     return tokens, pros, rhymePart_cnt, rhyming_parts, rhyme_cnt, rhymeDensity
    return rhymeDensity
