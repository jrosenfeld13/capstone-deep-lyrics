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
        tokens = [line[-1] for line in parse_tokens(tokens, lines=True, tags=False)\
                  if line]
        
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
    
def bleu(tokens, ref_list, nGram=4, nGramType='cumulative', shouldSmooth=True):
    '''
    calculates BLEU score

    Parameters
    ----------
    ref_list: list
        expects a list of reference texts to compare (as strings)
    candidate_token_list: list
        the new token list that represents the string that needs to be scored
    nGram: int
        choose between 1-4.  Determines which ngram(s) to use in the scoring
    nGramType: string
        'cumulative' uses a simple average of all ngrams from 1 to nGram. 'exclusive' is the chosen nGram only.
    shouldSmooth: boolean
        if False, calculates the BLEU score without smoothing. Recommended to use smoothing (set to True)

    Returns
    -------
    score: BLEU score using nGram settings input, smoothed by default (can be turned off)
    '''

    # basic checks
    if nGram not in [1,2,3,4]:
        raise ValueError('nGram must be between 1 and 4')

    if nGramType not in ['cumulative','exclusive']:
        raise ValueError('nGramType must either be cumulative (average of nGrams less than n) or exclusive (1=unigram, etc.)')

    # pre-score
    weight_dict = {('cumulative',1):(1,0,0,0)
                  ,('cumulative',2):(.5,.5,0,0)
                  ,('cumulative',3):(.3333,.3333,.3333,0)
                  ,('cumulative',4):(.25,.25,.25,.25)
                  ,('exclusive',1):(1,0,0,0)
                  ,('exclusive',2):(0,1,0,0)
                  ,('exclusive',3):(0,0,1,0)
                  ,('exclusive',4):(0,0,0,1)}

    candidate = parse_tokens(tokens, lines=False, tags=False)
    references = [parse_tokens(r, lines=False, tags=False) for r in ref_list]

    weights = weight_dict[(nGramType,nGram)]

    # scoring
    if shouldSmooth==True:
        smoother = bleu_score.SmoothingFunction().method5
    else:
        smoother = None
    score = bleu_score.sentence_bleu(references, candidate, weights, smoothing_function=smoother)
    return score

def findLineStress(tokenized_line):
    '''
    find accentual stress of a given tokenized line, based on CMU dict.
    Uses relative stress per word, so somewhat limited.

    Parameters
    ----------
    tokenized_line : list
        list of tokens from line, usually preprocessed to remove non-words

    Returns
    -------
    parselist: list of potential stresses after parsing.
        0 is unstressed, 1 is primary stress, 2 is secondary stress (middle)
    '''
    
    parses = ['']
    for word in tokenized_line:
        pros = pronouncing.phones_for_word(word)
        if pros:
            for phonelist in [pronouncing.phones_for_word(word)]:
                stressOptions = deepcopy(parses)
                currLen = len(parses)
                newparse = []
                # I don't really need to loop through pronunciations
                # just distinct stress patterns, so a little inefficient here
                for pronunciation in phonelist:
                    wordStress = pronouncing.stresses(pronunciation)
                    for option in range(currLen):
                        newparse.append(''+str(stressOptions[option]) + str(wordStress))
            parses = newparse

    return list(set(parses))

def levenshtein(s1, s2):
    '''calculate levenshtein distance for two input strings

    Parameters
    ----------
    s1: string
        first string for comparison
    s2: string
        second string for comparison

    Returns
    -------
    distance: levenshtein distance between two strings...that is,
    the lowest number of modifications to turn s1 into s2
    '''
    s1 = str(s1)
    s2 = str(s2)

    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # otherwise len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
    
def findMeter(tokens):
    '''finds meter with smallest edit distance

    Parameters
    ----------
    token_list : list
        list of tokens making up a song

    Result
    ------
    Updates attributes:
        edits_per_line: average lowest edit distance per line for any standard accentual-syllabic verse
        options: list of potential meters for the lowest edit distance
    '''

    # define
    meter_dict = {'0101':'Iambic dimeter'
                  ,'010101':'Iambic trimeter'
                  ,'01010101':'Iambic tetrameter'
                  ,'0101010101':'Iambic pentameter'
                  ,'010101010101':'Iambic hexameter'
                  ,'01010101010101':'Iambic heptameter'
                  ,'0101010101010101':'Iambic octameter'
                  ,'1010':'Trochaic dimeter'
                  ,'101010':'Trochaic trimeter'
                  ,'10101010':'Trochaic tetrameter'
                  ,'1010101010':'Trochaic pentameter'
                  ,'101010101010':'Trochaic hexameter'
                  ,'10101010101010':'Trochaic heptameter'
                  ,'1010101010101010':'Trochaic octameter'
                  ,'001001':'Anapestic dimeter'
                  ,'001001001':'Anapestic trimeter'
                  ,'001001001001':'Anapestic tetrameter'
                  ,'001001001001001':'Anapestic pentameter'
                  ,'001001001001001001':'Anapestic hexameter'
                  ,'001001001001001001001':'Anapestic heptameter'
                  ,'100100':'Dactyllic dimeter'
                  ,'100100100':'Dactyllic trimeter'
                  ,'100100100100':'Dactyllic tetrameter'
                  ,'100100100100100':'Dactyllic pentameter'
                  ,'100100100100100100':'Dactyllic hexameter'
                  ,'100100100100100100100':'Dactyllic heptameter'}

    # initialize
    vote_cnt = Counter()
    
    lines = parse_tokens(tokens, lines=True, tags=False)
    line_cnt = len(lines)
    minDist = 999

    # update distances
    for line in lines:
        for k,v in meter_dict.items():
            minDist = 999
            for reading in findLineStress(line)[0]:
                dist = levenshtein(k,reading)
                if dist < minDist:
                    minDist = dist
            vote_cnt[v] += minDist
    
    try:
        lowest = min(vote_cnt.values())
    except ValueError:
        return None, None
    options = [k for k,v in vote_cnt.items() if v==lowest]
    
    # use set_metric
    return options, lowest/float(line_cnt)

def get_POS_conformity(tokens):
    """
    Calculates absolute difference from published proportions of POS, between 0 and 1.

    """
    # define lookups
    mapping = {'CC':'CC','DT':'DT','PDT':'DT','WDT':'DT','IN':'IN','JJ':'JJ','JJR':'JJ','JJS':'JJ'
               ,'NN':'NN','NNS':'NN','NNP':'NN','NNPS':'NN','LS':'OT','CD':'OT','EX':'OT','FW':'OT'
               ,'POS':'OT','UH':'OT','RB':'RB','RBR':'RB','RBS':'RB','WRB':'RB','TO':'TO','MD':'VB'
               ,'RP':'VB','VB':'VB','VBD':'VB','VBG':'VB','VBN':'VB','VBP':'VB','VBZ':'VB','PRP':'WP'
               ,'PRP$':'WP','WP':'WP','WP$':'WP'}
    comp_dict = {'CC':0.0212,'DT':0.0982,'IN':0.0998,'JJ':0.0613,'NN':0.3051,'RB':0.0766,'TO':0.0351
                 ,'VB':0.285,'WP':0.0058,'OT':0.012}

    # initialize
    pos_cnt = Counter()
    total_word_cnt = 0
    pos_dict = defaultdict(float)
    pos_dict['adjustment'] = 0
    absdiff = 0

    # prepare data
    tokenized_text = parse_tokens(tokens, lines=False, tags=False)
    tag_list = nltk.pos_tag(tokenized_text)

    # initial proportions
    for t in tag_list:
        pos_cnt[t[1]] +=1
        total_word_cnt +=1
    pos_raw_dict = {k: v/float(total_word_cnt) for k,v in dict(pos_cnt).items()}

    # adjust for items missing in mapping (mostly punctuation)
    for k,v in pos_raw_dict.items():
        if k in mapping:
            pos_dict[mapping[k]] += v
        else:
            pos_dict['adjustment'] += v
    for k,v in pos_dict.items():
        pos_dict[k] = pos_dict[k]/(1-pos_dict['adjustment'])
    del pos_dict['adjustment']

    # compare to observed ratios, calculate absolute difference
    for k in comp_dict.keys():
        absdiff += abs(comp_dict[k] - pos_dict.get(k,0))

    # use set_metric
    return absdiff
