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

class Evaluator(DeepLyric):
    """
    Evaluates model and generated lyrics of DeepLyric instance
    
    There are three general types of evaluations:
        - model evaluation - e.g. weights analysis, perplexity, etc.
        - lyric evaluation - e.g. rhyme density, POS, etc.
        - lyric comparison - e.g. BLEU, etc.
        
    """
    INIT_METRICS = {
        'BLEU_1_excl_Unsmoothed': None,
        'BLEU_2_excl_Unsmoothed': None,
        'BLEU_3_excl_Unsmoothed': None,
        'BLEU_4_excl_Unsmoothed': None,
        'BLEU_3_cumul_Smoothed': None,
        'BLEU_4_cumul_Smoothed': None,
        'POS_confirmity': None,
        'rhymeDensityAP': None,
        'rhymeDensityAV': None,
        'rhymeDensityEP': None,
        'rhymeDensityEV': None,
        'closestMeters': None,
        'editsPerLine': None
    }
    

    
    AVAILABLE_METRICS = ['BLEU_1_excl_Unsmoothed'
                         ,'BLEU_2_excl_Unsmoothed'
                         ,'BLEU_3_excl_Unsmoothed'
                         ,'BLEU_4_excl_Unsmoothed'
                         ,'BLEU_3_cumul_Smoothed'
                         ,'BLEU_4_cumul_Smoothed'
                         ,'POS_confirmity'
                         ,'rhymeDensityAP'
                         ,'rhymeDensityAV'
                         ,'rhymeDensityEP'
                         ,'rhymeDensityEV'
                         ,'closestMeters'
                         ,'editsPerLine']
    
    def __init__(self, deep_lyric, set_lyric_state=True):
        """`DeepLyric` object stores all hyperparameters and configs"""
        self.deep_lyric = deep_lyric
        self.set_metric(metrics_dict=copy(self.INIT_METRICS))
        
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

        
    # def _get_lyrics(self, n):
    #     """
    #     Generates a batch of songs of size `n`
    #
    #     Returns:
    #     list (self.best_song) : list( list (`str`) )
    #     """
    #     songs = []
    #     for i in range(n):
    #         songs.append(self.get_lyric())
    #
    #     return songs
    
    def _remove_markup_and_punc(self,token_list):
        """
        Removes tags and punctuation other than an apostrophe
        
        Parameters
        ----------
        token_list : list
            list of tokens
            
        Returns:
        --------
        cleaned token list
        
        """
        
        tags = ['xbos','xgenre','xtitle','\n']  #need complete list
        ## remove tags and punctuation other than apostrophe
        clean_token_list = [token for token in token_list if token not in tags \
                            and not token.startswith('xbol') \
                            and token not in [c for c in string.punctuation if c not in ("'")]] 
        return clean_token_list

    def _bleu(self,ref_list,candidate_token_list,nGram=4,nGramType='cumulative',shouldSmooth=True):
        '''calculates BLEU score 

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
        
        
        
        candidate = [[token for token in _remove_markup_and_punc(candidate_token_list) if token <> 'xeol']]
        
        references = [[[token for token in _remove_markup_and_punc(ref) if token <> 'xeol'] for ref in ref_list]]
        weights = weight_dict[(nGramType,nGram)]


        # scoring
        if shouldSmooth==True:
            smoother = bleu_score.SmoothingFunction().method5
        else:
            smoother = None
        score = bleu_score.corpus_bleu(references, candidate, weights, smoothing_function=smoother)
        return score
    
        
        

    def _levenshtein(self, s1, s2):
        '''calculate levenshtein distance for two input strings

        Parameters
        ----------
        s1: string
            first string for comparison
        s2: string
            second string for comparison

        Returns
        -------
        distance: levenshtein distance between two strings...that is, the lowest number of modifications to turn s1 into s2
        '''
        s1 = str(s1)
        s2 = str(s2)

        if len(s1) < len(s2):
            return _levenshtein(s2, s1)

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
    
    def _tokens_to_lines(self,token_list):
        """
        Transforms input of a list of tokens and returns a list of a list of tokens in a given line
        
        Parameters
        ----------
        token_list : list
            list of tokens making up a song
                
        """
        start = 0
        lines = []
        tokens = _remove_markup_and_punc(token_list)
        xeol_idx = [i for i, x in enumerate(tokens) if x == 'xeol']
        
        for idx in xeol_idx:
            if len(tokens[start:idx]) >0:
                lines.append(tokens[start:idx])
            start = idx+1
        return lines 
        
    def _findLineStress(self, tokenized_line):
        '''find accentual stress of a given tokenized line, based on CMU dict.  Uses relative stress per word, so somewhat limited.

        Parameters
        ----------
        tokenized_line : list
            list of tokens from line, usually preprocessed to remove non-words

        Returns
        -------
        parselist: list of potential stresses after parsing. 0 is unstressed, 1 is primary stress, 2 is secondary stress (middle)
        '''
        
        parses = ['']
        for word in tokenized_line:
            pros = pronouncing.phones_for_word(word)
            if pros:
                for phonelist in [pronouncing.phones_for_word(word)]:           
                    stressOptions = copy.deepcopy(parses)
                    currLen = len(parses)
                    newparse = []
                    # I don't really need to loop through pronunciations, just distinct stress patterns, so a little inefficient here
                    for pronunciation in phonelist:
                        wordStress = pronouncing.stresses(pronunciation)
                        for option in range(currLen):
                            newparse.append(''+str(stressOptions[option]) + str(wordStress))
                parses = newparse 

        return list(set(parses))
        
    def get_POS_conformity(self):
        """
        Calculates absolute difference from published proportions of POS, between 0 and 1.
    
        """
        
        try:
            self.generated_song
        except AttributeError as e:
            print(f"{e} : first generate song using `set_lyric_state=True`")
            raise   
        
        
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
        tokenized_text = [token for token in _remove_markup_and_punc(self.generated_song) if token <> 'xeol']
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
        self.set_metric('POS_confirmity', absdiff)
        
    
    def get_rhyme_density(self):
        """
        Calculates Rhyme Density for given tokens
        
        Result:
        -------
        Updates attributes:
            'rhymeDensityAP': rhyme density using all words, perfect rhymes only
            'rhymeDensityAV': rhyme density using all words, vowel rhymes only
            'rhymeDensityEP': rhyme density using end words, perfect rhymes only
            'rhymeDensityEV': rhyme density using end words, vowel rhymes only
                           
        """
        try:
            self.generated_song
        except AttributeError as e:
            print(f"{e} : first generate song using `set_lyric_state=True`")
            raise   
            
        # initialize
        rhymePart_cnt_EP = Counter()
        rhymePart_cnt_EV = Counter()
        rhymePart_cnt_AP = Counter()
        rhymePart_cnt_AV = Counter()
        
        rhyme_cnt_EP = 0
        rhyme_cnt_EV = 0
        rhyme_cnt_AP = 0
        rhyme_cnt_AV = 0

        # prepare data
        
        ## remove tags and punctuation other than apostrophe
        words_A = _remove_markup_and_punc(self.generated_song)
        
        ## find 'xeol' indices, then use these to replace with line break and also find 'end' words
        xeol_idx = [i for i, x in enumerate(words_A) if x == 'xeol']
        for i in xeol_idx:
            words_A[i] = '\n'
        words_E = [words_A[idx-1] for idx in xeol_idx if idx>0]

        wordCount_A = len(words_A)
        wordCount_E = len(words_E)
        
        #process 'all' words 
        for word in words_A:
            pros = pronouncing.phones_for_word(word)
            if pros:     
                phonelist = pros[0]  #using first pronunciation for now
                if len(phonelist) > 0:
                        rhymePart_cnt_AP[pronouncing.rhyming_part(phonelist)] +=1
                        rhymePart_cnt_AV[pronouncing.rhyming_part(phonelist).split()[0]] +=1

        for v in rhymePart_cnt_AP.values():
            rhyme_cnt_AP += v-1
        for v in rhymePart_cnt_AV.values():
            rhyme_cnt_AV += v-1 
        
        if wordCount_A > 1:
            rhymeDensityAP = rhyme_cnt_AP/float(wordCount_A-1)
            rhymeDensityAV = rhyme_cnt_AV/float(wordCount_A-1)
        else:
            rhymeDensityAP = 0.0
            rhymeDensityAV = 0.0
            
        #process 'end' words 
        for word in words_E:
            pros = pronouncing.phones_for_word(word)
            if pros:     
                phonelist = pros[0]  #using first pronunciation for now
                if len(phonelist) > 0:
                        rhymePart_cnt_EP[pronouncing.rhyming_part(phonelist)] +=1
                        rhymePart_cnt_EV[pronouncing.rhyming_part(phonelist).split()[0]] +=1

        for v in rhymePart_cnt_EP.values():
            rhyme_cnt_EP += v-1
        for v in rhymePart_cnt_EV.values():
            rhyme_cnt_EV += v-1            
            
        if wordCount_E > 1:
            rhymeDensityEP = rhyme_cnt_EP/float(wordCount_E-1)
            rhymeDensityEV = rhyme_cnt_EV/float(wordCount_E-1)
        else:
            rhymeDensityEP = 0.0
            rhymeDensityEV = 0.0
        
        # use set_metric
        self.set_metric('rhymeDensityAP', rhymeDensityAP)
        self.set_metric('rhymeDensityAV', rhymeDensityAV)
        self.set_metric('rhymeDensityEP', rhymeDensityEP)
        self.set_metric('rhymeDensityEV', rhymeDensityEV)
 
    
    def get_bleu(self,reference_dir='../data/lyrics/reference/',candidate_token_list,nGram=4,nGramType='cumulative',shouldSmooth=True,max_refs=None)):
        '''
        Calculates BLEU score for nGrams 1-4 and smoothed cumulative for nGram in (3,4).
        
        Parameters
        ----------
        reference_dir: string
            expects a string of a directory with reference .txt files
        candidate_token_list: list
            the new token list that represents the string that needs to be scored
        nGram: int
            choose between 1-4.  Determines which ngram(s) to use in the scoring
        nGramType: string 
            'cumulative' uses a simple average of all ngrams from 1 to nGram. 'exclusive' is the chosen nGram only.
        shouldSmooth: boolean
            if False, calculates the BLEU score without smoothing. Recommended to use smoothing (set to True)
        max_refs: int or None
            maximum amount of reference texts to be considered
        
        Result
        -------
        Updates attributes scores: 
            BLEU_1_excl_Unsmoothed
            BLEU_2_excl_Unsmoothed
            BLEU_3_excl_Unsmoothed
            BLEU_4_excl_Unsmoothed
            BLEU_3_cumul_Smoothed
            BLEU_4_cumul_Smoothed

        
        '''
        try:
            self.generated_song
        except AttributeError as e:
            print(f"{e} : first generate song using `set_lyric_state=True`")
            raise   
        
        refs = glob.glob(reference_dir+'/*.txt')
        if max_refs is not None:
            if len(refs) < max_refs:
                max_refs = len(refs)
            refs = refs[0:max_refs]
        ref_list = []
    
        ## add BLEU reference code
        for ref in refs:
            with open(ref) as rf:
                ref_raw_text = rf.read()
                ref_list.append(ref_raw_text)

        # use set_metric
        self.set_metric('BLEU_1_excl_Unsmoothed', bleu(ref_list,self.generated_song,nGram=1,nGramType='exclusive',shouldSmooth=False,max_refs=1000)
        self.set_metric('BLEU_2_excl_Unsmoothed', bleu(ref_list,self.generated_song,nGram=2,nGramType='exclusive',shouldSmooth=False,max_refs=1000)
        self.set_metric('BLEU_3_excl_Unsmoothed', bleu(ref_list,self.generated_song,nGram=3,nGramType='exclusive',shouldSmooth=False,max_refs=1000)
        self.set_metric('BLEU_4_excl_Unsmoothed', bleu(ref_list,self.generated_song,nGram=4,nGramType='exclusive',shouldSmooth=False,max_refs=1000)
        self.set_metric('BLEU_3_cumul_Smoothed', bleu(ref_list,self.generated_song,nGram=3,nGramType='cumulative',shouldSmooth=True,max_refs=1000)
        self.set_metric('BLEU_4_cumul_Smoothed', bleu(ref_list,self.generated_song,nGram=4,nGramType='cumulative',shouldSmooth=True,max_refs=1000)
        
        
        
    def findMeter(self):
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
        
        try:
            self.generated_song
        except AttributeError as e:
            print(f"{e} : first generate song using `set_lyric_state=True`")
            raise   
        
        
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
        
        tokenized_list = _remove_markup_and_punc(self.generated_song)
        lines = _tokens_to_lines(tokenized_list)
        line_cnt = len(lines)
        minDist = 999

        # update distances
        for line in lines:
            for k,v in meter_dict.items():
                minDist = 999
                for reading in _findLineStress(line)[0]:
                    dist = _levenshtein(k,reading)
                    if dist < minDist:
                        minDist = dist    
                vote_cnt[v] += minDist

        lowest = min(vote_cnt.values()) 
        options = [k for k,v in vote_cnt.items() if v==lowest]
        
        # use set_metric        
        self.set_metric('closestMeters', options)
        self.set_metric('editsPerLine', lowest/float(line_cnt))
    
    def get_all_metrics(self):
        """
        Runs all available metrics and updates `self.metrics` state
        """
        
        self.get_rhyme_density()
        self.get_POS_conformity()
        self.findMeter()
        self.get_bleu()

        
    
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
        
        