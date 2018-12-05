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
            songs.append(self.get_lyric())
            
        return songs
        
    def get_rhyme_density(self, ):
        """
        Calculates Rhyme Density for given tokens
        
        Parameters
        ----------
        
        
        """
        
    def get_predicted_probs(self, seed_text='xbos', max_len=40, GPU=False,
                      context_length=30, beam_width=3, verbose=1,
                      temperature=1.5, top_k=3, multinomial=True, audio=None):
        """
        Idenitcal generation algorithm as `generate_text` but instead predicted probabilities
        
        Parameters
        ----------
        seed_text : list or str
            List of strings where each item is a token. (e.g. ['the', 'cat']) or string that is split on white space

        max_len : int
            Number of words in generated sequence
            
        gpu : bool
            If you're using a GPU or not...
        
        context_length : int
            Amount of words that get input as "context" into the model. Set to 0 for no limit
            
        beam_width : int
            How many new word indices to try out...computationally expensive
        
        verbose : int
            0: Print nothing to console
            1: Print currently generated word number to console
            2: Print currently generated word number and all currently considered song options to console
        
        temperature : float
            Hyperparameter for adjusting the predicted probabilities by scaling the logits prior to softmax
          
        top_k : int
            Number of song options to keep over each loop. This should normally be set to beam_width
        
        audio : 2darray - 1 x n
            audio features for a song. `n` should equal the size of the
            multimodal features in `model`
        
        Returns
        -------
        self._context_and_scores : list of lists
            Returns a sorted list of the entire tree search of contexts and their respective scores in the form:
            [[context, score], [context, score], ..., [context, score]]
        """
        if isinstance(seed_text, str):
            seed_text = self.tokenize(seed_text)
        
        seed_text = self.numericalize(seed_text)
        
        # List of candidate word sequence. We'll maintain #beam_width top sequences here.
        # The context is a list of words, the scores are the sum of the log probabilities of each word
        self._context_and_scores = [[seed_text, 0.0]]
        
        total_word_probs = []
        
        # Loop over max number of words
        for word_number in range(max_len):
            if verbose==1 or verbose==2: print(f'Generating word: {word_number+1} / {max_len}')

            candidates = []
            next_word_probs = []
            
            # For each possible context that we've generated so far, generate new probabilities,
            # and pick an additional #beam_width next candidates
            for i in range(len(self._context_and_scores)):
                # Get a new sequence of word indices and log-probability
                # Example: [[2, 138, 661], 23.181717]
                context, score = self._context_and_scores[i]
                # Obtain probabilities for next word given the context
                probabilities = self.get_text_distribution(context, context_length, temperature, GPU, audio)

                # Multinomial draw from the probabilities
                if multinomial:
                    multinom_draw = np.random.multinomial(beam_width, probabilities)
                    top_probabilities = np.argwhere(multinom_draw != 0).flatten()
                    
                # no multinomial draw
                else:
                    top_probabilities = np.argsort(-probabilities)[:beam_width]
                            
                # For each possible new candidate, update the context and scores
                for j in range(len(top_probabilities)):
                    next_word_idx = top_probabilities[j]
                    new_context = context + [next_word_idx]
                    candidate = [new_context, (score - np.log(probabilities[next_word_idx]))]
                    candidates.append(candidate)
                    
                    # store predicted probablities
                    next_word_prob = probabilities[next_word_idx]
                    potential_next_word = self.get_word_from_index(next_word_idx)
                    prior_context = [self.get_word_from_index(w) for w in context]
                    next_word_probs.append((prior_context, potential_next_word, next_word_prob))
                    
            total_word_probs.extend(next_word_probs)
            
            # Update the running tally of context and scores and sort by probability of each entry
            self._context_and_scores = candidates
            self._context_and_scores = sorted(self._context_and_scores, key = lambda x: x[1]) #sort by top entries

            self._context_and_scores = self._context_and_scores[:top_k] #for now, only keep the top 15 to speed things up but we can/should change this to beam_width or something else
            self.best_song, self.best_score = self._context_and_scores[0]

            if verbose==2:
                for context, score in self._context_and_scores:
                    self.print_lyrics(context)
                    print('\n')
            
        total_word_probs = sorted(total_word_probs, key=lambda x: -x[2]) # sort by highest probabilities
        return total_word_probs
