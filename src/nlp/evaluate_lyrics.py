import numpy as np
import nltk.tokenize
from fastai import *
from fastai.text import *

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
        'metric1': None,
        'metric2': None,
        'metric3': None
    }
    
    AVAILABLE_METRICS = ['list', 'of', 'metrics']
    
    def __init__(self, deep_lyric, set_lyric_state=True):
        """`DeepLyric` object stores all hyperparameters and configs"""
        self.deep_lyric = deep_lyric
        self.set_metric(metrics_dict=self.INIT_METRICS)
        
        if set_lyric_state:
            self._get_lyric()
    
    @property
    def metrics(self):
        return self._metrics
        
    def set_metric(self, kev=None, value=None, metrics_dict=None):
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
    
    def _get_lyric(self):
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
        
    def get_rhyme_density(self, ):
        """
        Calculates Rhyme Density for given tokens
        
        Parameters
        ----------
        
        
        """
        # code that comes up with metric
        
        self.set_metric('rhyme_density_a', rhyme_density_a)
        self.set_metric('rhyme_density_b', rhyme_density_b)
        
    def get_all_metrics(self):
        """
        Runs all available metrics and updates `self.metrics` state
        """
        
        self.get_rhyme_density()

    
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
            meta : `self.config`,
            lyric : ['these', 'are', 'lyric', 'tokens'],
            metrics : `self.metrics`
        }
        """
        
        if not name:
            name = str(round(datetime.timestamp(datetime.utcnow())))
            
        try:
            self.best_song
        except AttributeError as e:
            print(f"{e} : first generate song using generate_text()")
            raise
        
        song_idx = self.best_song
        song = [self.get_word_from_index(w) for w in song_idx]
        payload = {'meta': self.config, 'lyric': song}
        
        if dir:
            full_path = f"{dir}/{name}"
            with open(full_path, "w") as f:
                json.dump(payload, f, indent=4)
                
        if out:
            return payload
            
    def batch_csv(n=None):
        pass
    
    # def get_predicted_probs(self, seed_text='xbos', max_len=40, GPU=False,
    #                   context_length=30, beam_width=3, verbose=1,
    #                   temperature=1.5, top_k=3, multinomial=True, audio=None):
    #     """
    #     Idenitcal generation algorithm as `generate_text` but instead predicted probabilities
    #
    #     Parameters
    #     ----------
    #     seed_text : list or str
    #         List of strings where each item is a token. (e.g. ['the', 'cat']) or string that is split on white space
    #
    #     max_len : int
    #         Number of words in generated sequence
    #
    #     gpu : bool
    #         If you're using a GPU or not...
    #
    #     context_length : int
    #         Amount of words that get input as "context" into the model. Set to 0 for no limit
    #
    #     beam_width : int
    #         How many new word indices to try out...computationally expensive
    #
    #     verbose : int
    #         0: Print nothing to console
    #         1: Print currently generated word number to console
    #         2: Print currently generated word number and all currently considered song options to console
    #
    #     temperature : float
    #         Hyperparameter for adjusting the predicted probabilities by scaling the logits prior to softmax
    #
    #     top_k : int
    #         Number of song options to keep over each loop. This should normally be set to beam_width
    #
    #     audio : 2darray - 1 x n
    #         audio features for a song. `n` should equal the size of the
    #         multimodal features in `model`
    #
    #     Returns
    #     -------
    #     total_word_probs : list of list
    #         [context, potential_next_word, next_word_prob]
    #     """
    #     ####### get params from config ############################
    #     seed_text = self.deep_lyric.get_config('seed_text')
    #     max_len = self.deep_lyric.get_config('max_len')
    #     GPU = self.deep_lyric.get_config('GPU')
    #     context_length = self.deep_lyric.get_config('context_length')
    #     beam_width = self.deep_lyric.get_config('beam_width')
    #     verbose = self.deep_lyric.get_config('verbose')
    #     temperature = self.deep_lyric.get_config('temperature')
    #     top_k = self.deep_lyric.get_config('top_k')
    #     audio = self.deep_lyric.get_config('audio')
    #     ###########################################################
    #
    #     if isinstance(seed_text, str):
    #         seed_text = self.deep_lyric.tokenize(seed_text)
    #
    #     seed_text = self.deep_lyric.numericalize(seed_text)
    #
    #     # List of candidate word sequence. We'll maintain #beam_width top sequences here.
    #     # The context is a list of words, the scores are the sum of the log probabilities of each word
    #     self._context_and_scores = [[seed_text, 0.0]]
    #
    #     total_word_probs = []
    #
    #     # Loop over max number of words
    #     for word_number in range(max_len):
    #         if verbose==1 or verbose==2: print(f'Generating word: {word_number+1} / {max_len}')
    #
    #         candidates = []
    #         next_word_probs = []
    #
    #         # For each possible context that we've generated so far, generate new probabilities,
    #         # and pick an additional #beam_width next candidates
    #         for i in range(len(self._context_and_scores)):
    #             # Get a new sequence of word indices and log-probability
    #             # Example: [[2, 138, 661], 23.181717]
    #             context, score = self._context_and_scores[i]
    #             # Obtain probabilities for next word given the context
    #             probabilities = self.deep_lyric.get_text_distribution(context, context_length, temperature, GPU, audio)
    #
    #             # Multinomial draw from the probabilities
    #             if multinomial:
    #                 multinom_draw = np.random.multinomial(beam_width, probabilities)
    #                 top_probabilities = np.argwhere(multinom_draw != 0).flatten()
    #
    #             # no multinomial draw
    #             else:
    #                 top_probabilities = np.argsort(-probabilities)[:beam_width]
    #
    #             # For each possible new candidate, update the context and scores
    #             for j in range(len(top_probabilities)):
    #                 next_word_idx = top_probabilities[j]
    #                 new_context = context + [next_word_idx]
    #                 candidate = [new_context, (score - np.log(probabilities[next_word_idx]))]
    #                 candidates.append(candidate)
    #
    #                 # store predicted probablities
    #                 next_word_prob = probabilities[next_word_idx]
    #                 potential_next_word = self.deep_lyric.get_word_from_index(next_word_idx)
    #                 prior_context = [self.deep_lyric.get_word_from_index(w) for w in context]
    #                 next_word_probs.append((prior_context, potential_next_word, next_word_prob))
    #
    #         total_word_probs.extend(next_word_probs)
    #
    #         # Update the running tally of context and scores and sort by probability of each entry
    #         self._context_and_scores = candidates
    #         self._context_and_scores = sorted(self._context_and_scores, key = lambda x: x[1]) #sort by top entries
    #
    #         self._context_and_scores = self._context_and_scores[:top_k] # only keep `top_k`
    #         self.best_song, self.best_score = self._context_and_scores[0]
    #
    #         if verbose==2:
    #             for context, score in self._context_and_scores:
    #                 self.print_lyrics(context)
    #                 print('\n')
    #
    #     total_word_probs = sorted(total_word_probs, key=lambda x: -x[2]) # sort by highest probabilities
    #     return total_word_probs
