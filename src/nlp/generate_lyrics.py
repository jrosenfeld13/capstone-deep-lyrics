import numpy as np
import nltk.tokenize
from fastai import *
from fastai.text import *

from copy import copy, deepcopy
from enum import Enum

from datetime import datetime
import json
import requests
from pandas.io.json import json_normalize

def get_model(model_name, GPU=True):
    """
    Retrieve model from google cloud storage
    """
    if GPU:
        model_url = f'https://storage.googleapis.com/w210-capstone/models/{model_name}_architecture.pkl'
    else:
        model_url = f'https://storage.googleapis.com/w210-capstone/models/{model_name}_architecture_cpu.pkl'
    
    model = requests.get(model_url)
    model = model.content
    model = pickle.loads(model)
    return model

def get_itos(model_name):
    """
    Retrieve itos from google cloud storage
    """
    itos_url = f'https://storage.googleapis.com/w210-capstone/models/{model_name}_itos.pkl'
    itos = requests.get(itos_url)
    itos = itos.content
    itos = pickle.loads(itos)
    return itos

def get_preprocessor(model_name):
    """
    Retrieve preprocessor from google cloud storage
    """
    preprocessor_url = f'https://storage.googleapis.com/w210-capstone/models/{model_name}_preprocessor.pkl'
    preprocessor = requests.get(preprocessor_url)
    preprocessor = preprocessor.content
    preprocessor = pickle.loads(preprocessor)
    return preprocessor

class DeepLyric:
    """
    Generate deep lyrics given model and weights
    """
    DEFAULT_CONFIG = {
        'seed_text': None,
        'max_len': 40,
        'GPU': True,
        'context_length': 30,
        'beam_width': 3,
        'verbose': 0,
        'temperature': 1.5,
        'top_k': 3,
        'audio': None,
        'multinomial': True,
        'genre': None,
        'title': None
    }
    def __init__(self, model, itos=None, weights=None, model_type='language',
                 model_name=None, preprocessor=None, GPU=True):
        """
        Parameters:
        -----------
        model : str or PyTorch model or FastAI model abstraction
            Model object from PyTorch or FastAI.
            Provides model architecture and forward pass
        
        weights : filepath or url for .pth weight file.
            Weights for corresponding `model`
            *Currently Not Implemented*
            
        itos : nparray or list
            Language model int to string lookup
            Only required if Torch model is directly loaded to `model`
            
        model_type : str
            Indicates if model is one of the following
            - 'language' : pure language model
            - 'multimodal' : multimodal model; pre/post architecture is defined
                by `model`
                
        model_name : str
            Optional model name if Torch model is directly loaded to `model`
            If None the model name will be missing in the metadata output
            
        GPU : bool
            If machine has GPU or not
        """
        # initialize config dictionary to default
        self.set_config(config_dict=copy(self.DEFAULT_CONFIG))
        self.set_config('model_name', model_name)
        
        # load model and itos and preproc
        if isinstance(model, str):
            self.set_config('model_name', model)
            self.model = get_model(model, GPU=GPU)
            self.itos = get_itos(model)
            self.preprocessor = get_preprocessor(model)
        else:
            self.model = model
            self.itos = itos
            self.preprocessor = preprocessor
        
        # stoi init
        self.stoi = {v:k for k,v in enumerate(self.itos)}
        
        # model type init
        self.model_type = model_type
        self.set_config('model_type', model_type)
        
        if self.model_type == 'multimodal':
            self._get_multimodal_size()
            
    @property
    def config(self):
        return self._config
        
    def get_config(self, key):
        try:
            return self.config[key]
        except KeyError:
            raise KeyError(f"Missing one or more required parameter: {key}")
        
    def set_config(self, key=None, value=None, config_dict=None):
        """
        Set configuration (e.g. hyperparameters) for deep lyric object
        
        Parameters:
        -----------
        key : str
            configuration key, e.g. context, tempterature, beam_width, etc.
        value : str
            configuration value
        config_dict : dict
            dictionary of {`key`: `value`} for passing in ALL parameters
        """
        if not config_dict:
            self._config[key] = value
        else:
            self._config = config_dict
    
    def _create_initial_context(self):
        """
        Creates initial context for `generate_text()` based on
        `seed_text`, `genre`, and `title` config params
        
        Note: There are some limitations with the way we have tokenized genres
        and put them directly into the language models
        
        We notice that in the case of only genre, the seed is appended immediately
        following the `xtitle` tag. Also, when no genre or title is set,
        we append the seed immediately following `xbos`.
        
        These slight variations to the tag patterns could cause variations in the
        predicted probabilities that differ from the domain.
        """
        
        if self.get_config('genre'):
            genre = self.tokenize(self.get_config('genre'))
            
        if self.get_config('title'):
            title = self.tokenize(self.get_config('title'))
            
        if self.get_config('seed_text'):
            seed = self.tokenize(self.get_config('seed_text'))
            

        if self.get_config('genre') and self.get_config('title'):
            init_context = ['xbos', 'xgenre']+genre+['xtitle']+title+['xbol-1']
        elif self.get_config('genre'):
            genre = self.get_config('genre')
            init_context = ['xbos', 'xgenre']+[genre]+['xtitle']
        elif self.get_config('title'):
            title = self.get_config('title')
            init_context = ['xbos', 'xgenre', 'nan', 'xtitle']+title+['xbol-1']
        else:
            init_context = ['xbos']
            
        if seed:
            # append custom seed text
            init_context += seed
            
        return init_context

    def _vectorize_audio(self):
        """
        Converts the dataframe of audio features into feature vector
        using `self.preprocessor`.
        """
        audio_df = self.get_config('audio')
        if audio_df is not None:
            audio_features = self.preprocessor.transform(audio_df)
        else:
            audio_features = None
            
        return audio_features
        
    def _get_multimodal_size(self):
        """
        Decoder dimension - Encoder dimension = multimodal size
        Assumes we are only generating for post-RNN model.
        
        Returns:
        --------
        multimodalsize : int
        """
        enc_name = 'encoder.weight'
        dec_name = 'multidecoder.decoder.weight'
        enc_size = self.model.state_dict()[enc_name].shape[1]
        dec_size = self.model.state_dict()[dec_name].shape[1]
        
        self._multimodal_size = dec_size - enc_size
        
    def numericalize(self, t):
        "Convert a list of tokens `t` to their ids."
        return [self.stoi.get(w, 0) for w in t]

    def textify(self, nums, sep=' '):
        "Convert a list of `nums` to their tokens."
        return sep.join([self.itos[i] for i in nums])
    
    def get_word_from_index(self, idx):
        "like `textify` but for single word"
        return self.itos[idx]
    
    def tokenize(self, context):
        "Properly tokenize a string of words"
        tk = nltk.tokenize.LineTokenizer(blanklines='keep')
        context = tk.tokenize(context)

        re_tk = nltk.tokenize.RegexpTokenizer(r'\[[^\]]+\]|\w+|[\d\.,]+|\S+',
                                              discard_empty=False)
        context = re_tk.tokenize_sents(context)[0]

        context = [word.lower() for word in context]
        return context
    
    def save_json(self, dir=None, name=None, out=False, format_lyrics=False):
        """
        Saves generated lyric and `self.config` to json file in `dir`
        
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
            lyric : ['these', 'are', 'lyric', 'tokens']
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
        if format_lyrics:
            song = self.pretty_format(song)
        
        payload = {'meta': self.config, 'lyric': song}
        
        if dir:
            full_path = f"{dir}/{name}"
            with open(full_path, "w") as f:
                json.dump(payload, f, indent=4)
                
        if out:
            return payload
            
    def pretty_format(self, tokens):
        """
        Converts lyrics element of list into str with applied formatting
        
        Parameters:
        -----------
        Tokens : list(`str`)
            Tokenized strings of generated text
            *Note* input signature not the same as `print_lyrics`
            
        Returns:
        --------
        words : `str`
            Pretty formatted string
        """
                
        output = []
        for word in tokens:
            
            if word == 'xeol':
                word = '\n'
            elif word == 'xbol-1':
                word = '\n\n'
            elif 'xbol' in word:
                continue
            elif word =='xbos':
                word = 'SONG START\n'
            elif word == 'xtitle':
                word ='\n title:'
            elif word == 'xgenre':
                word ='genre:'
            elif word == 'xeos':
                word == 'SONG END'
                break
                
            output.append(word)
            
        return ' '.join(output)

    def print_lyrics(self, context=[]):
        """
        Directly print lyrics to standard output
        
        Parameters:
        -----------
        context : list or nparray of word indices
        """
        if not context:
            context = self.best_song
        
        for i in range(len(context)):
            step = context[i]
            word = self.textify([step])

            if word == 'xeol':
                word = '\n'
            elif word == 'xbol-1':
                print('\n')
                word == ''
            elif 'xbol' in word:
                continue
            elif word =='xbos':
                word = 'SONG START\n'
            elif word == 'xtitle':
                word ='\n title:'
            elif word == 'xgenre':
                word ='genre:'
            elif word == 'xeos':
                print('SONG END')
                break
                
            print(word, end=' ')
   
    def get_text_distribution(self, context, context_length, temperature, GPU, audio):
        """
        Produces predicted probabiltiies for the next word
              
        Parameters:
        -----------
        context : list or nparray
            context on which the next word prediction is based
            
        context_length : int
            `context`[-context_length:]
            
        temperature : float
            Hyperparameter for adjusting the predicted probabilities by scaling
            the logits prior to softmax
    
        audio : 2darray - 1 x n
            audio features for a song. `n` should equal the size of the
            multimodal features in `model`
            
        Returns:
        ----------
        List of probabilities with length of vocab size
        """
        if GPU:
            context = LongTensor(context[-context_length:]).view(-1,1).cuda()
        else:
            context = LongTensor(context[-context_length:]).view(-1,1).cpu()
        
        context = torch.autograd.Variable(context)
        
        self.model.reset()
        self.model.eval()

        # forward pass the "context" into the model
        if self.model_type == 'language':
            result, *_ = self.model(context)
        
        elif self.model_type == 'multimodal':
            audio_size = self._multimodal_size
            
            if audio is None:
                audio_features = Tensor([0]*audio_size*len(context))\
                    .view(-1, 1, audio_size).cuda()
            else:
                audio_features = np.tile(audio, len(context))
                audio_features = Tensor(audio_features)\
                    .view(-1, 1, audio_size).cuda()
            
            result, *_ = self.model(context, audio_features)
        
        
        result = result[-1]

        # set unk and pad to 0 prob
        # i.e. never pick unknown or pad
        result[0] = -np.inf
        result[1] = -np.inf

        # softmax and normalize
        probabilities = F.softmax(result/temperature, dim=0)
        probabilities = np.asarray(probabilities.detach().cpu(), dtype=np.float)
        probabilities /= np.sum(probabilities)
        return probabilities
    
    def generate_text(self):
        #, seed_text='xbos', max_len=40, GPU=False,
        #context_length=30, beam_width=3, verbose=1,
        #temperature=1.5, top_k=3, audio=None):
        """
        Primary function used to compose lyrics for a song
        
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
        ####### get params from config ############################
        # seed_text = self.get_config('seed_text')
        seed_text = self._create_initial_context()
        max_len = self.get_config('max_len')
        GPU = self.get_config('GPU')
        context_length = self.get_config('context_length')
        beam_width = self.get_config('beam_width')
        verbose = self.get_config('verbose')
        temperature = self.get_config('temperature')
        top_k = self.get_config('top_k')
        # audio = self.get_config('audio')
        audio = self._vectorize_audio()
        multinomial = self.get_config('multinomial')
        ###########################################################
        
        if isinstance(seed_text, str):
            seed_text = self.tokenize(seed_text)
        
        seed_text = self.numericalize(seed_text)
        
        # List of candidate word sequence. We'll maintain #beam_width top sequences here.
        # The context is a list of words, the scores are the sum of the log probabilities of each word
        self._context_and_scores = [[seed_text, 0.0]]
        
        # Loop over max number of words
        for word_number in range(max_len):
            if verbose==1 or verbose==2: print(f'Generating word: {word_number+1} / {max_len}')

            candidates = []
            
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
            
            # Update the running tally of context and scores and sort by probability of each entry
            self._context_and_scores = candidates
            self._context_and_scores = sorted(self._context_and_scores, key = lambda x: x[1]) #sort by top entries

            self._context_and_scores = self._context_and_scores[:top_k] #for now, only keep the top 15 to speed things up but we can/should change this to beam_width or something else
            self.best_song, self.best_score = self._context_and_scores[0]

            if verbose==2:
                for context, score in self._context_and_scores:
                    self.print_lyrics(context)
                    print('\n')

                    
    def get_predicted_probs(self):
        #, seed_text='xbos', max_len=40, GPU=False,
        #context_length=30, beam_width=3, verbose=1,
        #temperature=1.5, top_k=3, audio=None):
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
        
        ####### get params from config ############################
        # seed_text = self.get_config('seed_text')
        seed_text = self._create_initial_context()
        max_len = self.get_config('max_len')
        GPU = self.get_config('GPU')
        context_length = self.get_config('context_length')
        beam_width = self.get_config('beam_width')
        verbose = self.get_config('verbose')
        temperature = self.get_config('temperature')
        top_k = self.get_config('top_k')
        # audio = self.get_config('audio')
        audio = self._vectorize_audio()
        multinomial = self.get_config('multinomial')
        ###########################################################
        
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
