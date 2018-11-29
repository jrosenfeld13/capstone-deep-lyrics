class DeepLyric:
    """
    Generate deep lyrics given model and weights
    """
    def __init__(self, model, weights, itos):
        """
        Parameters:
        -----------
        model : PyTorch model or FastAI model abstraction
            Model object from PyTorch or FastAI.
            Provides model architecture and forward pass
        
        weights : filepath or url for .pth weight file.
            Weights for corresponding `model`
            
        itos : nparray
            Language model int to string lookup
            
        model_type : str
            Indicates if model is one of the following
            - 'language' : pure language model
            - 'pre-multi' : multimodal model prior to RNN layers
            - 'post-multi' : multimodal model after RNN layers
    
        """
        self.model = model
        self.weights = model_weights
        self.itos = itos
        self.model_type = model_type
        
    def generate_step(self, context, context_length, temp=1.0):
        """
        Generates next word
        
        Parameters:
        -----------
        context : list or nparray
            context on which the next word prediction is based
            
        context_length : int
            `context`[-context_length:]
            
        temp : float
            hyperparameter for adjusting the predicted probabilities
            (attach source here)
        
        """
        pass
        
    def get_word_from_index(self, idx):
        """
        Converts word integer to string
        
        Parameters:
        -----------
        idx : int
        
        """
        pass
        
    def save_lyrics_to_file(self, dir):
        """
        Saves lyrics to specified `dir`
        """
        pass
        
    def print_lyrics(self):
        """
        Directly print lyrics to standard output
        """
        pass
    
    def generate_text(self):
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
        
        verbose : bool
            If True, prints every possible context for a given word cycle

        Returns
        -------
        context_and_scores : list of lists
            Returns a sorted list of the entire tree search of contexts and their respective scores in the form:
            [[context, score], [context, score], ..., [context, score]]
        
                
        """
        pass
        
    def get_text_distribution(self):
        """
        Produces predicted probabiltiies for the next word
        
        """
        pass
