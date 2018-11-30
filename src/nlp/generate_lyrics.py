import nltk.tokenize

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
        self.weights = weights
        self.itos = itos
        self.stoi = {v:k for k,v in enumerate(self.itos)}
        self.model_type = model_type
                
    def numericalize(self, t):
        "Convert a list of tokens `t` to their ids."
        return [self.stoi[w] for w in t]

    def textify(self, nums:Collection[int], sep=' '):
        "Convert a list of `nums` to their tokens."
        return sep.join([self.itos[i] for i in nums])
    
    def tokenize(self, context):
        tk = nltk.tokenize.LineTokenizer(blanklines='keep')
        context = tk.tokenize(context)

        re_tk = nltk.tokenize.RegexpTokenizer(r'\[[^\]]+\]|\w+|[\d\.,]+|\S+',
                                              discard_empty=False)
        context = re_tk.tokenize_sents(context)[0]
        return context
    
    def save_lyrics_to_file(self, dir):
        """
        Saves lyrics to specified `dir`
        """
        with open(f"{dir}", "w") as f:
            lyrics = [f'{word}\n' if word == 'xeol' else word
                   for word in self.textify(self.best_song)]
            f.write(''.join(lyrics))

    def print_lyrics(self, context):
        """
        Directly print lyrics to standard output
        """
        for i in range(len(context)):
            step = context[i]
            word = self.textify([step])

            if word == 'xeol':
                word = '\n'
            elif 'xbol' in word:
                continue
            elif word == 'xeos': 
                print(word)
                break
                
            print(word, end=' ')  
   
    def get_text_distribution(self, context, context_length, temperature):
        """
        Produces predicted probabiltiies for the next word
              
        Parameters:
        -----------
        context : list or nparray
            context on which the next word prediction is based
            
        context_length : int
            `context`[-context_length:]
            
        temperature : float
            Hyperparameter for adjusting the predicted probabilities by scaling the logits prior to softmax

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
        result, *_ = self.model(context)
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
    
    def generate_text(self, seed_text='xbos', max_len=100, GPU=False, context_length=30, beam_width=5, verbose=True, temperature=1, top_k=15):
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
        
        temperature : float
            Hyperparameter for adjusting the predicted probabilities by scaling the logits prior to softmax
          
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
        
        # Loop over max number of words
        for word_number in range(max_len):
            if verbose: print(f'Generating word: {word_number+1} / {max_len}')

            candidates = []
            
            # For each possible context that we've generated so far, generate new probabilities, 
            # and pick an additional #beam_width next candidates
            for i in range(len(self._context_and_scores)):
                # Get a new sequence of word indices and log-probability
                # Example: [[2, 138, 661], 23.181717]
                context, score = self._context_and_scores[i]
                # Obtain probabilities for next word given the context 
                probabilities = self.get_text_distribution(context, context_length, temperature)

                # Multinomial draw from the probabilities
                multinom_draw = np.random.multinomial(beam_width, probabilities)
                top_probabilities = np.argwhere(multinom_draw != 0).flatten()
                            
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

            if verbose:
                for context, score in self._context_and_scores:
                    self.print_lyrics(context)
                    print('\n')