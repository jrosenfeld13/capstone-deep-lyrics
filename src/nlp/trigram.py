import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

import nltk
import utils, vocabulary


from collections import defaultdict
import pickle

def normalize_counter(c):
    """Given a dictionary of <item, counts>, return <item, fraction>."""
    total = sum(c.values())
    return {w:float(c[w])/total for w in c}


class SimpleTrigramLM(object):
    def __init__(self, words, infile=None):
        """Build our simple trigram model."""
        #if pre-defined model is provided, use that as probabilities
        if infile:
            with open(infile, 'rb') as main_dict:
                self.probas = pickle.load(main_dict)
        
        else:
            # Raw trigram counts over the corpus. 
            # c(w | w_1 w_2) = self.counts[(w_2,w_1)][w]
            self.counts = defaultdict(lambda: defaultdict(lambda: 0.0))

            # Iterate through the word stream once.
            w_1, w_2 = None, None
            for word in words:
                if w_1 is not None and w_2 is not None:
                    # Increment trigram count.
                    self.counts[(w_2,w_1)][word] += 1
                # Shift context along the stream of words.
                w_2 = w_1
                w_1 = word
            
            # Normalize so that for each context we have a valid probability
            # distribution (i.e. adds up to 1.0) of possible next tokens.
            self.probas = defaultdict(lambda: defaultdict(lambda: 0.0))
            for context, ctr in self.counts.items():
                self.probas[context] = normalize_counter(ctr)
            
    def next_word_proba(self, word, seq):
        """Compute p(word | seq)"""
        context = tuple(seq[-2:])  # last two words
        return self.probas[context].get(word, 0.0)
    
    def predict_next(self, seq):
        """Sample a word from the conditional distribution."""
        context = tuple(seq[-2:])  # last two words
        pc = self.probas[context]  # conditional distribution
        words, probs = zip(*pc.items())  # convert to list
        return np.random.choice(words, p=probs)
    
    def score_seq(self, seq, verbose=False):
        """Compute log probability (base 2) of the given sequence."""
        score = 0.0
        count = 0
        # Start at third word, since we need a full context.
        for i in range(2, len(seq)):
            if (seq[i] == "<s>" or seq[i] == "</s>"):
                continue  # Don't count special tokens in score.
            s = np.log2(self.next_word_proba(seq[i], seq[i-2:i]))
            score += s
            count += 1
            # DEBUG
            if verbose:
                print("log P({:s} | {:s}) = {.03f}".format(seq[i], " ".join(seq[i-2:i]), s))
        return score, count
    
    def generate_text(self, max_length=40):
        seq = ["<s>", "<s>"]
        for i in range(max_length):
            seq.append(self.predict_next(seq))
            # Stop at end-of-sentence
            if seq[-1] == "</s>": break
        print(" ".join(seq))

        
        
# "canonicalize_word" performs a few tweaks to the token stream of
# the corpus.  For example, it replaces digits with DG allowing numbers
# to aggregate together when we count them below.
def sents_to_tokens(sents, wordset):
    """Returns a flattened list of the words in the sentences, with padding for a trigram model."""
    padded_sentences = (["<s>", "<s>"] + s + ["</s>"] for s in sents)
    # This will canonicalize words, and replace anything not in vocab with <unk>
    return np.array([utils.canonicalize_word(w, wordset=wordset) 
                     for w in utils.flatten(padded_sentences)], dtype=object)

def build_trigram(infile='../../data/interim/genius_lyrics.csv',
                  outfile='../../data/models/trigram-weights.pkl',
                  USE_DUMMY_DATA = False):
        '''            
        Downloads data and preprocesses for feeding into LM. The data *must* be a csv and have a column called 'lyrics'
        
        Parameters
        ----------
        infile : str
        `infile` is a str representing a path to a csv file with a single column titled 'lyrics'
        '''

        if USE_DUMMY_DATA:
            nltk.download('brown') #sample corpus from nltk
            corpus_object = nltk.corpus.brown
            words = corpus_object.words() #singe list of words
        else:
            lyrics = pd.read_csv(infile, usecols=['lyrics'])
            full_text = lyrics.lyrics.str.cat()
            words = full_text.split(' ')
            corpus_object = lyrics.lyrics

        train_sents, test_sents = utils.get_train_test_sents(corpus_object, split=0.8, shuffle=True)
        vocab = vocabulary.Vocabulary(utils.canonicalize_word(w) for w in utils.flatten(train_sents))

        print("Tokenizing sentences...")
        train_tokens = sents_to_tokens(train_sents, wordset=vocab.wordset)
        test_tokens = sents_to_tokens(test_sents, wordset=vocab.wordset)
        vocab = vocabulary.Vocabulary(utils.canonicalize_word(w) for w in utils.flatten(train_sents))

        print("Building trigram...")
        lm = SimpleTrigramLM(train_tokens)
        print("Built trigram...")

        with open(outfile, 'wb') as f:
            pickle.dump(dict(lm.probas), f)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default='../../data/external/songdata.csv')
    parser.add_argument('--outfile', type=str, default='../../data/models/trigram-weights.pkl')
    args = parser.parse_args()

    build_trigram(args.infile, args.outfile)
