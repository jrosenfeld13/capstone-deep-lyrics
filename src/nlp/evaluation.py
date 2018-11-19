
import pickle
import numpy as np
import nltk
from nltk.translate import bleu_score 
from collections import Counter
from collections import defaultdict
import string
import copy
#!pip install pronouncing
import pronouncing
import numpy as np

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

### Trigram perplexity

def load_model(file_name):
    '''input file name (without extention)'''
    # load model
    m1_pkl = open("../data/models/" + file_name + ".pkl", "rb")
    model = pickle.load(m1_pkl)
    m1_pkl.close()
    return model
    
    
def get_entropy(text, model, _n=3, _lpad = ['<s>'], _rpad = ['<s>']):
    ''' calculate average log probability of each word in text, given context
        
        IMPORTANT NOTE:  For initial implementation, we do not have bigram or unigram prob in our dict 'model',
                         and this handles missing or unknown entries naively
    '''
    
    e = 0.0
    padded_string = "<s> " + example1 + " <s>"
    text = padded_string.split(' ')
    for i in range(_n - 1, len(text)):
        context = tuple(text[i - _n + 1:i])
        token = text[i]
        #print(context,token)
        #print(e)
        e += -np.log2(model.get(context,dict()).get(token,0.0000001))  # this is a poor placeholder until we get backoff dicts
    entropy = e / float(len(text) - (_n - 1))
    return entropy

def get_perplexity(text, model, _n = 3, _lpad = ['<s>'], _rpad = ['<s>']):
    return np.power(2,get_entropy(text=text, model=model , _n=_n, _lpad=_lpad, _rpad=_rpad))


### POS tagging

def nltkPOS(text,verbose=False):
    '''For an input text, return absolute difference from published proportions, between 0 and 1.'''
    
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
    absdiff = 0
    
    # prepare data  
    text = str(text)
    tokenized_text = nltk.word_tokenize(text)
    tag_list = nltk.pos_tag(tokenized_text)
    
    if not tag_list:
        raise ValueError("Please provide more complete text")
    
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
        if verbose==True: 
            print(k,"- benchmark:",comp_dict[k],", text:",pos_dict.get(k,0),"abs diff:",abs(comp_dict[k] - pos_dict.get(k,0)))
    return absdiff


### Rhyme

# some basic preprocessing techniques
def prepString(s):
    '''removes punctuation other than apostrophes from string'''
    return str(s).lower().translate({ord(c): None for c in string.punctuation if c not in ("'")})

def removePunc(s):
    '''removes punctuation from string'''
    return str(s).lower().translate({ord(c): None for c in string.punctuation})
    
    
# calculate rhyme density
def calcRhymeDensity(text,rhymeType='perfect',rhymeLocation='all',lineStartStop=(1,-2),printExamples=False):
    '''calculates rhyme density (count of rhymes over n-1 words). \n\n
    
       _parameters_
       text: input text for measurement
       rhymeType: 'perfect' is a perfect rhyme, 'vowel' is a rhyming in the vowel sound + stress only
       rhymeLocation: choose to look at 'all' text, 'section' by line numbers, or 'end' (last word in each line)    
       lineStartStop: tuple of (start,stop) line numbers
       printExamples: if True, print most common values of the selected rhymeType
       
       _returns_
       rhyme_cnt: count of rhymes of specified rhymeType and rhymeLocation
       wordCount: count of words of specified rhymeType and rhymeLocation
       rhymeDensity: rhyme_cnt/float(wordCount-1)
    '''
    # restrict location to (end=last word, internal line = line, all= full text)
    # count tokens
    # 
    
    # initialize
    rhymePart_cnt = Counter()
    rhyme_cnt = 0
    
    # prepare data
    text = prepString(text)
    if rhymeLocation == 'all':
        words = text.split()
    
    if rhymeLocation == 'end':
        lines = text.split("\n")
        words = [line.split()[-1] for line in lines]
    
    if rhymeLocation == 'section':
        lines = text.split("\n")
        words = [line.split()[-1] for line in lines[lineStartStop[0]:lineStartStop[1]+1]]
    
    # 
    wordCount = len(words)
    for word in words:
        pros = pronouncing.phones_for_word(word)
        if pros:     
            phonelist = pros[0]  #using first pronunciation for now
            if len(phonelist) > 0:
                if rhymeType == 'perfect':
                    rhymePart_cnt[pronouncing.rhyming_part(phonelist)] +=1

                #if rhymeType == 'rime':
                #    pass
                #if rhymeType == 'soft':
                #    pass
                #if rhymeType == 'consonant':
                #    pass

                elif rhymeType == 'vowel':
                    rhymePart_cnt[pronouncing.rhyming_part(phonelist).split()[0]] +=1
    
    for v in rhymePart_cnt.values():
        rhyme_cnt += v-1
    
    if wordCount>1: 
        rhymeDensity = rhyme_cnt/float(wordCount-1)
    else:
        rhymeDensity = 0.0
    
    if printExamples == True:
        print(rhymePart_cnt.most_common(5))
    
    return rhyme_cnt, wordCount, rhymeDensity

       
### BLEU

def bleu(ref_list,candidateText,nGram=4,nGramType='cumulative',shouldSmooth=True):
    '''calculates BLEU score 
    
        _parameters_
        ref_list: expects a list of reference texts to compare (as strings)
        candidateText: the new text needing to be scored
        nGram: choose between 1-4.  Determines which ngram(s) to use in the scoring
        nGramType: 'cumulative' uses a simple average of all ngrams from 1 to nGram
        shouldSmooth: if False, calculates the BLEU score without smoothing. Recommended to use smoothing (set to True)
        
        _returns_
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
    candidate = [removePunc(str(candidateText)).split()]
    references = [[removePunc(str(ref)).split() for ref in ref_list]]
    weights = weight_dict[(nGramType,nGram)]
       
    
    # scoring
    if shouldSmooth==True:
        smoother = bleu_score.SmoothingFunction().method7
    else:
        smoother = None
    score = bleu_score.corpus_bleu(references, candidate, weights, smoothing_function=smoother)
    #print(score)
    return score


### Meter

def findLineStress(line):
    line = prepString(line)
    words = line.split()
    wordCount = len(words)
    parses = ['']
    for word in words:
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

    return list(set(parses)), [len(parse) for parse in list(set(parses))], wordCount


def levenshtein(s1, s2):
    '''calculate levenshtein distance for two input strings
    
    _parameters_
    s1: first input string
    s2: second input string
    
    _returns_
    distance: levenshtein distance between two strings...that is, the lowest number of modifications to turn s1 into s2
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
    
def findMeter(text):
    '''finds meter with smallest edit distance
    
    _parameters_
    text: input text, usually a poem of some kind
    
    _returns_
    lowest: lowest edit distance for any standard accentual-syllabic verse
    options: list of potential meters for the lowest edit distance.
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
    text = prepString(text)
    lines = text.split('\n')
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
    
    #options = min(vote_cnt, key=vote_cnt.get)  #chooses one in the event of ties
    lowest = min(vote_cnt.values()) 
    options = [k for k,v in vote_cnt.items() if v==lowest]
    return lowest, options #, vote_cnt    






