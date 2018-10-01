from . import trigram
import argparse

def generate_text(infile, cloudstore=False, **kwargs):
    '''            
    Downloads model weights, instantitates a Language Model, and generates a song.
    This is the main function that talks with the swagger API. It should also eventually
    integrate things like genre, tone, danceability, etc.
    
    Parameters
    ----------
    infile : str
    `infile` is a str representing a path to a pkl file that corresponds with a LM
    
    Returns
    ----------
    text : str
    `text` represents a full song as a string, to be passed onto the API
    '''
    
    lm = trigram.SimpleTrigramLM(words=None, infile=infile, cloudstore=cloudstore)
    text = lm.generate_text(max_length=500)
    return text
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default='../../data/models/trigram-weights.pkl')
    parser.add_argument('-w', action='store_true', default=False,
                        dest='cloudstore',
                        help='Set web/cloudstore to true')    
    args = parser.parse_args()
    generate_text(infile=args.infile, cloudstore=args.cloudstore)