import trigram
import argparse

def main(infile):
    #instantiate a trigram model with pretrained weights and generate text
    lm = trigram.SimpleTrigramLM(words=None, infile=infile)
    lm.generate_text(max_length=500)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default='../../data/models/trigram-weights.pkl')
    args = parser.parse_args()
    
    main(infile=args.infile)