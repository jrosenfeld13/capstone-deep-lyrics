import trigram
    
def main():
    lm = trigram.SimpleTrigramLM(words=None, probas_file='../../data/models/trigram-weights')
    lm.generate_text(max_length=500)
    
if __name__ == '__main__':
    main()