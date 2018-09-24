import trigram
    
def main():
    lm = trigram.SimpleTrigramLM(words=None, probas_file='../../data/models/trigram-weights')
    lm.generate_text()
    
if __name__ == '__main__':
    main()