(Choose A or B)

## A. From scratch
### 1) Train model from csv in cloudstore
 `python trigram.py`
 
 * args: [--infile] [--outfile]

### 2) Generate text from trained model just created:
`python generate_text.py`

 * args: [-w] [--infile]


## B. Pretrained
### Generate text from pre-trained weights stored on cloudstore:
 `python generate_text.py -w --infile https://storage.googleapis.com/capstone-deep-lyrics/trigram-weights.pkl`

