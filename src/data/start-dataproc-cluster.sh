#!/bin/bash

gcloud dataproc clusters create w210-capstone\
  --metadata "JUPYTER_CONDA_PACKAGES=pandas:tqdm:beautifulsoup4:python-dotenv" \
  --bucket w210-capstone \
  --subnet default \
  --zone us-central1-a \
  --master-machine-type n1-standard-1 \
  --master-boot-disk-size 80 \
  --num-workers 2 \
  --worker-machine-type n1-standard-1 \
  --worker-boot-disk-size 80 \
  --image-version 1.2 \
  --project w261-215522 \
  --initialization-actions \
      'gs://dataproc-initialization-actions/jupyter/jupyter.sh'
  
