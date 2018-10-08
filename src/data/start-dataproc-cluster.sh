#!/bin/bash

# number of clusters
if [ "$1" != "" ]; then
    WORKERS=$1
else
    WORKERS=2
fi

gcloud dataproc clusters create w210-capstone \
  --metadata "JUPYTER_CONDA_CHANNELS=conda-forge,JUPYTER_CONDA_PACKAGES=pandas:tqdm:beautifulsoup4:python-dotenv" \
  --bucket w210-capstone \
  --subnet default \
  --zone us-central1-a \
  --master-machine-type n1-standard-1 \
  --master-boot-disk-size 80 \
  --num-workers $WORKERS \
  --worker-machine-type n1-standard-1 \
  --worker-boot-disk-size 80 \
  --image-version 1.2 \
  --project w261-215522 \
  --initialization-actions \
      'gs://dataproc-initialization-actions/jupyter/jupyter.sh'
  
