#!/bin/bash

# # number of clusters
# if [ "$1" == "" ]; then
#   echo "ERROR: Input number of workers to provision"
#   exit 1
# else
#   WORKERS=$1
# fi

# option parse
while getopts "g:w:" opt; do
  case ${opt} in
    g)
      ZONE="${OPTARG}"
      ;;
    w)
      WORKERS="${OPTARG}"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# default options
if [[ -z "$ZONE" ]]; then
  ZONE=us-central1-a
else
  echo 'Creating cluster in ' $ZONE
fi

echo Create $WORKERS workers in $ZONE

gcloud dataproc clusters create w210-capstone \
  --metadata "JUPYTER_CONDA_CHANNELS=conda-forge,JUPYTER_CONDA_PACKAGES=pandas:tqdm:beautifulsoup4:python-dotenv" \
  --bucket w210-capstone \
  --subnet default \
  --zone $ZONE \
  --master-machine-type n1-standard-1 \
  --master-boot-disk-size 80 \
  --num-workers $WORKERS \
  --worker-machine-type n1-standard-1 \
  --worker-boot-disk-size 80 \
  --image-version 1.2 \
  --project w261-215522 \
  --initialization-actions \
      'gs://dataproc-initialization-actions/jupyter/jupyter.sh'
  
