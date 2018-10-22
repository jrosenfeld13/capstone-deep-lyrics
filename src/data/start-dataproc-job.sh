#!/bin/bash

# Usage: bash start-dataproc-job.sh gs:///genius-meta/metadata_part_aw

# number of clusters
if [ "$1" != "" ]; then
    METAPART=$1
else
    echo "Input metadata part number"
    echo "e.g. ``gs:///genius-meta/metadata_part_aw``"
    exit 1
fi

echo "submitting job for " $METAPART
gcloud dataproc jobs submit pyspark \
  --cluster w210-capstone \
  gs://w210-capstone/src/genius-scrape-pyspark.py \
  -- $METAPART gs:///distributed-scrape
