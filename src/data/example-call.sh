# alpha=()
# for letter1 in {a..b}; do
#   for letter2 in {a..z}; do
#     alpha+=( "metadata_part_$letter1$letter2" )
#   done
# done

bash start-dataproc-cluster.sh -w 12 -g us-central1-f; \
bash start-dataproc-job.sh gs:///genius-meta/metadata_part_ag; \
gcloud dataproc clusters delete w210-capstone --quiet

# verify what is causing failure in metadata_part_as
# is it the IP of a worker? Or is it a specific request that keeps failing?
