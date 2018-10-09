# alpha=()
# for letter1 in {a..b}; do
#   for letter2 in {a..z}; do
#     alpha+=( "metadata_part_$letter1$letter2" )
#   done
# done

bash start-dataproc-cluster.sh -w 12 -g us-west2-c; \
bash start-dataproc-job.sh gs:///genius-meta/metadata_part_at; \
gcloud dataproc clusters delete w210-capstone --quiet
