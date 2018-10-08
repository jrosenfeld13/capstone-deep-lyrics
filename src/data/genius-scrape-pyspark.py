#!/usr/bin/python
import os
import subprocess
import sys
import requests
from bs4 import BeautifulSoup
import time

import pyspark
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("Genius Test Program") \
    .getOrCreate()
sc = spark.sparkContext

# test paths in GCS
# metapath = 'gs:///genius-meta/genius_metadata.csv'
# metapath = 'gs:///test-data/genius-meta/genius_metadata_tail.csv'
# lyricspath = 'gs:///test-data/genius-lyrics'

########################## Actual web scrape method ##########################
def extract_lyrics(url, max_retry=5):
    
    # request with backoff
    for i in range(max_retry):
        page = requests.get(url)
        if page.status_code == 200:
            html = BeautifulSoup(page.content, "html.parser")
            break
        elif page.status_code == 404:
            return None
        elif i+1 <= max_retry:
            time.sleep(2 ** i)
        else:
            pass
    assert i+1 != max_retry, "Reached maximum retries."
    
    lyrics = html.find("div", class_="lyrics").get_text()
    return lyrics
##############################################################################

metapath = sys.argv[1]
lyricspath = sys.argv[2]

# 1. First we gather a list of all lyrics that have previously been scraped

scraped1 = spark.read.format("CSV")\
    .option("header","true")\
    .option("multiLine", "true")\
    .option("escape", '"')\
    .load('gs:///genius-lyrics/genius_lyrics.csv')
scraped2 = spark.read.format("CSV")\
    .option("header","true")\
    .option("multiLine", "true")\
    .option("escape", '"')\
    .load('gs:///subset-10k/genius_lyrics.csv')
# check if we've already started scraping and file has been initialized
try:
    subprocess.check_call(['hdfs', 'dfs', '-test', '-d', lyricspath])
    lyrics_exist = True
except:
    lyrics_exist = False

if lyrics_exist:
    scraped3 = spark.read.format("CSV")\
        .option("header","true")\
        .option("multiLine", "true")\
        .option("escape", '\\')\
        .load(lyricspath)

df1 = scraped1.select('msd_id')
df2 = scraped2.select('msd_id')
if lyrics_exist:
    df3 = scraped3.select('msd_id')
    df_all = df1.union(df2).union(df3).distinct().cache()
else:
    df_all = df1.union(df2).distinct().cache()

# broadcast previously pulled keys
pulled_keys = df_all.rdd.map(lambda x: x[0]).collect()
pulled_keys_bc = sc.broadcast(pulled_keys)

print("Previously Scraped: ", len(pulled_keys))
    
# 2. Filter out keys that have already been pulled
meta = sc.textFile(metapath)
meta = meta.map(lambda x: (x.split(',')[0], x.split(',')[2]))\
    .filter(lambda x: x[0] != 'msd_id')\
    .filter(lambda x: x[1] != '')\
    .filter(lambda x: x[0] not in pulled_keys_bc.value).cache()
    
print("Now Scraping: ", meta.count())

# 3. Scrape Lyrics and dump to GCS
if meta.count() > 0:
    meta.repartition(10)
    rdd = meta.mapValues(extract_lyrics)
    df = rdd.toDF(['msd_id', 'lyrics'])
    # df2 = df.collect()
    df.write.format("CSV")\
        .option('header', 'true')\
        .option("multiLine", "true")\
        .mode('append')\
        .save(lyricspath)
