from bs4 import BeautifulSoup
import os, requests, time, re, json, sys, csv
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

def return_top_hit(query, token, max_retry=5):
    """
    Args
      `query`: list of query terms (e.g. [artist name, song name])
      `token`: Auth token
    Returns
      Top hit meta data
    """
    assert type(query) == list
    base_url = "https://api.genius.com"
    headers = {'Authorization': f'Bearer {token}'}
    search_url = base_url + "/search"
    
    search_term = " ".join(query)
    params = {'q': search_term}
    
    # request with backoff
    for i in range(max_retry):
        response = requests.get(search_url,
                                params=params,
                                headers=headers)
        if response.status_code == 200:
            json = response.json()
            break
        elif i+1 <= max_retry:
            time.sleep(2 ** i)
            pass
        else:
            pass
    assert i+1 != max_retry, "Reached maximum retries."
    
    # if response is not none
    try:
        res = json.get('response', {}).get('hits', None).pop(0)
    except IndexError:
        return None
    
    # otherwise, parse response
    res_type = res.get('result', {}).get('url').split('-')[-1]
    
    if res.get('type') == 'song' and res_type == 'lyrics':
        return res
    else:
        return None

def extract_url(hit):
    """Extract URL"""
    return hit.get('result', {}).get('url')

def extract_lyrics(url, max_retry=5):
    
    # request with backoff
    for i in range(max_retry):
        page = requests.get(url)
        if page.status_code == 200:
            html = BeautifulSoup(page.content, "html.parser")
            break
        elif i+1 <= max_retry:
            time.sleep(2 ** i)
        else:
            pass
    assert i+1 != max_retry, "Reached maximum retries."
    
    lyrics = html.find("div", class_="lyrics").get_text()
    return lyrics

def remove_parentheses(phrase):
    return re.sub(r'\([^)]+\)', '', phrase).strip()
    
def create_metadata_file(outfile):
    if not os.path.isfile(outfile):
        with open(outfile, 'w') as f:
            wt = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            wt.writerow(['msd_id', 'song_id', 'song_url', 'artist_id', 'artist'])
    return None
    
def create_lyrics_file(outfile):
    if not os.path.isfile(outfile):
        with open(outfile, 'w', newline='') as f:
            wt = csv.writer(f, quoting=csv.QUOTE_MINIMAL,
                            doublequote=True)
            wt.writerow(['msd_id', 'lyrics'])
    return None
    
def get_metadata(songs, token, outfile, batchsize=100):
    """
    Args
      Songs (list(tuple)) : (`MSD ID`, [search, terms])
      token : genius API token
      outfile : csv file to dump API results to
      batchsize : spill to file every `n` rows
    Return
      True if total rows written to CSV == total rows in input
    """
    genius = []
    _rows = 0
    for ix, song in enumerate(tqdm(songs, desc="Genius Search API")):
        trackid, search_terms = song
        hit = return_top_hit(search_terms, token=token)
        if hit:
            song_id = hit.get('result', {}).get('id')
            song_url = hit.get('result', {}).get('url')
            artist_id = hit.get('result', {}).get('primary_artist', {}).get('id')
            artist = hit.get('result', {}).get('primary_artist', {}).get('name')
            tup = (trackid, song_id, song_url, artist_id, artist)
        else:
            tup = (trackid, None, None, None, None)
        genius.append(tup)

        # dump every n rows
        if ix % batchsize == batchsize-1:
            create_metadata_file(outfile)
            with open(outfile, 'a') as f:
                wt = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                wt.writerows(genius)
            _rows += len(genius)
            genius = []

    # dump remaining
    create_metadata_file(outfile)
    with open(outfile, 'a') as f:
        wt = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        wt.writerows(genius)
    _rows += len(genius)
    
    if len(songs) == _rows:
        return True, outfile

def get_lyrics(meta_file, outfile):
    """
    Args
      meta_file (path) : file produced by `get_metadata`
      outfile (path) : csv file to dump song lyrics
    """
    # get line count for progress bar
    with open(meta_file, 'r') as f:
        row_count = sum(1 for row in f)
    row_count -= 1
    
    # stream metadata file and output line by line
    with open(meta_file, 'r') as f:
        rd = csv.reader(f)
        # skip header
        next(rd)
        for row in tqdm(rd, desc="Genius Pull Lyrics", total=row_count):
            if row[2]: # if URL exists
                msd_id = row[0]
                url = row[2]
                track_lyrics = extract_lyrics(url)
                tup = (msd_id, track_lyrics)

                create_lyrics_file(outfile)
                with open(outfile, 'a', newline='') as f:
                    wt = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                    wt.writerow(tup)
                    
    return None
    
if __name__ == '__main__':
    
    load_dotenv()
    GENIUS = os.getenv("GENIUS")

    # assert len(sys.argv) == 2,\
    # 'Required arg missing: MSD unique tracklist'
    
    ut = '../../data/external/MillionSongSubset/AdditionalFiles/subset_unique_tracks.txt'

    df = pd.read_csv(ut, sep='<SEP>', header=None, engine='python',
                     names=['trackid', 'songid', 'artist', 'title'])
    
    # build search terms from track list
    songs = []
    for row in df.itertuples(index=False):
        artist = remove_parentheses(str(row.artist).lower())
        title = remove_parentheses(str(row.title).lower())
        tup = (row.trackid, [artist, title])
        songs.append(tup)
        
    _success, metafile = get_metadata(songs, token=GENIUS, batchsize=500,
                            outfile='../../data/interim/genius_metadata.csv')
                        
    assert _success, "Input rows don't match output rows"
    
    get_lyrics(metafile, outfile='../../data/interim/genius_lyrics.csv')
