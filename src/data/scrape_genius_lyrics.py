from bs4 import BeautifulSoup
import os, requests, time

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
        elif i+1 < max_retry:
            time.sleep(2 ** i)
            pass
        else:
            pass
    assert i+1 != max_retry, "Reached maximum retries."
            
    res = json.get('response', {}).get('hits', None).pop(0)
    if res.get('type') == 'song':
        return res
    else:
        return None

def extract_url(hit):
    """Extract URL"""
    return hit.get('result', {}).get('url')

def extract_lyrics(url):
    page = requests.get(url)
    html = BeautifulSoup(page.content, "html.parser")
    lyrics = html.find("div", class_="lyrics").get_text()
    return lyrics
