import requests
import json
from bs4 import BeautifulSoup

def scrape_website_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator='\n')
        title_tag = soup.find('title')
        cleaned_text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])
        return (cleaned_text,title_tag.get_text(strip=True))
    else:
        return None,None