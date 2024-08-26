import requests
import json
from bs4 import BeautifulSoup
import tqdm


def scrape_links(url):
    links = []
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and href.startswith('http'):
                href = href.split(" ")[0]
                links.append(href)
    except requests.exceptions.RequestException as e:
        print(f"Error scraping links from {url}: {e}")
    return links

def save_to_json(links, filename):
    with open(filename, 'w') as f:
        json.dump(links, f, indent=4)

def crawl(start_url, max_depth=1, output_format="json"):
    visited = set()
    queue = [(start_url, 0)]  # Store tuples of (url, current_depth)
    all_links = []

    while queue:
        url, depth = queue.pop(0)
        if depth > max_depth:
            continue
        if url not in visited:
            visited.add(url)
            links = scrape_links(url)
            all_links.extend(links)
            if depth < max_depth:
                for link in links:
                    if link not in visited:
                        queue.append((link, depth + 1))

    # Save all collected links
    save_to_json(all_links, "links_and_content.json")
    
    return set(all_links)

def scrape_website_text(url):
    # Send a request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()  # Completely remove the tags

        # Extract all text content
        text = soup.get_text(separator='\n')
        # soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the <title> tag
        title_tag = soup.find('title')

        # Clean up the text by removing extra whitespace
        cleaned_text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])

        return (cleaned_text,title_tag.get_text(strip=True))
    else:
        return None,None



def main_get_links(start_url):
    all_links = list(crawl(start_url, max_depth=0, output_format="json"))
    dataJson = []
    for i in tqdm.tqdm(all_links):
      data,title = scrape_website_text(i)
      if data is None:
        continue
      else:
        dataJson.append({
                      "url": i,
                      "title": title,
                      "content": data
                  })
      print(i)
    save_to_json(dataJson, 'Url_Title_Content.json')
    return dataJson
