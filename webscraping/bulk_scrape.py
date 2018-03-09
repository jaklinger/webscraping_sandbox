'''
Bulk scrape any site down to a maximum depth.

TODO: Also implement Selenium scraper
'''
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from abc import ABC
from abc import abstractmethod

'''
Base class scraper, basically uses BeautifulSoup to get visible text and links
'''
class Scraper(ABC):
    def __init__(self, max_depth=3):
        self.soup = None
        self.urls = []
        self.top_url = None
        self.data = defaultdict(list)
        self.all_texts = set()
        self.max_depth = max_depth
        
    @abstractmethod
    def read(self,url):
        pass

    '''Get all links on the current web page'''
    def get_links(self):
        links = []
        for link in self.soup.find_all('a', href=True):
            url = link["href"]
            url = standardise_url(url)
            # Ignore unknown URL prefixes
            if not url.startswith(self.top_url):
                continue
            # Ignore files
            if any(url.endswith(x) for x in (".png",".jpeg",".jpg",".xlsx",".xls",".csv",
                                             ".pdf",".doc",".docx")):
                   continue                   
            # Don't double count URLs
            if url not in self.urls:
                links.append(url)
                self.urls.append(url)
        return links

    '''Get all visible text on the current web page'''
    def get_visible_text(self):
        texts = self.soup.find_all(text=True)
        visible_texts = filter(tag_visible, texts)  
        visible_texts = [t.strip() for t in visible_texts]
        # May need to revisit the encoding in the future
        visible_texts = [t for t in visible_texts if len(t) > 1]
        return visible_texts

    '''Recursively extract visible text down to depth = self.max_depth'''
    def recursive_extraction(self, url, depth=0):
        # Read the page
        self.read(url)
        # Get any new / unique visible text
        visible_text = []
        for text in self.get_visible_text():
            if text in self.all_texts:
                continue
            self.all_texts.add(text)
            visible_text.append(text)
        # Append the visible text at this depth
        self.data[depth] += visible_text
        # Recurse, if required
        if depth < self.max_depth:
            for link in self.get_links():
                self.recursive_extraction(link, depth+1)

'''Implementation of Scraper which reads using requests'''
class RequestsScraper(Scraper):
    def read(self, url):
        url = standardise_url(url)
        # Set the top_url is not already set
        if self.top_url is None:
            self.top_url = url
        # Fill the soup with the request text
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        r = requests.get(url,headers=headers)
        r.raise_for_status()
        self.soup = BeautifulSoup(r.text, 'html.parser')

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 
                               'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def standardise_url(url):
    if url.endswith("/"):
        url = url[:-1]
    if not url.startswith("http"):
        url = "http://"+url 
    return url.lower()

if __name__ == "__main__":
    rs = RequestsScraper(max_depth=3)
    rs.recursive_extraction(url="http://www.consultoria-ti.com.mx/")
    for depth, data in rs.data.items():
        print(depth)
        for text in data:
            print("\t",text)
        print("\n--------------------------\n")

