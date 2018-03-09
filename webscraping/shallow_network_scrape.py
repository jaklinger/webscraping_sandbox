from bs4 import BeautifulSoup
from collections import defaultdict
import multiprocessing
import networkx as nx
import requests
import sys
import time


def read_parallel(url):
    soup_text = None
    # Fill the soup with the request text                                       
    headers = {'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/50.0.2661.102 Safari/537.36')}
    try:
        r = requests.get(url,headers=headers,timeout=5)
        status = r.status_code        
        if status == 200:
            soup_text = r.text
    except Exception as err:
        status = str(err)
    return url, status, soup_text


class NetworkScrape(nx.DiGraph):
    def __init__(self,top_url,max_depth=3,kw_depth=1,
                 keywords=[],url_substring="",n_proc=1):
        nx.DiGraph.__init__(self)
        self.top_url = standardise_url(top_url)
        self.new_keys = [self.top_url]
        self.url_substring = url_substring
        self.add_node(self.top_url,link_text="",has_kw=False)
        self.depth = 0
        self.soup = None
        self.max_depth = max_depth
        self.kw_depth = kw_depth
        self.keywords = keywords
        self.session = None
        self.timers = defaultdict(int)
        self.n_proc = n_proc
        if n_proc > 1:
            self.pool = multiprocessing.Pool(n_proc)
        else:
            self.session = requests.session()

    def read(self, url):
        start = time.time()
        self.soup = None
        # Set the top_url is not already set                                        
        if self.top_url is None:
            self.top_url = url
        # Fill the soup with the request text                                       
        headers = {'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/50.0.2661.102 Safari/537.36')}

        try:
            r = self.session.get(url,headers=headers,timeout=5)
            status = r.status_code        
            if status == 200:
                self.soup = BeautifulSoup(r.text, 'html.parser')
        except Exception as err:
            status = str(err)
        self.timers["read"] += time.time() - start
        return status

    '''Get all links on the current web page'''
    def get_links(self):
        start = time.time()
        link_urls = []
        link_texts = []
        for link in self.soup.find_all('a', href=True):
            url = link["href"]
            url = standardise_url(url)
            # Ignore unknown URL prefixes                                                
            if not self.url_substring in url:
                continue
            # Ignore emails:                                                             
            if "@" in url:
                continue
            # Ignore files                                                               
            if any(url.endswith(x) for x in (".png",".jpeg",".jpg",
                                             ".txt",
                                             ".xlsx",".xls",".csv",
                                             ".pdf",".doc",".docx")):
                   continue
            # If this URL hasn't been found before, check whether it contains a keyword  
            need_to_append = True
            link_urls.append(url)
            link_texts.append(link.text)

        self.timers["get_links"] += time.time() - start
        return zip(link_urls, link_texts)


    def has_kw(self,key):
        if any(kw in key.lower() for kw in self.keywords):
            return True
        url_title = self.node[key]['url_title']
        if any(kw in url_title.lower() for kw in self.keywords):        
            return True
        return False

    def expand_network(self):
        new_keys = set()
        for ikey, key in enumerate(self.new_keys):
            #if ikey == 20:
            #    break
            status = self.read(key)
            if status != 200:
                self.node[key]['status'] = status
                continue
            title = ""
            if self.soup.title:
                title = str(self.soup.title.string)
                
            # Check if link contains keyword
            self.node[key]['url_title'] = title
            if not self.node[key]['has_kw']:
                self.node[key]['has_kw'] = self.has_kw(key)
            has_kw = self.node[key]['has_kw']
            if not has_kw and self.depth > self.kw_depth:
                continue                

            link_info = self.get_links()
            start = time.time()
            for link_url, link_text in link_info:
                if link_url in self:
                    _depth = nx.shortest_path_length(self,self.top_url,link_url)
                    if _depth <= self.depth:
                        continue
                new_keys.add(link_url)
                self.add_edge(key,link_url)
                self.node[link_url]["link_text"] = link_text
                self.node[link_url]["has_kw"] = has_kw
            self.timers['path_search'] += time.time() - start

        print(self.depth, len(new_keys), len(self))

        self.new_keys = new_keys
        self.depth += 1

    def expand_network_parallel(self):
        new_keys = set()
        start = time.time()
        results = self.pool.map(read_parallel, self.new_keys)
        self.timers['read'] += time.time() - start

        for key, status, soup_text in results:
            if self.top_url is None:
                self.top_url = url
            if status != 200:
                self.node[key]['status'] = status
                continue
            try:
                self.soup = BeautifulSoup(soup_text,"html.parser")
            except Exception as err:
                self.node[key]['status'] = str(err)
                continue

            title = ""
            if self.soup.title:
                title = str(self.soup.title.string)

            # Check if link contains keyword
            self.node[key]['url_title'] = title
            if not self.node[key]['has_kw']:
                self.node[key]['has_kw'] = self.has_kw(key)
            has_kw = self.node[key]['has_kw']
            if not has_kw and self.depth > self.kw_depth:
                continue                

            link_info = self.get_links()
            start = time.time()
            for link_url, link_text in link_info:
                if link_url in self:
                    _depth = nx.shortest_path_length(self,self.top_url,link_url)
                    if _depth <= self.depth:
                        continue
                new_keys.add(link_url)
                self.add_edge(key,link_url)
                self.node[link_url]["link_text"] = link_text
                self.node[link_url]["has_kw"] = has_kw
            self.timers['path_search'] += time.time() - start

        print(self.depth, len(new_keys), len(self))

        self.new_keys = new_keys
        self.depth += 1


    def run(self):
        while self.depth <= self.max_depth:
            if self.n_proc > 1:
                self.expand_network_parallel()
            else:
                self.expand_network()
        
    def pickle(self,filename):
        if self.n_proc > 1:
            self.pool.close()
            self.pool.terminate()    
            self.pool = None
        sys.setrecursionlimit(10000)
        nx.write_gpickle(self,filename)


def standardise_url(url):
    if url.endswith("/"):
        url = url[:-1]
    if not url.startswith("http"):
        url = "http://"+url
    return url.lower()



# top_url = "-"
# G.add_node(top_url,url=top_url)

# for i in range(0,3):
#     parent_url = str(i)
#     G.add_edge(top_url,parent_url)
#     G.node[parent_url]['url'] = "Not dummy"
#     for j in range(0, i):
#         child_url = str(i)+str(j)
#         G.add_edge(parent_url,child_url)
#         G.node[child_url]['url'] = "Dummy"

# for n in G.nodes(data=True):
#     print(n[0],n[1]["url"])
#     print(nx.shortest_path_length(G,top_url,n[0]))

# #print(G["1"])
# #print(G.keys())

if __name__ == "__main__":
    ns = NetworkScrape(top_url='http://www.unipd.it',
                       max_depth=4, url_substring="unipd.it",
                       kw_depth=1, keywords=['dipartiment','courses',
                                             'department','scuol',
                                             'school','cors'],
                       n_proc=4)
    ns.run()
    ns.pickle("unipd.pickle")

    print("------------------")
    from collections import Counter
    for x,t in Counter(ns.timers).most_common():
        print(x, t)
    print("------------------")
