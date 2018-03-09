from webscraping.shallow_network_scrape import NetworkScrape
import networkx as nx
#from networkx.algorithms import community
import community
from collections import Counter

print("Unpickling")
ns = nx.read_gpickle("unipd.pickle")

print("Generating communities")
#communities_generator = community.girvan_newman(ns)
partition = community.best_partition(ns.to_undirected())
size = float(len(set(partition.values())))

print(size)
print(Counter(partition.values()).most_common())

for node, group in zip(ns.nodes(), partition.values()):
    if "dipart" in str(node):
        print(group,node)
        

# print("Getting top level")
# top_level_communities = next(communities_generator)
# for group in sorted(map(sorted, top_level_communities)):
#     print(group)
#     break

# print("--------------------------")
# print("Getting next level")
# next_level_communities = next(communities_generator)
# for group in sorted(map(sorted, next_level_communities)):
#     print(group)
#     break
