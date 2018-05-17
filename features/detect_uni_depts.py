#TODO: always read/write/analyse directly in neo4j
# Networks
from webscraping.shallow_network_scrape import NetworkScrape
import networkx as nx
import community

# Regular python
from collections import Counter
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import random
import os
import json

# NLP
import gensim
from nltk.corpus import stopwords
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Other ML
from sklearn.ensemble import RandomForestClassifier

# Translation service
from google.cloud import translate

# Clients etc
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ec2-user/gcloud.json"
translate_client = translate.Client()


def is_this_lang(word,lang):
    """Function to test whether `word` is in the language `lang`"""
    try:
        return detect(word) == lang
    except LangDetectException:
        return False

def read_and_prune_network(network_scrape_path):
    """Read and prune a NetworkScrape object"""

    # Unpickle the network and find the homepage    
    ns = nx.read_gpickle(network_scrape_path)
    homepage = None
    for node in ns.nodes:
        # Homepage has no predecessors
        if len(ns.pred[node]) == 0:
            homepage = node
            break

    # Get maximum depth of the network       
    depths = set()
    for k,v in ns.nodes.items():
        depths.add(nx.shortest_path_length(ns,homepage,k))

    # Get dead nodes, with no new links below a shallow depth                          
    nodes_to_remove = []
    for k,v in ns.nodes.items():
        # First check whether there was a data collection error
        if 'status' in v:
            nodes_to_remove.append(k)
            continue
        depth = nx.shortest_path_length(ns,homepage,k)
        # Don't kill at large depths
        if depth > 3:
            continue
        if len(ns.succ[k]) == 0:
            nodes_to_remove.append(k)

    # Remove dead nodes
    for node in nodes_to_remove:
        ns.remove_node(node)
    return ns, homepage


def find_communities(ns, homepage):
    """Perform network detection and generate a dataframe of the results"""    

    partition = community.best_partition(ns.to_undirected())
    size = float(len(set(partition.values())))
    df = pd.DataFrame([dict(url=node,community=group,
                            depth=nx.shortest_path_length(ns,homepage,node),
                            **ns.nodes[node])
                       for node, group in zip(ns.nodes(), partition.values())])
    return df


def generate_vocab(df):
    """Generate vocabulary for the entire dataset"""
    vocab = defaultdict(list)
    for _,row in df.iterrows():
        all_text = row["link_text"].lower().split()
        if not pd.isnull(row['url_title']):
            all_text += row["url_title"].lower().split()
            for word in set(all_text):
                vocab[word].append(row["community"])
    return vocab

def find_word_clusters(vocab, model, network_threshold=0.4, 
                       min_cluster_size=20, native_frac=0.3,
                       language_code=None, stopword_language="english"):
    """Find clusters of words using community detection"""

    # Filter out words that don't exist in the WV model, or are stopwords
    stopwords_ = stopwords.words(stopword_language)
    wv_words = [word for word in vocab if (word in model) 
                and (word not in stopwords_)]

    # Convert the word vector space into a network
    wv_graph = nx.Graph()
    for word in wv_words:
        distances = model.distances(word,wv_words)
        for w,d in zip(wv_words,distances):
            if w == word:
                continue
            if d < network_threshold:
                if w in wv_graph and word in wv_graph:
                    continue
                wv_graph.add_edge(word,w)

    # Use Louvain network detection to find clusters
    wv_communities = community.best_partition(wv_graph.to_undirected())
    df_wv = pd.DataFrame([dict(word=node,community=group)
                          for node, group in zip(wv_graph.nodes(), wv_communities.values())])    

    # Remove non-native-language clusters
    wv_clusters = {}
    for cluster,grouped in df_wv.groupby("community"):
        # Small groups are basically noise
        if len(grouped) < min_cluster_size:
            continue
        if language_code == None:
            continue
        # Reject groups containing less than <native_frac>% Native words
        n_native = 0
        n_not_native = 0
        kill = False
        for word in grouped["word"]:
            if is_this_lang(word,language_code):
                n_native += 1
                if n_native > native_frac*len(grouped):
                    break
            else:
                n_not_native += 1
                if n_not_native > (1-native_frac)*len(grouped):
                    kill = True
                    break
        if kill:
            continue
        # Assign language clusters to website nodes (to be used for prediction)
        for word in grouped["word"]:
            wv_clusters[word] = cluster
    return wv_clusters


def prepare_community_labels(df,wv_clusters):
    """Build one-hot vector of wv clusters for each URL"""

    # Add the clusters as one-hot columns
    for cluster_id in sorted(set(wv_clusters.values())):
        df[cluster_id] = 0

    # Hot up the one-hot columns, if the wv cluster is in the URL
    for irow,row in df.iterrows():
        all_text = row["link_text"].lower().split()
        if not pd.isnull(row['url_title']):
                all_text += row["url_title"].lower().split()
        # Assign one-hots
        for w in all_text:
            if w not in wv_clusters:
                continue
            cluster_id = wv_clusters[w]
            df[cluster_id].iat[irow] = 1
    

def predict_community_labels(ns, df, wv_clusters, path_to_language_cluster,
                             rf_args={"n_jobs":4, "random_state":0, "n_estimators":50}):
    """Bootstrap communities with respect to each other in order 
    to find wv cluster which is most predictive of each community"""

    # Iterate through communities
    community_departments = {}
    word_cols = sorted(set(wv_clusters.values()))
    for com_id, grouped in df.groupby("community"):
        # Make a copy of this iteration's community, and a random sample to predict against
        this_community = grouped.copy()
        other_communities = df.loc[df.community != com_id].sample(len(grouped)).copy()
        # Assign target variable as True/False 
        this_community["community"] = 1
        other_communities["community"] = 0
        # Concatenate the samples, and remove unrequired columns
        df_train = pd.concat([this_community,other_communities])
        df_train.drop(columns=["depth","has_kw","url","url_title","link_text"],inplace=True)

        # Fit a classifier
        clf = RandomForestClassifier(**rf_args)
        clf.fit(df_train.drop(columns=["community"]), df_train["community"])
        # Extract the language groups in terms of their importance...
        language_groups = {}
        for i,f in enumerate(clf.feature_importances_):
            c = word_cols[i]
            language_groups[c] = f
        # ... and assign to communities by ID
        community_departments[com_id] = language_groups
        # Clean up
        del this_community
        del other_communities
        del df_train


    # Invert the word vector clusters mapping
    iclusters = defaultdict(list)
    for w,c in wv_clusters.items():
        iclusters[c].append(w)
    # Save for future reference
    with open(path_to_language_cluster, 'w') as fp:
        json.dump(iclusters, fp)

    # Add three new columns to the dataframe
    df["most_descriptive_language_group"] = None
    df["most_descriptive_language_text"] = None
    df["most_descriptive_language_text_en"] = None

    for com_id, lang_importance in community_departments.items():
        # Build an indexer for this community
        condition = df.community == com_id
        # I can't remember why this line was here... so comment it out for now
        #condition = condition & (df.depth == df.loc[condition,"depth"].max())
        # Get the most imporant language group
        lang = [L for L,i in lang_importance.items() 
                if i == max(lang_importance.values())][0]
        # Find the most commonly occuring words in this group for this community
        word_counts = defaultdict(int)
        for col in ["link_text","url_title"]:
            for sentence in df.loc[condition,col].values:
                if pd.isnull(sentence):
                    continue            
                for word in sentence.lower().split():
                    if word in iclusters[lang]:
                        word_counts[word] += 1
        # Assign this information to the community rows
        most_common = ", ".join(w for w,c in Counter(word_counts).most_common(5))
        translation = translate_client.translate(most_common,target_language='en')
        df.loc[df.community == com_id,"most_descriptive_language_group"] = lang
        df.loc[df.community == com_id,"most_descriptive_language_text"] = most_common
        df.loc[df.community == com_id,"most_descriptive_language_text_en"] = translation["translatedText"]


    # Reassign this new information to the original network
    cols_to_add = [c for c in df.columns if type(c) is str 
                   and c.startswith("most_descriptive_language")]
    cols_to_add += ["community"]
    for _,row in df.iterrows():
        for col in cols_to_add:
            ns.nodes[row['url']][col] = row[col]
            
def generate_compact_graph(df, path_to_save):
    """Generate a compacted version of the NetworkScrape graph, collapsing
    all community nodes into a single node.
    """
    # Iterate through communities
    com_network = nx.Graph()
    for com_id, grouped in df.groupby("community"):
        # Add this node if it doesn't already exist (it could have been added
        # from another edge)
        com_id = int(com_id)
        if com_id not in com_network.nodes:
            com_network.add_node(com_id)
        # Initialise the number of internal edges
        if "internal_edges" not in com_network.nodes[com_id]:
            com_network.nodes[com_id]["internal_edges"] = 0

        # Loop over 'internal nodes', i.e. those belonging to the community
        subject = grouped["most_descriptive_language_group"].values[0]
        internal_nodes = set(grouped.url)
        for node in internal_nodes:
            # Find internal and external neighbours
            neighbours = ns.neighbors(node)
            for nbour in neighbours:
                # Increment then ignore internal edges
                if nbour in internal_nodes:
                    com_network.nodes[com_id]["internal_edges"] += 1
                    continue
                # Otherwise add an edge to an external node
                ncom_id = int(df.loc[df.url == nbour,"community"].values[0])
                if com_network.has_edge(com_id,ncom_id):
                    com_network[com_id][ncom_id]['weight'] += 1
                else:
                    com_network.add_edge(com_id,ncom_id,weight=1)
        # Assign the group size, and the most descriptive language
        com_network.nodes[com_id]["size"] = len(grouped)
        com_network.nodes[com_id]["most_descriptive_language_group"] = int(subject)
    # Generate a parameter called 'internality' = n_internal_edges / n_total_edges
    for node in com_network.nodes:
        i = com_network.nodes[node]["internal_edges"]
        j = com_network.degree(node)
        com_network.nodes[node]["internality"] = float(i/(i+j))
    # Write the data to disk
    nx.write_gexf(com_network,path_to_save)

if __name__ == "__main__":

    # Inputs
    stopword_language = 'italian'
    pretrained_wv_path = '/dev/shm/wiki.it.vec'
    network_scrape_path = 'unipd.big.pickle'
    language = 'it'
    label = 'unipd'
    outpath = '/dev/shm/test'
    # Generate output paths
    path_to_language_cluster=outpath+'language_clusters_'+label+'.json'
    path_to_save=outpath+label+'_compacted_clusters.gexf'

    # Generate communities
    print("Generating communities")
    ns, homepage = read_and_prune_network(network_scrape_path)
    df = find_communities(ns, homepage)
    # Find and generate clusters of language
    print("Getting w2v model")
    model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_wv_path, binary=False)
    print("Generating language")
    vocab = generate_vocab(df)
    wv_clusters = find_word_clusters(vocab, model, language_code=language, 
                                     stopword_language=stopword_language)
    # Assign language to communities
    print("Predicting labels")
    prepare_community_labels(df,wv_clusters)
    predict_community_labels(ns, df, wv_clusters, path_to_language_cluster)
    # Save an output
    print("Getting final output")
    generate_compact_graph(df, path_to_save)
