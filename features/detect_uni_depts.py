# Networks
from webscraping.shallow_network_scrape import NetworkScrape
import networkx as nx
import community

# Regular python
from collections import Counter
from collections import defaultdict
import pandas as pd
import os
import json
import boto3
import io
import numpy as np
import random

# NLP
from gensim.models import FastText as ft
from nltk.corpus import stopwords
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Other ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from datetime import datetime


def is_this_lang(word, lang):
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
    for k, v in ns.nodes.items():
        depths.add(nx.shortest_path_length(ns, homepage, k))

    # Get dead nodes, with no new links below a shallow depth
    nodes_to_remove = []
    for k, v in ns.nodes.items():
        # First check whether there was a data collection error
        if 'status' in v:
            nodes_to_remove.append(k)
            continue
        depth = nx.shortest_path_length(ns, homepage, k)
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
    # size = float(len(set(partition.values())))
    df = pd.DataFrame([dict(url=node, community=group,
                            depth=nx.shortest_path_length(ns, homepage, node),
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
    wv_words = [word for word in vocab if (word in model.vocab)
                and (word not in stopwords_)]

    # Convert the word vector space into a network
    wv_graph = nx.Graph()
    for word in wv_words:
        distances = model.distances(word, wv_words)
        for w, d in zip(wv_words, distances):
            if w == word:
                continue
            if d < network_threshold:
                if w in wv_graph and word in wv_graph:
                    continue
                wv_graph.add_edge(word, w)

    # Use Louvain network detection to find clusters
    wv_communities = community.best_partition(wv_graph.to_undirected())
    df_wv = pd.DataFrame([dict(word=node, community=group)
                          for node, group in zip(wv_graph.nodes(),
                                                 wv_communities.values())])

    # Remove non-native-language clusters
    wv_clusters = {}
    for cluster, grouped in df_wv.groupby("community"):
        # Small groups are basically noise
        if len(grouped) < min_cluster_size:
            continue
        if language_code is None:
            continue
        # Reject groups containing less than <native_frac>% Native words
        n_native = 0
        n_not_native = 0
        kill = False
        for word in grouped["word"]:
            if is_this_lang(word, language_code):
                n_native += 1
                if n_native > native_frac*len(grouped):
                    break
            else:
                n_not_native += 1
                if n_not_native > (1-native_frac)*len(grouped):
                    kill = True
                    break
        if kill:
            print(list(grouped["word"].values))
            continue
        # Assign language clusters to website nodes (to be used for prediction)
        for word in grouped["word"]:
            wv_clusters[word] = cluster
    return wv_clusters


def prepare_community_labels(df, wv_clusters):
    """Build one-hot vector of wv clusters for each URL"""

    # Add the clusters as one-hot columns
    for cluster_id in sorted(set(wv_clusters.values())):
        df[cluster_id] = 0

    # Hot up the one-hot columns, if the wv cluster is in the URL
    for irow, row in df.iterrows():
        all_text = row["link_text"].lower().split()
        if not pd.isnull(row['url_title']):
                all_text += row["url_title"].lower().split()
        # Assign one-hots
        for w in all_text:
            if w not in wv_clusters:
                continue
            cluster_id = wv_clusters[w]
            df[cluster_id].iat[irow] = 1


def fit(df, wv_clusters, rf_args={"n_jobs": 4, "random_state": 0,
                                  "n_estimators": 50}):
    """Bootstrap communities with respect to each other in order
    to find wv cluster which is most predictive of each community"""

    # Iterate through communities
    community_departments = {}
    drop_cols = ["depth", "has_kw", "url", "url_title", "link_text"]
    word_cols = sorted(set(wv_clusters.values()))
    for com_id, grouped in df.groupby("community"):
        # Make a copy of this iteration's community, and
        # a random sample to predict against
        this_community = grouped.copy()
        _n = len(grouped)
        condition = df.community != com_id
        if _n > condition.sum():
            _n = condition.sum()
        if _n == 0:
            continue
        other_communities = df.loc[condition].sample(_n).copy()
        # Assign target variable as True/False
        this_community["community"] = 1
        other_communities["community"] = 0
        # Concatenate the samples, and remove unrequired columns
        df_train = pd.concat([this_community, other_communities])
        df_train.drop(drop_cols, axis=1, inplace=True)
        # Fit a classifier
        clf = RandomForestClassifier(**rf_args)
        clf.fit(df_train.drop(["community"], axis=1), df_train["community"])
        # Extract the language groups in terms of their importance...
        language_groups = {}
        for i, f in enumerate(clf.feature_importances_):
            c = word_cols[i]
            language_groups[c] = f
        # ... and assign to communities by ID
        community_departments[com_id] = language_groups
        # Clean up
        del this_community
        del other_communities
        del df_train
    return community_departments


def transform(df, wv_clusters, community_departments):
    # Invert the word vector clusters mapping
    iclusters = defaultdict(list)
    for w, c in wv_clusters.items():
        iclusters[c].append(w)

    # Add three new columns to the dataframe
    df["most_descriptive_language_group"] = None
    df["most_descriptive_language_text"] = None
    df["most_descriptive_language_text_en"] = None

    for com_id, lang_importance in community_departments.items():
        # Build an indexer for this community
        condition = df.community == com_id
        # I can't remember why I wanted to only use bottom depth... so comment
        # it out for now
        # condition = condition & (df.depth == df.loc[condition,"depth"].max())
        # Get the most imporant language group
        max_importance = max(lang_importance.values())        
        lang = [L for L, i in lang_importance.items()
                if i == max_importance][0]        
        # Find most commonly occuring words in this group for this community
        word_counts = defaultdict(int)
        for col in ["link_text", "url_title"]:
            for sentence in df.loc[condition, col].values:
                if pd.isnull(sentence):
                    continue
                for word in sentence.lower().split():
                    if word in iclusters[lang]:
                        word_counts[word] += 1
        # Assign this information to the community rows
        most_common = ", ".join(w for w, c
                                in Counter(word_counts).most_common(10))
        df.loc[df.community == com_id,
               "most_descriptive_language_group"] = lang
        df.loc[df.community == com_id,
               "most_descriptive_language_text"] = most_common

    return iclusters


def s3io(obj):
    '''Generate an BytesIO from an S3 object'''
    obj_io = obj.get()['Body'].read()
    bytes_io = io.BytesIO(obj_io)
    return bytes_io


def chunks(whole, n_chunks):
    '''Randomly chunk up an iterable'''
    # Make sure that it makes sense to chunk up the object
    if n_chunks > len(whole) or n_chunks <= 0:
        yield whole
        return

    # Copy the iterable (we'll delete it later anyway) and shuffle it
    whole = whole.copy()
    random.shuffle(whole)

    # Calculate the chunk sizes
    whole_size = len(whole)
    chunk_size = int(whole_size / n_chunks)
    remainder = whole_size % n_chunks

    # Chunk it up
    for start in range(0, n_chunks):
        end = (start + 1)*chunk_size
        # Add the remainder for the final chunk
        if start == n_chunks - 1:
            end += remainder
        yield whole[start*chunk_size: end]
    # Clean up
    del whole


def get_isolated_words(wv_words, strapsize, model, radius):
    '''Find words which are isolated in a Euclidean vector space'''
    isolated_words = set()
    n_chunks = int(len(wv_words) / strapsize)
    for chunk in chunks(wv_words, n_chunks):
        for word in chunk:
            distances = model.wv.distances(word, chunk)
            mind = min(d for w, d in zip(chunk, distances) if w != word)
            # If there are no nearby words
            if mind > radius:
                isolated_words.add(word)
    return isolated_words


def dict_to_json(filename, data_dict):
    '''Write dict as json'''
    with open(filename, "w") as f:
        txt = json.dumps(data_dict)
        f.write(txt)


class ModelsManager(dict):
    '''Container for gensim models, to avoid loading models needlessly'''

    def __init__(self, wv_path, iso2langcode, iso2langname, grid_df):
        self.wv_path = wv_path
        self.iso2langcode = iso2langcode
        self.iso2langname = iso2langname
        self.grid_df = grid_df
        super().__init__()

    def get_model(self, key):
        # Get grid info
        grid_id = ".".join(key.split(".")[0:-2])
        condition = self.grid_df["ID"] == grid_id
        country_code = self.grid_df.loc[condition, 'country_code'].values[0]
        print("Retrieving",
              grid_df.loc[condition].to_dict(orient="records"))
        # Get the model and stopwords
        langcode = self.iso2langcode[country_code]
        langname = self.iso2langname[country_code]
        model = self[langcode]
        _stopwords = stopwords.words(langname)
        return model, _stopwords

    def __getitem__(self, key):
        if key not in self:
            path = self.wv_path.format(key)
            print("Loading FastText for", key)
            value = ft.load_fasttext_format(path)
            self[key] = value
        return super().__getitem__(key)


class StatsReporter(dict):
    '''Convenience method for reporting statistics on the fly'''
    def __setitem__(self, key, value):
        text = " ".join(key.split("_"))
        print("\tFound", value, text)
        super().__setitem__(key, value)


if __name__ == "__main__":
    # Prepare data for ModelsManager
    s3 = boto3.resource("s3")
    wv_path = "/Users/jklinger/Downloads/wiki.{}.bin"
    # TODO: Don't hard code these for full scale-up
    iso2langname = dict(IT="italian", GB="english")
    iso2langcode = dict(IT="it", GB="en")
    obj = s3.Object("nesta-inputs", "university_urls.csv")
    grid_df = pd.read_csv(s3io(obj))
    # Construct the manager
    mod_mgr = ModelsManager(wv_path, iso2langcode, iso2langname, grid_df)
    strapsize = 10000
    outpath = ""

    # Get shallow scrape outputs
    bucket = s3.Bucket('shallow-scrape')
    objects = iter(bucket.objects.all())
    object_keys = [obj.key for obj in objects]

    print("Found", len(object_keys), "keys")
    before = datetime.now()
    for iobj, objkey in enumerate(object_keys):
        filekey = ".".join(objkey.split(".")[0:-1])
        if any(x.startswith(filekey) for x in os.listdir()):
            continue

        obj = s3.Object("shallow-scrape", objkey)

        # Output container for processing statistics
        stats = StatsReporter()
        model, _stopwords = mod_mgr.get_model(objkey)

        # Read the network
        print("Reading the network")
        obj_io = s3io(obj)
        ns, homepage = read_and_prune_network(obj_io)
        del obj_io
        stats["nodes"] = len(ns.nodes())
        if stats["nodes"] < 5000:
            print("Skipping")
            print("------------------\n\n")
            continue

        # Find communities
        print("Finding communities")
        df = find_communities(ns, homepage)
        stats["communities"] = len(set(df.community))

        # Generate vocab
        print("Generating vocab")
        vocab = generate_vocab(df)
        wv_words = [word for word in vocab if (word in model.wv.vocab)
                    and (word not in _stopwords)]
        stats["words_in_vocab"] = len(wv_words)

        # Prune the vocab
        print("Pruning vocab")
        isolated_words = get_isolated_words(wv_words, strapsize,
                                            model, radius=0.4)
        clean_words = [w for w in wv_words if w not in isolated_words]
        stats["clean_words"] = len(clean_words)

        # WV dimensionality reduction
        print("Reducing dimensionality of word vector")
        vectors = [model.wv[w] for w in clean_words]
        fitted = PCA(n_components=0.75).fit_transform(vectors)
        stats["wv_components"] = len(fitted[0])

        # Get synonym clusters
        print("Generating synonym clusters")
        clusterer = AffinityPropagation()
        sample = fitted
        if len(fitted) > strapsize:
            sample = random.sample(list(fitted), strapsize)
        # Fit to the sample, but predict out-of-sample
        clusterer.fit_predict(sample)
        results = clusterer.predict(fitted)
        wv_clusters = {w: r for r, w in zip(results, clean_words)}
        stats["clusters"] = len(set(results))

        # Assign labels
        print("Assigning community labels")
        prepare_community_labels(df, wv_clusters)
        community_groups = fit(df, wv_clusters)
        iclusters = transform(df, wv_clusters, community_groups)

        # Make final cuts, based on label features
        print("Applying label cut")
        df_final = df
        langs = {c: iclusters[c]
                 for c in set(df_final.most_descriptive_language_group)}
        stats["final_nodes"] = len(df_final)
        stats["final_communities"] = len(set(df_final.community))

        # Generate stats about community sizes
        col = "community"
        stats["mean_com_size"] = np.mean([len(_df)
                                          for _, _df in df.groupby(col)])
        stats["mean_com_size_final"] = np.mean([len(_df)
                                                for _, _df in
                                                df_final.groupby(col)])

        # To disk
        filename = "{}_urls.csv".format(filekey)
        cols = ["url", "community", "most_descriptive_language_group"]
        df_final[cols].to_csv(filename, index=False)
        dict_to_json("{}_stats.json".format(filekey), stats)
        dict_to_json("{}_langs.json".format(filekey), langs)

        print("--->", (datetime.now() - before).total_seconds(),
              "seconds since start")
        print("--------------------\n\n")

    client = boto3.client('s3')
    for fname in os.listdir():
        if not (fname.endswith("_stats.json")
                or fname.endswith("_langs.json")
                or fname.endswith("_urls.csv")):
            continue
        s3.meta.client.upload_file(fname, 'shallow-scrape-labelled', fname)
