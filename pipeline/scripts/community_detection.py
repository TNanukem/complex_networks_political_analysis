import scipy

import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
from cdlib import algorithms
from cdlib.evaluation import variation_of_information, surprise, newman_girvan_modularity

import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm

def community_detection(G, algorithm='louvain'):
    if algorithm == 'louvain':
        G_part = nx_comm.louvain_communities(G, seed=42)
    elif algorithm == 'leiden':
        G_ = nx.convert_node_labels_to_integers(G, label_attribute='old')
        G_part = algorithms.leiden(G_)
        
        mapper = {i[0]: i[1]['old'] for i in G_part.graph.nodes(data=True)}

        communities = []
        for comm in G_part.communities:
            communities.append(set([mapper[i] for i in comm]))
            
    elif algorithm == 'surprise':
        G_ = nx.convert_node_labels_to_integers(G, label_attribute='old')
        G_part = algorithms.surprise_communities(G_)
        
        mapper = {i[0]: i[1]['old'] for i in G_part.graph.nodes(data=True)}

        communities = []
        for comm in G_part.communities:
            communities.append(set([mapper[i] for i in comm]))
            

    G_mod = newman_girvan_modularity(G_, G_part).score
    G_surp = surprise(G_, G_part).score
    
    return communities, G_mod, G_surp