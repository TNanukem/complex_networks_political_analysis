import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import networkx.algorithms.community as nx_comm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from edgeseraser import polya, noise_score
from unimodal import MLF

def backbone_extract(G, weight, method):
    if method == 'disparity_filter':
        g = disparity_filter(G, weight)
        g = apply_extraction(g)
    elif method == 'polya_urn_filter':
        g = polya_urn_filter(G, weight)
    elif method == 'noise_corrected':
        g = noise_corrected(G, weight)
    elif method == 'marginal_likelihood_filter':
        g = marginal_likelihood_filter(G, weight)
        g = apply_extraction(g)
    else:
        raise Exception('Method not available')
        
    return g

def disparity_filter(G, weight):
    g = nx.Graph()
    nodes = [i for i in G.nodes(data=True)]
    
    for n in G:
        k = len(G[n])
        
        if k > 1:
            sum_w = sum(np.absolute(G[n][v][weight]) for v in G[n])
            for v in G[n]:
                w = float(np.absolute(G[n][v][weight]))
                p_ij = w / sum_w
                
                alpha = 1 - (k-1) * scipy.integrate.quad(lambda x: (1-x)**(k-2), 0, p_ij)[0]
                
                g.add_edge(n, v, weight=w, alpha=float(alpha))
    g.add_nodes_from(nodes)
                
    return g

def apply_extraction(N):
    total_nodes = len(N.nodes())
    alphas = [i[2]['alpha'] for i in N.edges(data=True)]
    thresholds = np.linspace(min(alphas), max(alphas), 30)
    perc = []

    for threshold in thresholds:
        M = N.copy()
        edges_to_remove = [i for i in M.edges(data=True) if i[2]['alpha'] >= threshold]
        M.remove_edges_from(edges_to_remove)

        component_size = max([len(i) for i in nx.connected_components(M)])

        perc.append(component_size / total_nodes)

    d = np.array(perc)
    final_perc = min(d[d>=0.8])

    alpha = thresholds[np.where(d == final_perc)][0]

    M = N.copy()
    edges_to_remove = [i for i in M.edges(data=True) if i[2]['alpha'] >= alpha]
    M.remove_edges_from(edges_to_remove)

    return M

def polya_urn_filter(G, weight):
    _, prob = polya.filter_nx_graph(G)

    df = nx.to_pandas_edgelist(G)

    df['alpha'] = 1 - prob
    g = nx.from_pandas_edgelist(df, edge_attr=[weight, 'alpha'])

    return g

def noise_corrected(G, weight):
    M = G.copy()
    h = noise_score.filter_nx_graph(M, param=1.64, save_scores=False)
    return M

def marginal_likelihood_filter(G, weight):
    df = nx.to_pandas_edgelist(G)
    df.rename(columns={weight: 'weight'})
    mlf = MLF(directed=False)
    df_edgelist_2 = mlf.fit_transform(df)
    df_edgelist_2.rename(columns={'significance': 'alpha'}, inplace=True)

    g = nx.from_pandas_edgelist(df_edgelist_2, edge_attr=['weight', 'alpha'])
    
    return g

def topological_metrics(G, Gb):
    metrics = pd.DataFrame()
    
    original_degree = [i[1] for i in G.degree()]
    new_degree = [i[1] for i in Gb.degree()]
    
    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    axs[0].hist([i for i in original_degree])
    axs[0].set_title('Original Degree Distribution')
    axs[1].hist([i for i in new_degree])
    axs[1].set_title('Backbone Degree Distribution')
    
    original_clustering = [i for i in nx.clustering(G).values()]
    new_clustering = [i for i in nx.clustering(Gb).values()]
    
    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    axs[0].hist([i for i in original_clustering])
    axs[0].set_title('Original Clustering Coefficient Distribution')
    axs[1].hist([i for i in new_clustering])
    axs[1].set_title('Backbone Clustering Coefficient Distribution')
    
    G_part = nx_comm.louvain_communities(G, seed=42)
    Gb_part = nx_comm.louvain_communities(Gb, seed=42)
    
    G_mod = nx_comm.modularity(G, G_part)
    Gb_mod = nx_comm.modularity(Gb, Gb_part)

    metrics = metrics.append({
        'G_Connected_Components':len([i for i in nx.connected_components(G)]),
        'Gb_Connected_Components':len([i for i in nx.connected_components(Gb)]),
        'G_Biggest_Component_Size': max([len(i) for i in nx.connected_components(G)]),
        'Gb_Biggest_Component_Size':max([len(i) for i in nx.connected_components(Gb)]),
        'G_Density':nx.density(G), 'Gb_Density':nx.density(Gb),
        'G_Transitivity':nx.transitivity(G), 'Gb_Transitivity':nx.transitivity(Gb),
        'G_Modularity': G_mod, 'Gb_Modularity': Gb_mod
    }, ignore_index=True)
    
    return metrics
    
def contextual_metrics(Gb, test_set, metadata_to_use=None):
    df = pd.DataFrame()
    for edge in [i for i in Gb.edges(data=True)]:
        edge[2]['n1'] = edge[0]
        edge[2]['n2'] = edge[1]
        df = df.append(edge[2], ignore_index=True)
        
    if metadata_to_use is not None:
        df = df[metadata_to_use]
        
    df = df.drop_duplicates()
    df['edge_id'] = df['n1'].astype(str) + '_' + df['n2'].astype(str)
    
    train_set = df[~df['edge_id'].isin(test_set)]
    test_set = df[df['edge_id'].isin(test_set)]
    
    train_set.drop(columns=['n1', 'n2'], inplace=True)
    train_set.set_index(['edge_id'], inplace=True)
    test_set.drop(columns=['n1', 'n2'], inplace=True)
    test_set.set_index(['edge_id'], inplace=True)
    
    reg = LinearRegression()
    reg.fit(train_set.drop(columns=['weight']), train_set['weight'])
    
    preds = reg.predict(test_set.drop(columns=['weight']))
    
    r2 = r2_score(test_set['weight'], preds)
    mse = mean_squared_error(test_set['weight'], preds)
    
    return {'r2': r2, 'mse': mse}