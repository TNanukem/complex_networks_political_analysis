import networkx as nx
from itertools import product

# Passar o código para um script depois

def coalition(A, B, G, shortest_paths, att='party', verbose=False):
    if verbose:
        print(f'Calculating Coalition for {A} and {B}')
    nodesA = [x for x,y in G.nodes(data=True) if y[att] == A]
    nodesB = [x for x,y in G.nodes(data=True) if y[att] == B]
    
    den = len(list(product(nodesA, nodesB)))

    coalition = 0
    
    for a in nodesA:
        for b in nodesB:
            coalition += shortest_paths[a][b]
            
    return coalition / den

def fragmentation(A, G, shortest_paths, att='party', verbose=False):
    if verbose:
        print(f'Calculating Fragmentation for Party {A}')
    return coalition(A, A, G, shortest_paths, att, verbose)

def isolation(A, G, other_com, shortest_paths, att='party', verbose=False):
    if verbose:
        print(f'Calculating Isolation for Party {A}')
    d = 0
    mod_X = 0

    for X in other_com:
        nodesX = [x for x,y in G.nodes(data=True) if y[att] == X]
        
        d += coalition(A, X, G, shortest_paths, att, verbose)
        mod_X += len(nodesX)
    
    return d / mod_X