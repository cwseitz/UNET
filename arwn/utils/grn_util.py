import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pydot

def build_linear_system(adj):
    n = adj.shape[0]
    mat = np.zeros((2*n,2*n))
    gam = np.ones((2*n,2*n))
    np.fill_diagonal(mat, -0.1*np.ones((2*n,)))
    mat[n:,:n] = 0.1*np.diag(np.ones((n,)))
    mat[:n,n:] = 0.1*adj
    mat *= gam
    return mat

def build_graph(dir, filename, ext, plot=True):

    #strip newlines in a temp file to prevent extra nodes from being added
    file = open(dir+filename+ext, "r")
    new = ""
    for line in file:
        line = line.strip()
        new += line

    file.close()
    fout = open(dir+filename+'-tmp'+ext, "w")
    fout.write(new)
    fout.close()

    #construct the graph by hand using pydot
    graphs = pydot.graph_from_dot_file(dir+filename+'-tmp'+ext)
    graph = graphs[0]

    #iterate over nodes
    nx_graph = nx.DiGraph()
    nodes = graph.get_nodes()
    edges = graph.get_edges()

    for node in nodes:
        nx_graph.add_node(node.get_name().replace('"', ''))

    for edge in edges:
        src = edge.get_source().replace('"', '')
        dst = edge.get_destination().replace('"', '')
        val = edge.obj_dict['attributes']['value'].replace('"', '')
        if val == '+': color = 'blue'; weight = 1
        if val == '-': color = 'red'; weight = -1
        nx_graph.add_edge(src,dst,color=color,value=val, weight=weight)

    if plot:
        colors = [nx_graph[u][v]['color'] for u,v in nx_graph.edges]
        pos = nx.circular_layout(nx_graph)
        nx.draw(nx_graph,pos=pos,node_color='cornflowerblue',edge_color=colors,with_labels=True)
        plt.show()

    return nx_graph
