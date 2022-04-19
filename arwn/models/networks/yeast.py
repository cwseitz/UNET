from sentinel.util import *
import os

def load_yeast_example():
    cwd = os.getcwd()
    dir = '/'.join(cwd.split('/')[:-1]) + '/data/'
    filename = 'yeast'
    ext = '.dot'
    graph = build_graph(dir, filename, ext, plot=False)
    adj = nx.adjacency_matrix(graph).todense()
    return graph, adj
