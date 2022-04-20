from sentinel.util import *
import os

class YeastExample:
    def __init__(self):
        cwd = os.getcwd()
        dir = '/'.join(cwd.split('/')[:-1]) + '/data/'
        filename = 'yeast'
        ext = '.dot'
        self.graph = build_graph(dir, filename, ext, plot=False)
        self.adj = nx.adjacency_matrix(graph).todense()
        
