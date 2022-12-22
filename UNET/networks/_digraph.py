import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pydot
import re
from matplotlib import cm

class DiGraphDot:
	"""Base class for digraphs read from .dot files"""
	def __init__(self, path, plot=False, cmap=None):

		file = open(path, "r")
		new = ''.join([line.strip() for line in file])
		file.close()
		
		temp_path = self._get_temp(path)		
		fout = open(temp_path, "w")
		fout.write(new)
		fout.close()

		self._build_graph(temp_path)
		self.adj = nx.to_numpy_matrix(self.graph)
		self.N = self.adj.shape[0]
		if cmap: self._set_node_colors(cmap=cmap) 
		self.nodes = self.graph.nodes
	
	def _set_node_colors(self, cmap='viridis'):
		map = cm.get_cmap(cmap)
		node_colors = map(np.linspace(0,1,self.N))
		for n,node in enumerate(self.graph.nodes(data=True)):
			name, attr = node
			attr['color'] = node_colors[n]
	
	def _build_graph(self, dot_path):		

		temp_graph = pydot.graph_from_dot_file(dot_path)[0]
		self.graph = nx.DiGraph()
		nodes = temp_graph.get_nodes()
		edges = temp_graph.get_edges()

		for n,node in enumerate(nodes):
			name = node.get_name().replace('"', '')
			self.graph.add_node(name,color='red')
		for edge in edges:
		    src = edge.get_source().replace('"', '')
		    dst = edge.get_destination().replace('"', '')
		    val = edge.obj_dict['attributes']['value'].replace('"', '')
		    if val == '+': color = 'black'; weight = 1
		    if val == '-': color = 'gray'; weight = -1
		    self.graph.add_edge(src,dst,color=color,value=val, weight=weight)
		    
	def _get_temp(self, path):
		basepath, filename, ext = self._split_path(path)
		path = basepath + '/' + filename + '-tmp' + ext
		return path
		
	def _split_path(self, path):
		pathlist = re.split('/', path)
		basepath = '/'.join(pathlist[:-1])
		file, ext = pathlist[-1].split('.')
		ext = '.' + ext
		return basepath, file, ext

