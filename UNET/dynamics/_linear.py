from ..networks import *
from arwn import backend
from matplotlib.colors import to_hex
import numpy as np
import networkx as nx
import sys
np.set_printoptions(threshold=sys.maxsize)

class DynamicsBase:
    def __init__(self,N,T,Nt,trials,Nrecord):
        self.N = N
        self.T = T
        self.Nt = Nt
        self.dt = self.N/self.Nt
        self.trials = trials
        self.Nrecord = Nrecord
        self.shape = (self.N,self.trials,self.Nt)

class LinearDynamicsMixin(DynamicsBase):
    def __init__(self,N,T,Nt,trials,Nrecord,a,b,c,x0_min,x0_max,y0_min,y0_max,mu_nx,sg_nx,mu_ny,sg_ny):
        super(LinearDynamicsMixin, self).__init__(N,T,Nt,trials,Nrecord)
        self.a = a
        self.b = b
        self.c = c
        self.x0_min = x0_min
        self.x0_max = x0_max
        self.y0_min = y0_min
        self.y0_max = y0_max
        self.mu_nx = mu_nx
        self.mu_ny = mu_ny
        self.sg_nx = sg_nx
        self.sg_ny = sg_ny
        self.X = []
        self.Y = []
        self._initialize()

    def run_dynamics(self):
        mat = np.squeeze(np.asarray(self.mat.flatten())).tolist()
        for i in range(self.trials):
            x0 = list(self.x0[i])
            y0 = list(self.y0[i])
            this_noise_x = list(self.noise_x[i].flatten())
            this_noise_y = list(self.noise_y[i].flatten())
            params = [self.N,self.Nrecord,self.T,self.Nt,x0,y0,
                      mat,this_noise_x,this_noise_y,self.a,self.b,self.c]
            X,Y = backend.Linear(params)
            self.add_trial(X,Y)

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        return self.X, self.Y
        
    def _initialize(self):
            self.x0 = np.random.randint(self.x0_min,self.x0_max,size=(self.trials,self.N))
            self.y0 = np.random.randint(self.y0_min,self.y0_max,size=(self.trials,self.N))
            self.noise_x = np.sqrt(self.dt)*np.random.normal(self.mu_nx,self.sg_nx,size=(self.trials,self.N,self.Nt))
            self.noise_y = np.sqrt(self.dt)*np.random.normal(self.mu_ny,self.sg_ny,size=(self.trials,self.N,self.Nt))
            self.mat = np.random.normal(0,1,size=(self.N,self.N))
            self.mat = np.multiply(self.mat, self.adj)

    def add_trial(self,X,Y):
        self.X.append(X)
        self.Y.append(Y)

class LinearYeast5(DiGraphDot, LinearDynamicsMixin):
    def __init__(self,T,Nt,trials,a,b,c,x0_min,x0_max,y0_min,y0_max,mu_nx,sg_nx,mu_ny,sg_ny,plot=False,cmap=None):
        path = os.path.dirname(__file__) + '/networks/Yeast_5_net5.dot'
        DiGraphDot.__init__(self, path, plot=plot, cmap=cmap)
        LinearDynamicsMixin.__init__(self,self.N,T,Nt,trials,self.N,a,b,c,x0_min,x0_max,y0_min,y0_max,mu_nx,sg_nx,mu_ny,sg_ny)
        
    def _add_graph_to_axis(self, ax):
        pos = nx.circular_layout(self.graph)
        node_colors = []; edge_colors = []
        for n,node in enumerate(self.graph.nodes(data=True)):
            name, attr = node
            rgba = tuple(attr['color'])
            hex = to_hex(rgba,keep_alpha=True)
            node_colors.append(rgba)
        for n,edge in enumerate(self.graph.edges(data=True)):
            src, dst, attr = edge
            edge_colors.append(attr['color'])
        nx.draw(self.graph,node_color=node_colors,edge_color=edge_colors,pos=pos,with_labels=True, font_size=8, ax=ax)
        
    def _add_dyn_to_axis(self, ax1, ax2, trial_idx=0):
         for n,node in enumerate(self.graph.nodes(data=True)):
             name, attr = node
             rgba = tuple(attr['color'])
             ax1.plot(self.X[trial_idx,:,n],color=rgba)
             ax2.plot(self.Y[trial_idx,:,n],color=rgba)

class LinearYeast10(DiGraphDot, LinearDynamicsMixin):
    def __init__(self,T,Nt,trials,a,b,c,x0_min,x0_max,y0_min,y0_max,mu_nx,sg_nx,mu_ny,sg_ny,plot=False,cmap=None):
        path = os.path.dirname(__file__) + '/networks/Yeast_10_net1.dot'
        DiGraphDot.__init__(self, path, plot=plot, cmap=cmap)
        LinearDynamicsMixin.__init__(self,self.N,T,Nt,trials,self.N,a,b,c,x0_min,x0_max,y0_min,y0_max,mu_nx,sg_nx,mu_ny,sg_ny)
        
    def _add_graph_to_axis(self, ax):
        pos = nx.circular_layout(self.graph)
        node_colors = []; edge_colors = []
        for n,node in enumerate(self.graph.nodes(data=True)):
            name, attr = node
            rgba = tuple(attr['color'])
            hex = to_hex(rgba,keep_alpha=True)
            node_colors.append(rgba)
        for n,edge in enumerate(self.graph.edges(data=True)):
            src, dst, attr = edge
            edge_colors.append(attr['color'])
        nx.draw(self.graph,node_color=node_colors,edge_color=edge_colors,pos=pos,with_labels=True, font_size=8, ax=ax)
        
    def _add_dyn_to_axis(self, ax1, ax2, trial_idx=0):
         for n,node in enumerate(self.graph.nodes(data=True)):
             name, attr = node
             rgba = tuple(attr['color'])
             ax1.plot(self.X[trial_idx,:,n],color=rgba)
             ax2.plot(self.Y[trial_idx,:,n],color=rgba)  
        
    
class LinearYeast25(DiGraphDot, LinearDynamicsMixin):
    def __init__(self,T,Nt,trials,a,b,c,x0_min,x0_max,y0_min,y0_max,mu_nx,sg_nx,mu_ny,sg_ny,plot=False,cmap=None):
        path = os.path.dirname(__file__) + '/networks/Yeast_25_net1.dot'
        DiGraphDot.__init__(self, path, plot=plot, cmap=cmap)
        LinearDynamicsMixin.__init__(self,self.N,T,Nt,trials,self.N,a,b,c,x0_min,x0_max,y0_min,y0_max,mu_nx,sg_nx,mu_ny,sg_ny)
        
    def _add_graph_to_axis(self, ax):
        pos = nx.shell_layout(self.graph)
        node_colors = []; edge_colors = []
        for n,node in enumerate(self.graph.nodes(data=True)):
            name, attr = node
            rgba = tuple(attr['color'])
            hex = to_hex(rgba,keep_alpha=True)
            node_colors.append(rgba)
        for n,edge in enumerate(self.graph.edges(data=True)):
            src, dst, attr = edge
            edge_colors.append(attr['color'])
        nx.draw(self.graph,node_color=node_colors,edge_color=edge_colors,pos=pos,with_labels=True, font_size=8, ax=ax)
        
    def _add_dyn_to_axis(self, ax1, ax2, trial_idx=0):
         for n,node in enumerate(self.graph.nodes(data=True)):
             name, attr = node
             rgba = tuple(attr['color'])
             ax1.plot(self.X[trial_idx,:,n],color=rgba)
             ax2.plot(self.Y[trial_idx,:,n],color=rgba)  
        
 
