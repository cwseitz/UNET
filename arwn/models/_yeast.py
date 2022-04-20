from ._digraph import *
from arwn import backend
from matplotlib.colors import to_hex
import numpy as np
import networkx as nx

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
    def __init__(self,N,T,Nt,trials,Nrecord):
        super(LinearDynamicsMixin, self).__init__(N,T,Nt,trials,Nrecord)
        self.X = self.Y = []
        self._initialize()

    def run_dynamics(self):
        mat = np.squeeze(np.asarray(self.mat.flatten())).tolist()
        for i in range(self.trials):
            x0 = list(self.x0[i])
            y0 = list(self.y0[i])
            this_noise_x = list(self.noise_x[i].flatten())
            this_noise_y = list(self.noise_y[i].flatten())
            params = [self.N,self.Nrecord,self.T,self.Nt,x0,y0,mat,this_noise_x,this_noise_y]
            X,Y = backend.Linear(params)
            self.add_trial(X,Y)

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        return self.X, self.Y
        
    def _initialize(self):
            self.x0 = np.random.randint(10,100,size=(self.trials,self.N))
            self.y0 = np.random.randint(10,100,size=(self.trials,self.N))
            self.noise_x = np.sqrt(self.dt)*np.random.normal(0,1,size=(self.trials,self.N,self.Nt))
            self.noise_y = np.sqrt(self.dt)*np.random.normal(0,1,size=(self.trials,self.N,self.Nt))
            self.mat = np.random.normal(0,1,size=(self.N,self.N))*self.adj

    def add_trial(self, X, Y):
        self.X.append(X)
        self.Y.append(Y)

class LinearYeastExample(DiGraphDot, LinearDynamicsMixin):
    def __init__(self, T, Nt, trials, plot=False, cmap=None):
        path = os.path.dirname(__file__) + '/networks/yeast.dot'
        DiGraphDot.__init__(self, path, plot=plot, cmap=cmap)
        LinearDynamicsMixin.__init__(self,self.N,T,Nt,trials,self.N)
        
    def _add_graph_to_axis(self, ax):
        pos = nx.circular_layout(self.graph)
        node_colors = []
        for n,node in enumerate(self.graph.nodes(data=True)):
            name, attr = node
            rgba = tuple(attr['color'])
            hex = to_hex(rgba,keep_alpha=True)
            node_colors.append(rgba)
        nx.draw(self.graph,node_color=node_colors,pos=pos,with_labels=True, font_size=8, ax=ax)
        
    def _add_dyn_to_axis(self, ax1, ax2, trial_idx=0):
         for n,node in enumerate(self.graph.nodes(data=True)):
             name, attr = node
             rgba = tuple(attr['color'])
             ax1.plot(self.X[trial_idx,:,n],color=rgba)
             ax2.plot(self.Y[trial_idx,:,n],color=rgba)
              
        
 
