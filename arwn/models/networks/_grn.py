import numpy as np
import sentinel_core

################################################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@iu.edu
################################################################################

class GRN_Base:

    def __init__(self, N, T, dt, trials, Nrecord):

        """
        GRN base class

        Parameters
        ----------
        """

        #Basic parameters common to all GRNs
        self.N = N
        self.T = T #simulation period
        self.dt = dt #time resolution
        self.trials = trials #number of trials
        self.Nt = int(round((self.T/dt)))
        self.Nrecord = Nrecord
        self.Irecord = list(np.random.randint(0,self.Nrecord,size=(Nrecord,)))
        self.Irecord = [x.item() for x in self.Irecord] #convert to native type
        self.shape = (self.N,self.trials,self.Nt)

class LinearGRN(GRN_Base):

    def __init__(self,N,trials,Nrecord,T,Nt):

        """
        GRN object that invokes core functions in C

        Parameters
        ----------
        See comments below
        """

        dt = T/Nt
        super(LinearGRN, self).__init__(N, T, dt, trials, Nrecord)
        self.X = [] #concentration

    def call(self,x0,W,mat):

        #Construct parameter list for call to C backend
        x0 = list(x0)
        mat = list(mat.flatten())

        for i in range(self.trials):
            this_W = list(W[i].flatten())
            params = [self.N,self.Nrecord,self.T,self.Nt,x0,this_W,mat]
            ctup = sentinel_core.lin_grn(params)
            self.add_trial(ctup)

        self.X = np.array(self.X)

    def add_trial(self, tup):

        X = tup
        self.X.append(X)

class HillGRN(GRN_Base):

    def __init__(self,N,trials,Nrecord,T,Nt):

        """
        GRN object that invokes core functions in C

        Parameters
        ----------
        See comments below
        """

        dt = T/Nt
        super(HillGRN, self).__init__(N, T, dt, trials, Nrecord)
        self.X = [] #concentration

    def call(self,x0,w1,w2,h,K,b,lam,q,n):

        #Construct parameter list for call to C backend
        x0 = list(x0)
        h = list(h.flatten())
        K = list(K.flatten())
        b = list(b)
        lam = list(lam)
        q = list(q)
        n = list(n.flatten())

        for i in range(self.trials):
            this_w1 = list(w1[i].flatten())
            this_w2 = list(w2[i].flatten())
            params = [self.N,self.Nrecord,self.T,self.Nt,x0,this_w1,this_w2,h,K,b,lam,q,n]
            ctup = sentinel_core.lmc_grn(params)
            self.add_trial(ctup)

        self.X = np.array(self.X)

    def add_trial(self, tup):

        X = tup
        self.X.append(X)
