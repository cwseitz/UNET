import numpy as np
from copy import deepcopy

class OrnsteinUhlenbeck:

    def __init__(self, T, dt, tau, sigma, dx=0.1, x_max=1, x0=0, trials=1000, dtype=np.float32):

        """

        Integrate a Langevin equation for constant drift and
        diffusion. Diffusion is a stationary Gaussian white noise

        Parameters
        ----------
        """

        #Params
        self.nsteps = int(round(T/dt))
        self.dt = dt
        self.dx = dx
        self.x0 = x0
        self.x_max = x_max
        self.nx = int(round(2*self.x_max/self.dx))
        self.alpha = 1/tau
        self.sigma = sigma
        self.trials = trials
        self._x = np.linspace(-self.x_max, self.x_max, self.nx)
        self.mu = np.zeros((self.nsteps,))
        self.var = np.zeros((self.nsteps,))

        #Arrays for simulation history
        self.X = np.zeros((self.nsteps, self.trials))
        self.p1 = np.zeros((self.nx, self.nsteps))
        self.p2 = deepcopy(self.p1)

    def solve(self):

        for n in range(self.nsteps):
            var = (self.sigma**2/(2*self.alpha))*(1-np.exp(-2*self.alpha*n*self.dt))
            mu = self.x0*np.exp(-self.alpha*n*self.dt)
            self.mu[n] = mu
            self.var[n] = var
            p0 = np.sqrt(1/(2*np.pi*var))*np.exp(-((self._x-mu)**2)/(2*var))
            self.p2[:,n] = p0
        return self.p2

    def histogram(self):

        for i in range(self.nsteps):
            vals, bins = np.histogram(self.X[i,:], bins=self.nx, range=(-self.x_max,self.x_max), density=True)
            self.p1[:,i] = vals

    def forward(self):

        self.X[0,:] = self.x0
        noise = np.random.normal(loc=0.0,scale=1.0,size=(self.nsteps,self.trials))*np.sqrt(self.dt) #define noise process
        for i in range(1,self.nsteps):
            for j in range(self.trials):
                self.X[i,j] = self.X[i-1,j] - self.dt*self.alpha*(self.X[i-1,j]) + self.sigma*noise[i,j]

class OrnsteinUhlenbeck2D:

    def __init__(self, T, dt, tau, sigma, dx=0.1, x_max=1, x0=0, trials=1000, dtype=np.float32):

        """

        Integrate a Langevin equation for constant drift and
        diffusion. Diffusion is a stationary Gaussian white noise

        Parameters
        ----------
        """

        #Params
        self.nsteps = int(round(T/dt))
        self.dt = dt
        self.dx = dx
        self.x0 = x0
        self.x_max = x_max
        self.nx = int(round(2*self.x_max/self.dx))
        self.alpha = 1/tau
        self.sigma = sigma
        self.trials = trials
        self._x = np.linspace(-self.x_max, self.x_max, self.nx)
        self.mu = np.zeros((self.nsteps,))
        self.var = np.zeros((self.nsteps,))

        #Arrays for simulation history
        self.X = np.zeros((self.nsteps, self.trials))
        self.p1 = np.zeros((self.nx, self.nsteps))
        self.p2 = deepcopy(self.p1)

    def solve(self):

        for n in range(self.nsteps):
            var = (self.sigma**2/(2*self.alpha))*(1-np.exp(-2*self.alpha*n*self.dt))
            mu = self.x0*np.exp(-self.alpha*n*self.dt)
            self.mu[n] = mu
            self.var[n] = var
            p0 = np.sqrt(1/(2*np.pi*var))*np.exp(-((self._x-mu)**2)/(2*var))
            self.p2[:,n] = p0
        return self.p2

    def histogram(self):

        for i in range(self.nsteps):
            vals, bins = np.histogram(self.X[i,:], bins=self.nx, range=(-self.x_max,self.x_max), density=True)
            self.p1[:,i] = vals

    def forward(self):

        self.X[0,:] = self.x0
        noise = np.random.normal(loc=0.0,scale=1.0,size=(self.nsteps,self.trials))*np.sqrt(self.dt) #define noise process
        for i in range(1,self.nsteps):
            for j in range(self.trials):
                self.X[i,j] = self.X[i-1,j] - self.dt*self.alpha*(self.X[i-1,j]) + self.sigma*noise[i,j]
