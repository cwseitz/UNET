from sentinel.models import *
import matplotlib.pyplot as plt
from matplotlib import cm

T = 1000
dt = 10
tau = 100
sigma = 0.1

ou = OrnsteinUhlenbeck(T,dt,tau,sigma,x_max=5,dx=0.2,x0=5)
ou.forward()
ou.solve()
ou.histogram()

coolwarm = cm.get_cmap('coolwarm')
colors = coolwarm(np.linspace(0, 1, ou.nsteps))

fig, ax = plt.subplots(1,3,figsize=(10,3))
ax[0].plot(ou.mu,color='red',label=r'$\mu$')
ax[0].plot(np.sqrt(ou.var),color='blue',label=r'$\sigma$')
ax[0].set_xlabel('Time')
ax[0].legend()
ax[1].set_xlabel('x')
ax[2].set_xlabel('x')
ax[1].set_ylabel('PDF')
ax[2].set_ylabel('PDF')
for i in range(1,ou.nsteps):
    ax[1].plot(ou._x,ou.p2[:,i],color=colors[i])
    ax[2].plot(ou._x,ou.p1[:,i],color=colors[i])
plt.tight_layout()
plt.show()
