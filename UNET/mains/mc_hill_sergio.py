import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from arwn.dynamics import sergio
from arwn import get_root
from arwn.utils import format_ax
root = get_root()

#sim = sergio(number_genes=100, number_bins=9, number_sc=300, noise_params=1, decays=0.8, sampling_state=15, noise_type='dpd')
#target_path = root + '/datasets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt'
#regs_path = root + '/datasets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt'
#sim.build_graph(input_file_taregts=target_path, input_file_regs=regs_path, shared_coop_state=2)


#sim.simulate()
#expr = sim.getExpressions()
#expr_clean = np.concatenate(expr, axis=1)
#np.savez('experiment', expr=expr, expr_clean=expr_clean)

exp = np.load('experiment.npz')
expr = exp['expr']
expr_clean = exp['expr_clean']

fig, ax = plt.subplots(10,1,sharex=True,sharey=True)
map = cm.get_cmap('coolwarm')
colors = map(np.linspace(0,1,10))
for i in range(10):
	ax[i].plot(expr[0,i,:],color=colors[i])
	format_ax(ax[i],ax_is_box=False,xlabel='Sample index (time step)')
	ax[i].set_yticks([])
plt.title('RNA Concentration',rotation='vertical',x=-0.05,y=2)
plt.tight_layout()
plt.show()
