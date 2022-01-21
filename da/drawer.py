import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

oc = [93.4,94,92.1, 89.2,87.1,  54.8, 57.9, 92.9,95.1,  95.5,95.1]
p_oc = [0.7,0.7,0.7, 0.8,0.8,  0.9,0.9,  0.6,0.6,  0.5,0.5]

vada = [69.8,63.4,62.9,  46.0,42.3,  35.4, 34.9,  74.5,75.4,  92.1,93.2]
p_vada = [0.7,0.7,0.7, 0.8,0.8,  0.9,0.9,  0.6,0.6,  0.5,0.5]

vanilla = [52.1, 51.5,   49.2,50.0,  43.0,44.7,  56.2,51.9,   56.3,55.5]
p_vanilla = [0.7,0.7, 0.8,0.8,  0.9,0.9,  0.6,0.6,  0.5,0.5]


sns.set(style='whitegrid', font_scale=1.4,rc={"lines.linewidth": 3.5})

d_2 = {'Test accuracy (%)': vada , 'p': p_vada}
pdnumsqr_2 = pd.DataFrame(d_2)
sns.lineplot(x='p', y='Test accuracy (%)', data=pdnumsqr_2,label='VADA')

d_2 = {'Test accuracy (%)': oc, 'p': p_oc}
pdnumsqr_2 = pd.DataFrame(d_2)
sns.lineplot(x='p', y='Test accuracy (%)', data=pdnumsqr_2,label='VADA+$w_x$\ $w_1$')

d_2 = {'Test accuracy (%)': vanilla, 'p': p_vanilla}
pdnumsqr_2 = pd.DataFrame(d_2)
sns.lineplot(x='p', y='Test accuracy (%)', data=pdnumsqr_2,label='Vanilla')

plt.title('MNIST to MNISTM')
plt.legend()
plt.savefig('da_p0.pdf', dpi=300, bbox_inches='tight')
plt.show()

