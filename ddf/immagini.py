import numpy as np

from matplotlib import cm, colors, ticker
import matplotlib.pyplot as plt
import matplotlib
plt.rc('text', usetex=True)
preamble = r'''
\usepackage{siunitx}
\usepackage{amssymb}
\usepackage{amsmath}
'''
# https://matplotlib.org/stable/users/explain/customizing.html
plt.rc('text.latex', preamble=preamble)
plt.rc('figure', dpi=600)
plt.rc('axes', titlesize='medium')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
plt.rc('scatter', edgecolors='black')

# https://tex.stackexchange.com/questions/24599/what-point-pt-font-size-are-large-etc
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# t = ax.text(0.5, 0.5, 'Text')
# fonts = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large', 'larger', 'smaller']
# for font in fonts:
#     t.set_fontsize(font)
#     print (font, round(t.get_fontsize(), 2))
# plt.close()    
# print([ (k,v) for k,v in matplotlib.rcParams.items() if 'size' in k ])

sp_kw = dict(s=16, edgecolors='black', linewidth=0.5) # scatter
sf_kw = dict(transparent=1, bbox_inches='tight', pad_inches=0.01) # savefig
linewidth = 345/72 # (paper) 72 = pt/inch
# linewidth = 483/72 # (tesi) 72 = pt/inch
from PIL import Image

blupoli = np.r_[114, 143, 165]/255
blu = np.r_[88, 147, 191]/255
rosso = np.r_[191, 88, 88]/255

# https://stackoverflow.com/questions/72287305/matplotlib-convert-colormap-to-pastel-colors
n = 12; c = 0.23
colori = (1. - c) * plt.get_cmap('turbo')(np.linspace(0., 1., n)) + c * np.ones((n, 4))
colori = colori[1:-1]
cmap_w = colors.ListedColormap(colori)
pc_w = dict(cmap=cmap_w, levels=np.linspace(0, 1, 11))

n = 12; c = 0.33
colori = (1. - c) * plt.get_cmap('turbo')(np.linspace(0., 1., n)) + c * np.ones((n, 4))
colori = colori[1:-1]
cmap_ww = colors.ListedColormap(colori)

n = 10; c = 0.2
colori = (1. - c) * plt.get_cmap('Spectral_r')(np.linspace(0., 1., n)) + c * np.ones((n, 4))
# colori = colori[1:-1]
cmap_dw = colors.ListedColormap(colori)
pc_dw = dict(cmap=cmap_dw, levels=10)

n = 11; c = 0.25
colori = (1. - c) * plt.get_cmap('viridis')(np.linspace(0., 1., n)) + c * np.ones((n, 4))
colori = colori[1:]
cmap_p = colors.ListedColormap(colori)
pc_p = dict(cmap=cmap_p, levels=10)

n = 10; c = 0.2
colori_m = (1 - c) * np.array([ colors.to_rgb(f'C{i}') for i in range(10) ]) + c * np.ones((n, 3))
