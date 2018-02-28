import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
import time
import string

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
#from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.patheffects as PathEffects
import matplotlib

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy



import pandas as pd
df = pd.read_csv('letter-recognition.data.txt',sep=',')

t1 = df.sample(1000)

target = t1['T']
del t1['T']

letter_proj = TSNE(random_state=RS).fit_transform(t1)


fig = plt.figure()
ax = fig.gca(projection='3d')

y1 = np.array([ord(x.lower())-97 for x in target])
# We choose a color palette with seaborn.
palette = np.array(sns.color_palette("hls", 26))

ax.scatter(letter_proj[:,0], letter_proj[:,1], letter_proj[:,2], c=palette[y1])


scatter(letter_proj, target)

def scatter(x, target):

    y1 = np.array([ord(x.lower())-97 for x in target])
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 26))

    # We create a scatter plot.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sc = ax.scatter(x[:,0], x[:,1], x[:,2],
                    c=palette[y1])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each letter.
    # txts = []
    # for i in range(26):
    #     # Position of each label.
    #     xtext, ytext = np.median(x[y1 == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)

    return fig, ax





