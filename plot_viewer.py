# data wrangling
import numpy as np
import pandas as pd

# plots
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection



df = pd.read_csv("skeleton_data/keep_walk.csv")

pressures = df.iloc[:,:44].values


# create the figure
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={'width_ratios': [.9, .1]})
fig.patch.set_alpha(1)

# make right plot invisible and only update left one
axes[1].axis('off')
ax = axes[0]

# get the cmap to use
cmap = get_cmap('RdYlGn')

current_slice = indexes_rolling.values[:261, :]
index_names = indexes_rolling.columns
index_dates = indexes_rolling.index
