# TO_TRY
from tqdm import tqdm
for e in tqdm(range(nb_epoch))

pandas_profiling

%config InlineBackend.figure_format = 'retina' # enable hi-res output

# Basic libraries import
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys

import itertools
import collections

# Plotting
%matplotlib notebook
%matplotlib inline

sns.set_context("paper")
sns.set_context("notebook", font_scale=1.5)
sns.set_style("dark")

color = sns.color_palette()
ax.xaxis_date()
plt.xticks(rotation='vertical')

import plotly
import cufflinks as cf
cf.go_offline(connected=True)
plotly.offline.init_notebook_mode(connected=True)

from ipywidgets import interact, widgets
from IPython.display import display

## Animation
from matplotlib import pyplot as plt, animation, rc
rc('animation', html='html5')
from IPython.display import HTML
HTML(ani.to_html5_video())

# Reload modules
%load_ext autoreload
%autoreload 2

%load_ext autoreload
%aimport foo, bar
%autoreload 1

import importlib
import foo; importlib.reload(foo)
from foo import bar

import spam
import imp
imp.reload(spam)

# Add path to system path
import sys
import os
from os.path import abspath, join, dirname

sys.path.append(join(os.getcwd(), *[os.pardir]*3, 'data'))
sys.path.append(join(dirname(__file__))

sys.path.insert(0, abspath(dirname(__file__), 'src'))

import pkgutil
data = pkgutil.get_data(__package__, 'somedata.dat')

# Logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers[0].stream = sys.stdout

# Debug
import pdb
pdb.set_trace()

# [Kaggle-CLI](https://github.com/floydwch/kaggle-cli)

"kg.exe" submit submission.csv -u -p  -c challenge-name -m "message"

# Jupyter Themes
jt -t grade3 -T -nfs 9 -cellw 90%

# Video Conversion
ffmpeg -i in.WMV -filter:v "setpts=0.7*PTS" -c:v libx264 -crf 23 -c:a libfaac -q:a 100 -ss 00:00:35 out.mp4

ffmpeg -i in.WMV -filter:v "setpts=0.6*PTS" -ss 00:00:10 out.gif
