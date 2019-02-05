# 4/29/2018
from __future__ import division
import calendar
import csv
from collections import Counter
import gensim
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import pandas as pd
import platform
import os
import random
import re, ast
import scipy
import sklearn
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation, datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import sys
import time
import xlsxwriter
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import imblearn
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
