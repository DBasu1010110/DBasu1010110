"""
Canada Weather Data Visualization
____________________________________________________________________________________________
Data obtained from: https://www.kaggle.com/datasets/hemil26/canada-weather?resource=download

This script performs advanced data cleaning and visualization on Canadian cities' weather data.
This is self-initiated for practice using Pythons visualization tools.

Author: Debopriya Basu
"""

# --- IMPORT LIBRARIES ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

