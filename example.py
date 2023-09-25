#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:40:42 2023

@author: bcolsen
"""

import pandas as pd
import matplotlib.pyplot as plt
import doe_machine_fit as doe

df = pd.read_csv("acsnano.csv")
var_prop_labels={'pce': 'Power Conversion Efficiency (%)',
        'don_con': 'wt% Donor Concentration',
        'spin_s': 'Spin Speed (rpm)',
        'total_con': 'Total Concentration (mg/ml)',
        }
doe.fit_svm(df, 'pce', ['don_con', 'spin_s', 'total_con'],gamma=0.15,
        mark_err=0.1, var_prop_labels=var_prop_labels)
plt.gcf().savefig("example_figure.png", dpi=300)