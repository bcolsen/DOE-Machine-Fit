#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:08:07 2017

@author: bcolsen
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd 

# fig = plt.gcf()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm

from IPython.lib.pretty import pprint

import pyDOE2  # in anaconda terminal: pip install pyDOE2
import mplcursors # in anaconda terminal: conda install -c conda-forge mplcursors

idx = pd.IndexSlice
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.sans-serif'] = 'Arial'


def make_experiment_array(value_list, factor_labels, reduction=None,
                          shuffle_rows=True, shuffle_values=True,
                          random_seed=42, verbose=True):
    """compute levels and factors"""
    
    # Shuffle the value list
    # pick whatever seed you want,
    # but this lets you get the same random shuffle
    #np.random.seed(random_seed)
    #for values in value_list:
    #    np.random.shuffle(values)
    pprint(value_list)

    levels = [len(y) for y in value_list]
    factors = len(levels)

    if reduction is None:
        # generate maximally reduced DOE table
        break_flag = False
        max_reduction = 100  # maximum reduction level
        for reduction in range(2, max_reduction):
            # make DOE table
            if verbose:
                print('\nReduction level = ', reduction)
            temp_array = pyDOE2.gsd(levels, reduction)
            if verbose:
                print('Number of experiments =', len(temp_array))
            for column in temp_array.T:
                var = np.bincount(column)
                print(var, 'polydispersity = ',
                      np.std(var)/np.mean(var)*100, '%')

                if(np.std(var)/np.mean(var) > 0.15):
                    break_flag = True
                    break

            if break_flag:
                reduction -= 1
                break

    exp_array = pyDOE2.gsd(levels, reduction)
    pprint(exp_array)
    
    # shuffle the DOE table
    np.random.shuffle(exp_array)
    n_exper = len(exp_array)  # actual number of experiments
    if verbose:
        print('Number of experiments for a full factorial design = ',
              np.prod(levels))
        print('Number of experiments for partiral factorial design= ', n_exper)
        print('Reduction factor = ', reduction)

    # print final DOE table with values
    values = []
    for factor in range(factors):
        values += [value_list[factor][exp_array[:, factor]]]
    values = np.array(values).transpose()

    # make dataframe and save to excel
    df = pd.DataFrame(values, columns=factor_labels)
    #df['label'] =  ''#df.apply(lambda _: '', axis=1)
    return df


def anova(data, prop, variables, labels = None):
    if labels is None: 
        labels = variables
    """Run ANOVA analysis"""
    devi_list = []
    for var in variables:
        data_mean = data[prop].mean()
        level_mean = data.groupby(var)[prop].mean()
        devi = ((level_mean - data_mean)**2).sum()
        devi_list += [devi]
    devi_a = np.array(devi_list)
    anova = devi_a/devi_a.sum()*100

    figname = f"{prop} {', '.join(variables)}"
    plt.figure(figname + ' ANOVA', constrained_layout=True, clear=True, figsize = (4,3))
    # plt.title(f'{prop} ANOVA') #Aaron's addition so they'd come with a title, idk they didn't before see line 106
    plt.bar(labels, anova)
    if labels is not None:
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='center')
    plt.xlabel('Variables')
    plt.ylabel(f'Contribution to {prop} (%)')
    return anova


def fit_svm(data, prop, variables, gamma, epsilon=1e-5, C=None, degree=5,
            kernel='rbf', **plot_kw_args):
    """Fit experimental data to a svm function and visualize"""
    # Compute an aproximate weight value C
    if C is None:
        C = 10 * np.abs(data[prop]).max()

    # Make a list of variables for machine learning
    var_data = data[variables]

    # Define a Pipeline that scales the data and applies the model
    reg = Pipeline([('scl', StandardScaler()),
                    ('clf', svm.SVR(kernel=kernel, degree=degree,
                                    gamma=gamma, epsilon=epsilon,
                                    tol=1e-5, C=C, verbose=True))])

    # Fit the variables to the PCE
    reg.fit(var_data, data[prop])

    data[prop + '_pred'] = reg.predict(var_data)
    plot_fit(data, reg.predict, prop, variables, **plot_kw_args)
    return reg

def plot_fit(data, pred, prop, variables, vlim=None, var_prop_labels={},
             plot_prop=False, cmap='viridis', figname= None,
             mark_err = 0.1, fitmap_3d = False):
    vmin, vmax = (data[prop + '_pred'].min(), data[prop + '_pred'].max()) if vlim is None else vlim
    # print(vlim, vmin, vmax)
    # Set the colors
    if data.get('color') is None:
        labels = data.label.unique()
        if(len(labels) <= 10):
            label_cmap = plt.cm.tab10
            num = 10
        elif(len(labels) <= 20):
            label_cmap = plt.cm.tab20
            num = 20
        else:
            label_cmap = plt.cm.rainbow
            num = len(labels)

        colors = label_cmap(np.linspace(0, 1, num)[0:len(labels)])
        color_dic = {label: color for label, color in zip(labels, colors)}
    else:
        color_dic = data.groupby('label').last().color

    prop_label = var_prop_labels.get(prop, prop)
    var_labels = {key:var_prop_labels.get(key, key) for key in variables}

    figname = f"{prop} {', '.join(variables)}" if figname is None else figname

    # ###########################
    # Make the plot fit plot
    plt.figure('Fit prediction ' + figname, constrained_layout=True,
               clear=True)
    for label, label_df in data.groupby('label'):
        plt.plot(prop, prop + '_pred', 'o', data=label_df,
                 color=color_dic[label], label=label, zorder=2)
    plt.legend()
    plt.xlabel(f'Measured {prop}',fontsize=14)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylabel(f'Predicted {prop}',fontsize=14)
    # plt.plot(filter_df.pce, filter_df.pce_pred_ridge, 'o')
    plt.autoscale(enable=False)
    for ax in plt.gcf().get_axes():
        mplcursors.cursor(ax.get_lines())
    plt.plot([-2, 100], [-2, 100], ls="--", c=".3", zorder=1)

    # ##############################################
    # Make the model gradiant plot
    var_n = len(variables)

    if var_n == 4:
        ui, vi, xi, yi = variables
    elif var_n == 3:
        vi, xi, yi = variables
        ui = None
    elif var_n == 2:
        xi, yi = variables
        ui, vi = None, None
    else:
        print('length of variables must be 2,3 or 4')
        return

    us = [None] if ui is None else data[ui].sort_values().unique()
    vs = [None] if vi is None else data[vi].sort_values().unique()

    x_len, y_len = 100, 100

    xmargin = (data[xi].max() - data[xi].min()) * 0.05
    ymargin = (data[yi].max() - data[yi].min()) * 0.05

    xl = np.linspace(data[xi].min()-xmargin, data[xi].max()+xmargin, x_len)
    yl = np.linspace(data[yi].min()-ymargin, data[yi].max()+ymargin, y_len)

    um, vm, xm, ym = np.meshgrid(us, vs, xl, yl)
    r = pd.DataFrame()
    r[ui] = um.ravel()
    r[vi] = vm.ravel()
    r[xi] = xm.ravel()
    r[yi] = ym.ravel()

    figsize = (len(vs)*2 + 1.5, len(us)*2 + 0.7)
    fig, axs = plt.subplots(nrows=len(us), ncols=len(vs), squeeze=False,
                            sharex=True, sharey=True, clear=True,
                            num='Fit map ' + figname, figsize=figsize,
                            constrained_layout=True)
    title_bool=True
    for axx, u in zip(axs, us[::-1]):
        for ax, v in zip(axx, vs):
            if u is None and v is None:
                rf = r
                dfuv = data
                xlabel = f'{var_labels[xi]}'
                ylabel = f'{var_labels[yi]}'
            else:
                if u is None:
                    qstr = f'{vi} == {v}'
                    ylabel = f'{var_labels[yi]}'
                    xlabel = f'{var_labels[xi]}'
                    ax.set_title(f'{v:.4g} {var_labels[vi]}', fontsize=10)
                else:
                    qstr = f'{ui} == {u} and {vi} == {v}'
                    ylabel = f'{u:.4g} {var_labels[ui]}\n{var_labels[yi]}'
                    if title_bool:
                        ax.set_title(f'{v:.4g} {var_labels[vi]}', fontsize=10)
                    xlabel = f'{var_labels[xi]}'
                rf = r.query(qstr)
                dfuv = data.query(qstr)
            if u == us[0]:
                ax.set_xlabel(xlabel)
            if v == vs[0]:
                ax.set_ylabel(ylabel)
            values = pred(rf[variables]).reshape(xm.shape[-2:])
            pmap = ax.contour(xm[0, 0], ym[0, 0], values, vmin=vmin,
                              vmax=vmax, cmap='gray_r')
            plt.clabel(pmap, inline=1, fontsize=10)
            pmap = ax.pcolormesh(xm[0, 0], ym[0, 0], values, shading='gouraud',
                                 vmin=vmin, vmax=vmax, cmap=cmap)
                            
            if plot_prop:
                ax.scatter(xi, yi, c=prop, data=dfuv, vmin=vmin, vmax=vmax,
                            edgecolors='k', linewidths=1, cmap=cmap)
            else:
                for label, label_df in dfuv.groupby('label'):
                    pred_error = (np.abs(label_df[prop] - label_df[prop + '_pred'])/label_df[prop]).iloc[0]
                    if pred_error > mark_err:
                        mec='w'
                    else:
                        mec='k'
                    ax.plot(xi, yi, 'o', data=label_df.iloc[0],
                            label=label, color=color_dic[label],
                            mec=mec, mew=1)
        title_bool=False
        
    cbar = plt.colorbar(pmap, ax=axs, aspect=40, pad=0.01)
    cbar.set_label(prop_label)
    
    for ax in plt.gcf().get_axes():
        mplcursors.cursor(ax.get_lines())
    
    if fitmap_3d:
        fig, ax = plt.subplots(clear=True, num='Fit map 3D ' + figname, figsize=figsize,
                            constrained_layout=True, subplot_kw={"projection": "3d"})
        
        ax.scatter(xi, yi, prop, c=prop, data=dfuv, vmin=vmin, vmax=vmax,
                    edgecolors='0.35', linewidths=1, cmap=cmap, depthshade=False)
        surf = ax.plot_surface(xm[0, 0], ym[0, 0], values, vmin=vmin, vmax=vmax,
                               cmap=cmap, linewidth=0, antialiased=False, rstride=1, cstride=1, shade=True)
        
        cbar = fig.colorbar(surf, shrink=0.7, aspect=20)
        cbar.set_label(prop_label)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(prop_label)

        
        # pmap = ax.pcolormesh(xm[0, 0], ym[0, 0], values, shading='gouraud',
        #                      vmin=vmin, vmax=vmax, cmap=cmap)

if __name__ == "__main__":
    df = pd.read_csv("acsnano.csv")
    var_prop_labels={'pce': 'Power Conversion Efficiency (%)',
            'don_con': 'wt% Donor Concentration',
            'spin_s': 'Spin Speed (rpm)',
            'total_con': 'Total Concentration (mg/ml)',
            }
    fit_svm(df, 'pce', ['don_con', 'spin_s', 'total_con'],gamma=0.15,
            mark_err=0.1, var_prop_labels=var_prop_labels)
    plt.gcf().savefig("acsnano_figure.png", dpi=300)