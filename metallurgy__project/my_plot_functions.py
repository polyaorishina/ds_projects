#!/usr/bin/env python
# coding: utf-8

# <h1>Оглавление<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

import pandas as pd
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial.polynomial import polyfit

    
def plot_distribution(data_list, xlabels, title=None, kind='hist', bins=None, rotate_text=False, figsize=(15,6)):
    if kind == 'hist':
        fig, axes = plt.subplots(
            nrows=2, ncols=len(data_list), figsize=figsize, gridspec_kw={'height_ratios': [1, 4]}
        )
    else: fig, axes = plt.subplots(nrows=1, ncols=len(data_list), figsize=figsize)    
    plt.subplots_adjust(wspace=0.3, hspace=0)
    if title:
        plt.suptitle(title, fontsize=16)
    
    for i in range(len(data_list)):             
        if kind == 'hist':
            axes[0][i].set_facecolor('whitesmoke')
            axes[1][i].set_facecolor('whitesmoke')
               
            bottom_side = axes[0][i].spines['bottom']
            top_side = axes[1][i].spines['top']
            bottom_side.set_visible(False)
            top_side.set_visible(False)
            
            data_list[i].plot.hist(ax=axes[1][i], bins=bins[i], color='tomato', edgecolor='indianred', linewidth=2)
            axes[0][i].boxplot(
                data_list[i].dropna(), vert=False, patch_artist=True, notch=False, widths=0.3,
                boxprops=dict(facecolor='salmon', color='whitesmoke', lw=2),
                medianprops=dict(color='whitesmoke', lw=2),
                flierprops=dict(marker='o', markersize=10, markerfacecolor='whitesmoke', markeredgecolor='tomato'),
                whiskerprops=dict(color='salmon', lw=2),
                capprops=dict(lw=0)
            )
            axes[1][i].set_xlabel(xlabels[i], fontsize=14)
            axes[1][i].set_ylabel('Число объектов', fontsize=12)
            axes[0][i].set_xticks([])
            axes[0][i].set_yticks([])
            axes[1][i].grid(color='w', lw=1, axis='y')
            axes[1][i].set_axisbelow(True) 
            axes[1][i].yaxis.set_ticks_position('none') 
            
        elif kind == 'bar':
            axes[i].set_facecolor('whitesmoke')
            values = data_list[i].value_counts().sort_index()
            values.plot.bar(
                ax=axes[i], color='tomato', edgecolor='tomato'
            )
            k = 0
            if rotate_text:
                angle = 90
                axes[i].set_ylim(0, values.max() * 1.4)
                
            else:
                angle=0
                axes[i].set_ylim(0, values.max() * 1.2)
            for v in values:
                axes[i].text(k, v + 0.1 * values.max(), str(v), ha='center', 
                             fontsize=12,color='dimgrey', rotation=angle)
                k += 1
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0)
            
            axes[i].grid(color='w', lw=1, axis='y')
            axes[i].set_axisbelow(True)
            axes[i].xaxis.set_ticks_position('none')
            axes[i].yaxis.set_ticks_position('none')
            axes[i].set_xlabel(xlabels[i], fontsize=14)
            axes[i].set_ylabel('Число объектов', fontsize=12)
    plt.show()
    

def plot_matrix(data, title='corr', fmt='.1f', figsize=(15,10)):
    size = len(data)
    mask = np.triu(np.ones((size, size)), k=1)    
    fig, axes = plt.subplots(figsize=figsize)
    if title == 'corr':
        axes = sns.heatmap(data, annot=True, fmt=fmt, mask=mask, square=False, vmin=-1., vmax=1.)
        axes.set_title('Корреляционная матрица', fontsize=16, pad=20)
    else:
        axes = sns.heatmap(data, annot=True, fmt='.1f', mask=mask, square=False)
        axes.set_title(title, fontsize=16, pad=20)
    axes.set_yticklabels(axes.get_yticklabels(), rotation=0)
    b, t = plt.ylim() 
    plt.ylim(b + 0.5, t - 0.5)
    plt.show()
    
    
def plot_scatter(data_list, xlabels, ylabels, title=None, pval=False, alphas=[0.1] * 6, figsize=(15,6)):
    fig, axes = plt.subplots(nrows=1, ncols=len(data_list), figsize=figsize)
    plt.subplots_adjust(wspace=0.4)
    if title:
        plt.suptitle(title, fontsize=16)
    for i in range(len(data_list)):             
        axes[i].set_facecolor('whitesmoke')
        x = np.array(data_list[i][0])
        y = np.array(data_list[i][1])
        intercept, slope = polyfit(x, y, 1)
        right, left, top, bottom = max(x), min(x), max(y), min(y)
        axes[i].scatter(x, y, alpha=alphas[i], s=70, label='Выборка', marker="X", facecolor='tomato', 
                        edgecolors='tomato')
        axes[i].plot((left - (right-left)/20, right + (right-left)/20), 
                     (intercept + slope*(left - (right-left)/20), intercept + slope*(right + (right-left)/20)), 
                     '-', label='Аппроксимация', color='firebrick', lw=4)
        if pval:
            r, p = st.pearsonr(x, y)
            p_format = '{:.2f}' * (p >= 0.01) + '{:.0e}' * (p < 0.01)
            axes[i].text(0.95, 0.05, ('r = {:.2f}\np-value = ' + p_format).format(r, p), ha='right', 
                         fontsize=12, color='dimgrey', transform=axes[i].transAxes)
        axes[i].set_xlabel(xlabels[i], fontsize=14)
        axes[i].set_ylabel(ylabels[i], fontsize=14)   
        axes[i].legend(facecolor='ghostwhite', loc='upper right', ncol=1, shadow=True, fancybox=False)
        axes[i].set_xlim(left - (right-left)/20, right + (right-left)/20)
        axes[i].set_ylim(bottom - (top-bottom)/20, top + (top-bottom)/20)
        
        axes[i].grid(color='w', lw=1, axis='both')
        axes[i].set_axisbelow(True)
        axes[i].xaxis.set_ticks_position('none')
        axes[i].yaxis.set_ticks_position('none')
    plt.show()
    
    
def plot_box(data_list, labels, ticks, title, figsize=(15,6)):    
    fig, axes = plt.subplots(figsize=figsize)    
    plt.subplots_adjust(wspace=0.3, hspace=0)
    plt.suptitle(title, fontsize=16)
    axes.set_facecolor('whitesmoke')
    axes.boxplot(
        data_list, vert=False, patch_artist=True, notch=False,
        boxprops=dict(facecolor='tomato', color='whitesmoke', lw=2),
        medianprops=dict(color='whitesmoke', lw=2),
        flierprops=dict(marker='o', markersize=15, markerfacecolor='whitesmoke', markeredgecolor='tomato'),
        whiskerprops=dict(color='orangered', lw=2),
        capprops=dict(lw=0)
    )        
    axes.set_xlabel(labels[0], fontsize=14)
    axes.set_ylabel(labels[1], fontsize=14)
    axes.grid(color='w', lw=1, axis='x')
    axes.set_yticklabels(ticks, fontsize=12)
    axes.xaxis.set_ticks_position('none')
    axes.yaxis.set_ticks_position('none')
    plt.show()


def plot_true_pred(true, pred, q=None, step=None, label='Модель', plot_sample=True, ylim=None, figsize=(8,6)):
    
    fig, axes = plt.subplots(figsize=figsize)
    axes.set_facecolor('whitesmoke')
    
    target_df = pd.DataFrame({'true': true, 'pred': pred})
    if step:
        target_df['true'] = pd.cut(target_df['true'], bins=np.arange(0,target_df['true'].max()+step, step))\
                            .apply(lambda interval: (interval.right + interval.left) / 2)
        axes.set_title('Предсказания модели на объектах с разными значениями таргета (step={})'.format(step),
                       fontsize=16, pad=20)
    else:
        target_df['true'] = pd.qcut(target_df['true'], q=q, duplicates='drop')\
                            .apply(lambda interval: interval.right) 
        axes.set_title('Предсказания модели на объектах с разными значениями таргета (q={})'.format(q), 
                       fontsize=16, pad=20)
    mean = target_df.groupby('true')['pred'].mean()
    error = target_df.groupby('true')['pred'].std()
    
    axes.plot((true.min(), true.max()), (true.min(), true.max()), label='Идеальная модель', 
              color='darkturquoise', lw=5)
    if plot_sample:
        axes.scatter(true, pred, marker='o', s=70, facecolor='whitesmoke', edgecolor='coral', label=label)
        axes.plot(np.array(mean.index), mean.values, '-', color='orangered', linewidth=5, label='$\mu$')
        axes.fill_between(np.array(mean.index), (mean - error).values, (mean + error).values,
                          color='tomato', label='Интервал $\mu\pm\sigma$', alpha=0.3, lw=0)
    else:
        axes.plot(np.array(mean.index), mean.values, '-', color='firebrick', linewidth=5, label=label)
        axes.fill_between(np.array(mean.index), (mean - error).values, (mean + error).values,
                          color='tomato', label='Интервал $\mu\pm\sigma$')
    
    axes.set_xlabel('Температура, $^{\circ}C$', fontsize=14)
    axes.set_ylabel('Предсказание модели, $^{\circ}$C', fontsize=14)
    axes.legend(
        *([x[i] for i in [0,2,1,3]] for x in plt.gca().get_legend_handles_labels()),
        facecolor='ghostwhite', fontsize=12, shadow=True, fancybox=False, loc='upper left')
    axes.grid(color='w', lw=1, axis='both')
    axes.set_axisbelow(True)
    axes.set_xlim(error[~error.isna()].index[0], error[~error.isna()].index[-1])
    axes.xaxis.set_ticks_position('none')
    axes.yaxis.set_ticks_position('none')
    if ylim:
        axes.set_ylim(ylim)
    plt.show()