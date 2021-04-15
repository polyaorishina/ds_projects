#!/usr/bin/env python
# coding: utf-8

# <h1>Оглавление<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial.polynomial import polyfit

def plot_distribution(data_list, xlabels, title=None, bins=None, figsize=(15,6)):
    fig, axes = plt.subplots(
        nrows=2, ncols=len(data_list), figsize=figsize, gridspec_kw={'height_ratios': [1, 4]}
    )  
    plt.subplots_adjust(wspace=0.3, hspace=0)
    if title:
        plt.suptitle(title, fontsize=16)
    
    for i in range(len(data_list)):             
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
    plt.show()


def plot_lines(x, y, xlabel, ylabel, title=None, xlim=None, ylim=None, labels=None, figsize=(15,4)):
    fig, axes = plt.subplots(figsize=figsize)
    axes.set_facecolor('whitesmoke')
    if labels:
        colors = ['lightsalmon', 'tomato', 'orangered', 'firebrick']
        for i in range(len(y)):
            axes.plot(x, y[i], linewidth=4, color=colors[i], label=labels[i])
        axes.legend(facecolor='ghostwhite', fontsize=12, ncol=1, shadow=True, fancybox=False)
    else:
        axes.plot(x, y, linewidth=4, color='tomato')
    if xlim:     
        axes.set_xlim(xlim)
    if ylim:
        axes.set_ylim(ylim)
    if title:
        axes.set_title(title, fontsize=16, pad=20)
    axes.set_xlabel(xlabel, fontsize=14)
    axes.set_ylabel(ylabel, fontsize=14)
    axes.grid(color='w', lw=1, axis='y')
    axes.yaxis.set_ticks_position('none')
    plt.show()


def plot_scatter(data_list, xlabels, ylabels, title=None, alphas=[0.1]*4, figsize=(15,6)):
    fig, axes = plt.subplots(nrows=1, ncols=len(data_list), figsize=figsize)    
    plt.subplots_adjust(wspace=0.4)
    if title:
        plt.suptitle(title, fontsize=16)    
    for i in range(len(data_list)):             
        axes[i].set_facecolor('whitesmoke')
        
        x = np.array(data_list[i].dropna().iloc[:, 0])
        y = np.array(data_list[i].dropna().iloc[:, 1])
        intercept, slope = polyfit(x, y, 1)
        right, left, top, bottom = max(x), min(x), max(y), min(y)
        axes[i].scatter(x, y, alpha=alphas[i], s=70, label='Выборка', marker="X", facecolor='tomato', 
                        edgecolors='tomato')
        axes[i].plot((left - (right-left)/20, right + (right-left)/20), 
                     (intercept + slope*(left - (right-left)/20), intercept + slope*(right + (right-left)/20)), 
                     '-', label='Аппроксимация', color='firebrick', lw=4)
        
        axes[i].set_xlabel(xlabels[i], fontsize=14)
        axes[i].set_ylabel(ylabels[i], fontsize=14)   
        axes[i].legend(facecolor='ghostwhite', title_fontsize=12, ncol=1, shadow=True, fancybox=False)
        axes[i].set_xlim(left - (right-left)/20, right + (right-left)/20)
        axes[i].set_ylim(bottom - (top-bottom)/20, top + (top-bottom)/20)
        
        axes[i].grid(color='w', lw=1, axis='both')
        axes[i].set_axisbelow(True)
        axes[i].xaxis.set_ticks_position('none')
        axes[i].yaxis.set_ticks_position('none')
    plt.show()


def plot_corr_matrix(data, labels=None, annot=True, figsize=(15,10)):
    size = len(data)
    mask = np.triu(np.ones((size, size)), k=1)   
    fig, axes = plt.subplots(figsize=figsize)
    axes = sns.heatmap(data, annot=annot, fmt='.1f', mask=mask, square=False, vmin=-1., vmax=1.)
    axes.set_title('Корреляционная матрица', fontsize=16, pad=20)
    if labels:
        axes.set_xticklabels(labels, rotation=45, fontsize=10, ha='right')
        axes.set_yticklabels(labels, rotation=0, fontsize=10)
    b, t = plt.ylim() 
    plt.ylim(b + 0.5, t - 0.5)
    plt.show()


def plot_kde(data_list, xlabel, labels, xlim, title=None, figsize=(12,5)):
    fig, axes = plt.subplots(figsize=figsize)
    colors = sns.color_palette('magma', len(data_list))
    if title:
        plt.suptitle(title, fontsize=16)             
    axes.set_facecolor('whitesmoke')
    for i in range(len(data_list)):
        data_list[i].plot.kde(ax=axes, color=colors[i], linewidth=4, label=labels[i])
    axes.set_xlabel(xlabel, fontsize=14)
    axes.set_ylabel('Ядерная оценка плотности', fontsize=12)
    axes.set_ylim(0,)
    axes.set_xlim(xlim)
    axes.grid(color='w', lw=1, axis='both')
    axes.xaxis.set_ticks_position('none') 
    axes.yaxis.set_ticks_position('none') 
    axes.legend(facecolor='ghostwhite', ncol=1, shadow=True, fancybox=False)
    plt.show()


def plot_bar(df, title, xlabel, figsize=(15,4)):
    fig, axes = plt.subplots(figsize=figsize)
    plt.suptitle(title, fontsize=16)
    axes.set_facecolor('whitesmoke')
    colors = sns.color_palette('magma', 4)
    df.plot(ax=axes, kind='barh', stacked=True, fontsize=12, colors=colors)
    axes.set_xlabel(xlabel, fontsize=14)
    axes.grid(color='w', lw=1, axis='x')
    axes.set_axisbelow(True)
    axes.xaxis.set_ticks_position('none')
    axes.yaxis.set_ticks_position('none')
    axes.legend(facecolor='ghostwhite', fontsize=12, ncol=1, shadow=True, fancybox=False, bbox_to_anchor=(1.1,1))
    plt.show()


def plot_true_pred(true, pred, pred2=None, q=None, step=None, label='Обученная модель', ylim=None, figsize=(7,5)):
    min_age, max_age = true.min(), true.max()
    fig, axes = plt.subplots(figsize=figsize)
    axes.set_facecolor('whitesmoke')
    if not isinstance(pred2, pd.Series):
        target_df = pd.DataFrame({'true': true, 'pred': pred})
    else:
        target_df = pd.DataFrame({'true': true, 'pred': pred, 'pred2': pred2})
    if step:
        target_df['true'] = pd.cut(target_df['true'], bins=np.arange(0, target_df['true'].max() + step, step))                            .apply(lambda interval: (interval.right + interval.left) / 2)
    else:
        target_df['true'] = pd.qcut(target_df['true'], q=q, duplicates='drop')                            .apply(lambda interval: interval.right)    
    if not isinstance(pred2, pd.Series):    
        axes.plot((min_age, max_age), (min_age, max_age), color='lightsalmon', lw=3, label='Идеальная модель')
        mean = target_df.groupby('true')['pred'].mean()
        error = target_df.groupby('true')['pred'].std()
        axes.plot(np.array(mean.index), mean.values, '-', color='firebrick', linewidth=4, label=label)
        axes.fill_between(np.array(mean.index), (mean - error).values, (mean + error).values,
                          color='tomato', label='Интервал $\mu\pm\sigma$')
    else:
        axes.plot((min_age, max_age), (min_age, max_age), color='mediumslateblue', lw=4, label='Идеальная модель')
        mean = target_df.groupby('true')[['pred', 'pred2']].mean()
        error = target_df.groupby('true')[['pred', 'pred2']].std()
        axes.plot(np.array(mean['pred'].index), mean['pred'].values, '-', color='darkturquoise', 
                  linewidth=4, label=label[0])
        axes.fill_between(np.array(mean['pred'].index), (mean['pred'] - error['pred']).values, 
                          (mean['pred'] + error['pred']).values, alpha=0.3, lw=0,
                          color='darkturquoise', label='Интервал $\mu\pm\sigma$ ({})'.format(label[0]))
        axes.plot(np.array(mean['pred2'].index), mean['pred2'].values, '-', color='tomato', 
                  linewidth=4, label=label[1])
        axes.fill_between(np.array(mean['pred2'].index), (mean['pred2'] - error['pred2']).values, 
                          (mean['pred2'] + error['pred2']).values, alpha=0.3, lw=0,
                          color='tomato', label='Интервал $\mu\pm\sigma$ ({})'.format(label[1]))
    if step:
        axes.set_title('Предсказания модели на объектах с разными значениями recovery (step={})'.format(step),fontsize=16,pad=20)
    else:
        axes.set_title('Предсказания модели на объектах с разными значениями recovery (q={})'.format(q), fontsize=16, pad=20)
    axes.set_xlabel('Recovery, %', fontsize=14)
    axes.set_ylabel('Предсказание модели, %', fontsize=14)
    axes.legend(facecolor='ghostwhite', title_fontsize=12, ncol=1, shadow=True, fancybox=False)
    
    axes.grid(color='w', lw=1, axis='both')
    axes.set_axisbelow(True)
    axes.set_xlim(np.min(mean.index), np.max(mean.index))
    axes.xaxis.set_ticks_position('none')
    axes.yaxis.set_ticks_position('none')
    if ylim:
        axes.set_ylim(ylim)
    
    plt.show()

