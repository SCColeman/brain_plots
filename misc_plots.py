#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Miscellaneous plotting functions.

@author: sebastiancoleman
"""

def regression_plot(x, y, ax):
    
    # regression
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    slope = results.params[1]
    intercept = results.params[0]
    y_pred = intercept + slope*x
    predictions = results.get_prediction(X)
    ci = predictions.conf_int()
    sort_i = np.argsort(x)
    stat, p = pearsonr(x,y)
    
    # plot
    ax.scatter(x,y)
    ax.plot(x[sort_i],y_pred[sort_i], color='red')
    ax.fill_between(x[sort_i], ci[sort_i, 0], ci[sort_i, 1], color='red', alpha=0.2, label='95% Confidence Interval')
    ax.grid('on')
    
    return stat, p

def lineplot(values, times, errors, color, ax):
    """
    Plots a line graph with shaded error regions.
   
    Parameters:
    values (array-like): The y-values to plot.
    times (array-like): The x-values corresponding to the y-values.
    errors (array-like): The errors or uncertainties for the y-values.
    color (str): The color to use for the line and shaded error region.
    ax (matplotlib.axes.Axes): The axes object to plot on.
    """
   
    # Plot the line
    ax.plot(times, values, color=color)
   
    # plot the error
    ax.fill_between(times, values + errors, values - errors, alpha=0.2, edgecolor=color, facecolor=color)
   
    # turn grid on for readability
    ax.grid(True)
