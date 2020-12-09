# -*- coding: utf-8 -*-

from . import plot_settings as pls
from . import plots as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from scipy.stats.kde import gaussian_kde
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None


def find_best_para(para_trace, bins):
    ''' find the best parameter and its 1-sigma/2-sigma for (non) Gaussian distribution '''
    para_trace ,bins = para_trace, bins
    para_trace = np.sort(para_trace)
    hist = np.histogram(para_trace, bins)
    bins, x = hist[0], hist[1]
    
    sort_bin_nums = np.sort(bins)
    best_bins = sort_bin_nums[-7:] # top 7

    best_bins_nums = np.r_[ np.where(bins==best_bins[0])[0], \
    np.where(bins==best_bins[1])[0], np.where(bins==best_bins[2])[0], \
    np.where(bins==best_bins[3])[0], np.where(bins==best_bins[4])[0], \
    np.where(bins==best_bins[5])[0], np.where(bins==best_bins[6])[0] ]
    
    # use the everage of top 7
    best_para = (x[min(best_bins_nums)] + x[max(best_bins_nums) +1])/2.
    
    left = np.where(para_trace <= best_para)[0]
    right = np.where(para_trace > best_para)[0]
    
    para_err_left = best_para - para_trace[int(len(left) * (1-0.6826))]
    para_err_right = para_trace[int(len(right) * 0.6826) + len(left)] - best_para
    
    para = np.r_[best_para, para_err_left, para_err_right]
    return para

def find_best_para_plt(para_trace, bins):
    para_trace ,bins = para_trace, bins
    para_trace = np.sort(para_trace)
    plt.figure()
    hist = plt.hist(para_trace, bins)
    bins, x = hist[0], hist[1]
    
    sort_bin_nums = np.sort(bins)
    best_bins = sort_bin_nums[-7:] # top 7

    best_bins_nums = np.r_[ np.where(bins==best_bins[0])[0], \
    np.where(bins==best_bins[1])[0], np.where(bins==best_bins[2])[0], \
    np.where(bins==best_bins[3])[0], np.where(bins==best_bins[4])[0], \
    np.where(bins==best_bins[5])[0], np.where(bins==best_bins[6])[0] ]

    # use the everage of top 7
    best_para = (x[min(best_bins_nums)] + x[max(best_bins_nums) +1])/2.
    
    left = np.where(para_trace <= best_para)[0]
    right = np.where(para_trace > best_para)[0]
    
    para_err_left = best_para - para_trace[int(len(left) * (1-0.6826))]
    para_err_right = para_trace[int(len(right) * 0.6826) + len(left)] - best_para
    
    para = np.r_[best_para, para_err_left, para_err_right]
    return para
    
def find_best_para2(para_trace, bins):
    para_trace ,bins = para_trace, bins
    para_trace = np.sort(para_trace)
    hist = np.histogram(para_trace, bins)
    bins, x = hist[0], hist[1]
    
    sort_bin_nums = np.sort(bins)
    best_bins = sort_bin_nums[-7:] # top 7

    best_bins_nums = np.r_[ np.where(bins==best_bins[0])[0], \
    np.where(bins==best_bins[1])[0], np.where(bins==best_bins[2])[0], \
    np.where(bins==best_bins[3])[0], np.where(bins==best_bins[4])[0], \
    np.where(bins==best_bins[5])[0], np.where(bins==best_bins[6])[0] ]

    # use the everage of top 7
    best_para = (x[min(best_bins_nums)] + x[max(best_bins_nums) +1])/2.
    
    left = np.where(para_trace <= best_para)[0]
    right = np.where(para_trace > best_para)[0]
    
    para_err_left1 = best_para - para_trace[int(len(left) * (1-0.6826))]
    para_err_right1 = para_trace[int(len(right) * 0.6826) + len(left)] - best_para
    para_err_left2 = best_para - para_trace[int(len(left) * (1-0.9544))]
    para_err_right2 = para_trace[int(len(right) * 0.9544) + len(left)] - best_para
    
    para = np.r_[best_para, para_err_left2,para_err_left1, para_err_right1,para_err_right2]
    return para

def _quantile(x, q, weights=None):
    """
    
    Compute sample quantiles with support for weighted samples.
    
    This is a copy of quantile in corner (https://github.com/dfm/corner.py). Copyright (c) 2013-2015 Daniel Foreman-Mackey.

    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.

    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These

    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.

    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, 100.0 * q)
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()

#old, not used
#def best_params(chains, best_values=None, chi2=None, bins=None, N_decimal=6, print_info=True):
#    '''
#    :param chains: The mcmc results
#    :param best_values: The best values for parameters, 1 dimension array
#    :param bins: The bins for calculate the best values by using "find_best_para"
#    :param N_decimal: The number of decimal -> this parameter is not used
#    
#    :return: the best fit values with 1-sigma error
#    '''
#    
#    if bins is None:
#        bins = 100
#    n_dim = len(chains[0,:]) # the number of parameters (n dimension)
#    if best_values is None:
##        best_values = np.array([find_best_para(chains[:,i], bins)[0] for i in range(n_dim)])
#        best_values = np.array([quantile(chains[:,i], 0.5) for i in range(n_dim)])
#    
#    paras = np.ones((1, 3*n_dim)) # the array to save best fit values and errors for every parameter
#    for i in range(n_dim):
#        paras[0, 3*i] = best_values[i]
#        paras[0, 3*i+1] = best_values[i] - quantile(chains[:,i], 0.1587) # 1sigma left error
#        paras[0, 3*i+2] = quantile(chains[:,i], 0.8413) - best_values[i] # 1sigma right error
#    
#    paras_list = list([i for i in range(n_dim)]) # the list to save best fit values and errors
#    paras_list_1 = list([i for i in range(n_dim)]) # the list to save best fit values and errors
#    for i in range(n_dim):
#        if N_decimal==2:
#            paras_list[i] = str(' & $%.2f_{-%.2f}^{+%.2f}$'%(paras[0,3*i],paras[0,3*i+1],paras[0,3*i+2]))
#            paras_list_1[i] = str(' & $%.2f\pm%.2f$'%(paras[0,3*i], (paras[0,3*i+1]+paras[0,3*i+2])/2.0))
#        elif N_decimal==3:
#            paras_list[i] = str(' & $%.3f_{-%.3f}^{+%.3f}$'%(paras[0,3*i],paras[0,3*i+1],paras[0,3*i+2]))
#            paras_list_1[i] = str(' & $%.3f\pm%.3f$'%(paras[0,3*i], (paras[0,3*i+1]+paras[0,3*i+2])/2.0))
#        elif N_decimal==5:
#            paras_list[i] = str(' & $%.5f_{-%.5f}^{+%.5f}$'%(paras[0,3*i],paras[0,3*i+1],paras[0,3*i+2]))
#            paras_list_1[i] = str(' & $%.5f\pm%.5f$'%(paras[0,3*i], (paras[0,3*i+1]+paras[0,3*i+2])/2.0))
#        elif N_decimal==6:
#            paras_list[i] = str(' & $%.6f_{-%.6f}^{+%.6f}$'%(paras[0,3*i],paras[0,3*i+1],paras[0,3*i+2]))
#            paras_list_1[i] = str(' & $%.6f\pm%.6f$'%(paras[0,3*i], (paras[0,3*i+1]+paras[0,3*i+2])/2.0))
#        elif N_decimal is None:
#            paras_list[i] = str(' & $%s_{-%s}^{+%s}$'%(paras[0,3*i],paras[0,3*i+1],paras[0,3*i+2]))
#            paras_list_1[i] = str(' & $%s\pm%s$'%(paras[0,3*i], (paras[0,3*i+1]+paras[0,3*i+2])/2.0))
#    paras_sum = ''
#    paras_sum_1 = ''
#    for i in range(n_dim):
#        paras_sum += paras_list[i] # Put all the numbers in a row
#        paras_sum_1 += paras_list_1[i] # Put all the numbers in a row
#    
#    if chi2 is not None:
#        paras_sum   = str('$%.3f$'%chi2) + paras_sum
#        paras_sum_1 = str('$%.3f$'%chi2) + paras_sum_1
#    
#    if print_info:
#        print(paras_sum), '\n'
#        print(paras_sum_1)
#        print('\n')
#    return paras_sum, paras_list_1

#new
def best_params(chains, best_values=None, chi2=None, bins=None, N_decimal=6, print_info=True, sigma=1):
    '''
    :param chains: The mcmc results
    :param best_values: The best values for parameters, 1 dimension array
    :param bins: The bins for calculate the best values by using "find_best_para"
    :param N_decimal: The number of decimal -> this parameter is not used
    
    :return: the best fit values with 1-sigma error
    '''
    
    if bins is None:
        bins = 100
    n_dim = len(chains[0,:]) # the number of parameters (n dimension)
    if best_values is None:
#        best_values = np.array([find_best_para(chains[:,i], bins)[0] for i in range(n_dim)])
        best_values = np.array([_quantile(chains[:,i], 0.5) for i in range(n_dim)])
    
    paras = np.ones((1, 7*n_dim)) # the array to save best fit values and errors for every parameter
    for i in range(n_dim):
        paras[0, 7*i] = best_values[i]
        paras[0, 7*i+1] = best_values[i] - _quantile(chains[:,i], 0.1587) # 1sigma left error
        paras[0, 7*i+2] = _quantile(chains[:,i], 0.8413) - best_values[i] # 1sigma right error
        paras[0, 7*i+3] = best_values[i] - _quantile(chains[:,i], 0.0228) # 2sigma left error
        paras[0, 7*i+4] = _quantile(chains[:,i], 0.9772) - best_values[i] # 2sigma right error
        paras[0, 7*i+5] = best_values[i] - _quantile(chains[:,i], 0.0013) # 3sigma left error
        paras[0, 7*i+6] = _quantile(chains[:,i], 0.9987) - best_values[i] # 3sigma right error

    paras_NoRound = np.copy(paras)
    for j in range(7*n_dim):
        paras[0,j] = round(paras[0,j], N_decimal)
    
    paras_list = list([i for i in range(n_dim)]) # the list to save best fit values and errors
    paras_list_1 = list([i for i in range(n_dim)]) # the list to save best fit values and errors
    for i in range(n_dim):
        if sigma==1:
            paras_list[i] = str(' & $%s_{-%s}^{+%s}$'%(paras[0,7*i], paras[0,7*i+1], paras[0,7*i+2]))
            paras_list_1[i] = str(' & $%s\pm%s$'%(paras[0,7*i], round((paras_NoRound[0,7*i+1]+paras_NoRound[0,7*i+2])/2.0, N_decimal) ))
        elif sigma==2:
            paras_list[i] = str(' & $%s_{-%s-%s}^{+%s+%s}$'%(paras[0,7*i],paras[0,7*i+1],paras[0,7*i+3],\
            paras[0,7*i+2],paras[0,7*i+4]))
            paras_list_1[i] = str(' & $%s\pm%s\pm%s$'%(paras[0,7*i], round((paras_NoRound[0,7*i+1]+paras_NoRound[0,7*i+2])/2.0, N_decimal),\
            round((paras_NoRound[0,7*i+3]+paras_NoRound[0,7*i+4])/2.0, N_decimal) ))
        elif sigma==3:
            paras_list[i] = str(' & $%s_{-%s-%s-%s}^{+%s+%s+%s}$'%(paras[0,7*i],paras[0,7*i+1],paras[0,7*i+3],paras[0,7*i+5],\
            paras[0,7*i+2],paras[0,7*i+4],paras[0,7*i+6]))
            paras_list_1[i] = str(' & $%s\pm%s\pm%s\pm%s$'%(paras[0,7*i], round((paras_NoRound[0,7*i+1]+paras_NoRound[0,7*i+2])/2.0, N_decimal),\
            round((paras_NoRound[0,7*i+3]+paras_NoRound[0,7*i+4])/2.0, N_decimal), round((paras_NoRound[0,7*i+5]+paras_NoRound[0,7*i+6])/2.0, N_decimal) ))
    paras_sum = ''
    paras_sum_1 = ''
    for i in range(n_dim):
        paras_sum += paras_list[i] # Put all the numbers in a row
        paras_sum_1 += paras_list_1[i] # Put all the numbers in a row
    
    if chi2 is not None:
        paras_sum   = str('$%.3f$'%chi2) + paras_sum
        paras_sum_1 = str('$%.3f$'%chi2) + paras_sum_1
    
    if print_info:
        print(paras_sum), '\n'
        print(paras_sum_1)
        print('\n')
    return paras_sum, paras_list_1


#%%
def get_best_params(chain):
    '''
    this function is used to get the best-fit parameters calculated with best_params
    
    return: the best-fit values of parameters in numpy array
    '''
    _, best_str = best_params(chain, N_decimal=10, print_info=False)
    param_n = len(best_str)
    
    best_all = []
    for i in range(param_n):
        best_1 = best_str[i]
        best_1 = best_1.split('$')[1].split('\\pm')
        best_1 = np.array([eval(i) for i in best_1])
        best_all.append(best_1)
    best_all = np.array(best_all)
    return best_all
    
def params_deviation(chain_2, chain_1):
    ''' get deviation of parameters in two chains '''
    best_1 = get_best_params(chain_1)
    best_2 = get_best_params(chain_2)
    dev = abs(best_1[:,0]-best_2[:,0]) / np.sqrt(best_1[:,1]**2+best_2[:,1]**2)
    dev = [round(i, 2) for i in dev]
    return dev


def _hist2d(x, y, bins=20, range=None, weights=None, levels=None, smooth=None,
           ax=None, color=None, plot_datapoints=False, plot_density=False,
           plot_contours=True, no_fill_contours=False, fill_contours=False,
           contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
           line_styles=None, line_width=None, legend_label=None, **kwargs):
    """
    
    Plot a 2-D histogram of samples.

    This is a copy of hist2d in corner (https://github.com/dfm/corner.py). Copyright (c) 2013-2015 Daniel Foreman-Mackey.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    y : array_like[nsamples,]
       The samples.

    levels : array_like
        The contour levels to draw.

    ax : matplotlib.Axes
        A axes instance on which to add the 2-D histogram.

    plot_datapoints : bool
        Draw the individual data points.

    plot_density : bool
        Draw the density colormap.

    plot_contours : bool
        Draw the contours.

    no_fill_contours : bool
        Add no filling at all to the contours (unlike setting
        ``fill_contours=False``, which still adds a white fill at the densest
        points).

    fill_contours : bool
        Fill the contours.

    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.

    ### the following is added by Guo-Jian Wang
    line_styles : '-'/'--'/-.'/':' for | 'solid' | 'dashed' | 'dashdot' | 'dotted' 
    """
    if ax is None:
        ax = plt.gca()

    # Set the default range based on the data range if not provided.
    if range is None:
        if "extent" in kwargs:
            logging.warn("Deprecated keyword argument 'extent'. "
                         "Use 'range' instead.")
            range = kwargs["extent"]
        else:
            range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if color is None:
        color = "k"

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # This is the color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)])

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=list(map(np.sort, range)),
                                 weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "'range' argument.")

    if smooth is not None:
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        logging.warning("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([
        X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
        X1,
        X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
    ])
    Y2 = np.concatenate([
        Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
        Y1,
        Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
    ])

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

    if line_styles is None:
        line_styles = '-'  # this is added later
    if line_width is None:
        line_width = 1.  # this is added later

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        contour_edge = ax.contour(X2, Y2, H2.T, V, linestyles=line_styles,linewidths=line_width,\
        **contour_kwargs)# linestyles is added later
        if legend_label is None:
            legend_label = 'Line'
        contour_edge.collections[0].set_label(legend_label) # added later

#    ax.set_xlim(range[0])
#    ax.set_ylim(range[1])
    return


def remove_unreliableData(data, filter_N=0):
    '''
    remove unreliable data, eg: very large or small data for a simulated distribution
    
    :param data: the input data, one-dimension or multi-dimension array
    :param filter_N: a number, the column that was chosen to filter
    
    Note: the selection interval is [best_fit-3*sigma_3l, best_fit+3*sigma_3r]
    '''
    if len(data.shape) == 1:
        data = data.reshape(len(data), 1)

    row_N = len(data[:,0])
    col_N = len(data[0,:])
    
    best_fit = _quantile(data[:,filter_N], 0.5)
    sigma_3l = _quantile(data[:,filter_N], 0.5) - _quantile(data[:,filter_N], 0.0013)
    sigma_3r = _quantile(data[:,filter_N], 0.9987) - _quantile(data[:,filter_N], 0.5)
    data_choose = list([[0] for i in range(col_N)])
    for i in range(row_N):
        if data[i,filter_N] >= best_fit-3*sigma_3l and data[i,filter_N] <= best_fit+3*sigma_3r:
            for j in range(col_N):
                data_choose[j].append(data[i,j])
    data_choose = np.array(data_choose)
    data_choose = data_choose.T
    data_choose = data_choose[1:,:]
    return data_choose

def pdf_1(X, bins, smooth):
    ''' estimate the probability density function for the given data '''
    hist = np.histogram(X, bins, normed=True)
    P, x = hist[0], hist[1]
    x = (x[:-1]+x[1:])/2.0
    P = gaussian_filter(P, smooth)
    return x, P

def pdf_2(X, smooth):
    ''' the same as pdf_1, but using different method '''
    kde = gaussian_kde(X)
    x = np.linspace(min(X), max(X), 100)
    P = kde(x)
    P = gaussian_filter(P, smooth)
    return x, P


class Plot_1d:
    """
    Plot 1-dimensional distribution of parameters.
    """
    def __init__(self, data):
        data = pl.makeList(data)
        N = len(data)
        self.data = data
        self.N = N # N data sets
    
    def __pdf(self, X, bins, smooth, pdf_method='2'):
        if pdf_method=='1':
            x, P = pdf_1(X, bins, smooth)
        elif pdf_method=='2':
            x, P = pdf_2(X, smooth)
        return x, P
    
    def __lims(self, bins,smooth, pdf_method='2'):
        ''' get the upper and lower limits of the axis '''
        max_P = np.ones(self.N)
        max_x = np.ones(self.N)
        min_x = np.ones(self.N)
        for i in range(self.N):
            x, P = self.__pdf(self.data[i], bins, smooth, pdf_method=pdf_method)
            max_x[i], min_x[i] = max(x), min(x)
            max_P[i] = max(P)
        min_x, max_x = min(min_x), max(max_x)
        max_P = max(max_P)
        return min_x,max_x,max_P
    
    # private method of this class
    def __setting(self, bins,smooth,x_lims,labels,ticks_size,major_locator_N,
                  minor_locator,minor_locator_N):
        ''' set the attribute of the panel '''
        max_P = self.__lims(bins,smooth)[2]
        if x_lims is None:
            min_x, max_x = self.__lims(bins,smooth)[:2]
        else:
            min_x, max_x = x_lims[0], x_lims[1]
        max_Y = max_P*1.0618
        
        lims = [min_x, max_x, 0, max_Y]
        
        ax = pls.PlotSettings().setting(lims=lims,labels=labels,ticks_size=ticks_size,\
        major_locator_N=major_locator_N,minor_locator=minor_locator,\
        minor_locator_N=minor_locator_N)
        return ax

    def plot(self, bins=100,smooth=2,x_lims=None,colors=None,line_styles=None,
             line_width=None,labels=None,ticks_size=None,major_locator_N=None,
             minor_locator=True,minor_locator_N=None,legend=False,legend_labels=None,
             legend_size=None,legend_loc=None,equal_1d=True,pdf_method='2',fig_size=None,
             show_title=False,title_fontsize=None,title_sigma=1):
        ''' 
        plot the probability density function (PDF)
        
        :param bins: the number of bins
        :param smooth: the number passed to 'scipy.ndimage.gaussian_filter' to smooth the probability density function.
        :param colors: the colors of lines
        :param line_styles: the styles of lines
        :param line_width: the width of lines
        :param labels: the labels of panels
        :param ticks_size: the size of ticks
        :param major_locator_N: the number of major locators
        :param minor_locator: if True(False), show the minor locators
        :param minor_locator_N: the number of minor locators
        :param equal_1d: making the maximum of the probability density function of multiple data sets equal
        :param pdf_method: '1' or '2', the method to estimate the probability density function for the given data, see the method pdf_1 and pdf_2
        :param show_title: if True(False), show(don't show) the best fit values
        :param title_fontsize: the font size of titles
        :param title_sigma: show 1-sigma errors if title_sigma=1, and 2-sigma errors if title_sigma=2
        '''
        if labels is None:
            labels = [r'$x$', r'${\rm P}(x)$']
        else:
            labels = pl.makeList(labels)
            if len(labels) == 1:
                labels = [labels[0], r'${\rm P}($%s$)$'%labels[0]]
        if line_styles is None:
            line_styles = pl.fiducial_line_styles
        else:
            line_styles = pl.makeList(line_styles)
        if colors is None:
            colors = pl.fiducial_colors
        else:
            colors = pl.makeList(colors)
        if line_width is None:
            line_width = 1.618
        if ticks_size is None:
            ticks_size = 12
        if legend_labels is None:
            legend_labels = [r'$\rm Line$' for i in range(self.N)]
        if legend_size is None:
            legend_size = ticks_size*(1+0.382) #ticks_size/0.618

        # best fit values and 1-sigma & 2-sigma errors
        best_fit = _quantile(self.data[0], 0.5)
        sigma_1l = _quantile(self.data[0], 0.5) - _quantile(self.data[0], 0.1587)
        sigma_1r = _quantile(self.data[0], 0.8413) - _quantile(self.data[0], 0.5)
        sigma_2l = _quantile(self.data[0], 0.5) - _quantile(self.data[0], 0.0228)
        sigma_2r = _quantile(self.data[0], 0.9772) - _quantile(self.data[0], 0.5)
        sigma_3l = _quantile(self.data[0], 0.5) - _quantile(self.data[0], 0.0013)
        sigma_3r = _quantile(self.data[0], 0.9987) - _quantile(self.data[0], 0.5)
        
        if fig_size is None:
            fig = plt.figure(figsize=(6, 4.5))
        else:
            fig = plt.figure(figsize=fig_size)        
        ax = self.__setting(bins,smooth,x_lims,labels,ticks_size,major_locator_N,minor_locator,minor_locator_N)
        if show_title is True:
            if title_fontsize is None:
                title_fontsize = ticks_size*1.5
            if title_sigma == 1:
                ax.set_title(r'%s$=%.3f^{+%.3f}_{-%.3f}$'\
                %(labels[0],best_fit,sigma_1r,sigma_1l),fontsize=title_fontsize)
            elif title_sigma == 2:
                ax.set_title(r'%s$=%.3f^{+%.3f+%.3f}_{-%.3f-%.3f}$'\
                %(labels[0],best_fit,sigma_1r,sigma_2r,sigma_1l,sigma_2l),fontsize=title_fontsize)
            elif title_sigma == 3:
                ax.set_title(r'%s$=%.3f^{+%.3f+%.3f+%.3f}_{-%.3f-%.3f-%.3f}$'\
                %(labels[0],best_fit,sigma_1r,sigma_2r,sigma_3r,sigma_1l,sigma_2l,sigma_3l),fontsize=title_fontsize)
        
        for i in range(self.N):
            x, P = self.__pdf(self.data[i], bins, smooth, pdf_method=pdf_method)
            if equal_1d:
                max_P = self.__lims(bins,smooth)[2]
                P = max_P/max(P) * P
            ax.plot(x, P, line_styles[i], color=colors[i], linewidth=line_width, label=legend_labels[i])
            if legend is True:
                if legend_loc is None:
                    legend_loc = 'upper right'
                ax.legend(loc=legend_loc, borderaxespad=0.5, fontsize=legend_size)
        return fig


class Plot_2d:
    """
    Plot 2-dimensional distribution of parameters.
    """
    def __init__(self, data):
        data = pl.makeListList(data)
        N = len(data) # N data sets
        self.data = data
        self.N = N

    def __lims(self):
        ''' get the upper and lower limits of the axis '''
        max_x = np.ones(self.N)
        min_x = np.ones(self.N)
        max_y = np.ones(self.N)
        min_y = np.ones(self.N)
        for i in range(self.N):
            min_x[i], max_x[i] = min(self.data[i][0]), max(self.data[i][0])
            min_y[i], max_y[i] = min(self.data[i][1]), max(self.data[i][1])
        lims = min(min_x), max(max_x), min(min_y), max(max_y)
        return lims
    
    # private method of this class
    def __setting(self, labels,lims,ticks_size,major_locator_N,
                  minor_locator,minor_locator_N):
        ''' set the attribute of the panel '''
        if lims is None:
            lims = self.__lims()

        ax = pls.PlotSettings().setting(lims=lims,labels=labels,\
        ticks_size=ticks_size,major_locator_N=major_locator_N,\
        minor_locator=minor_locator,minor_locator_N=minor_locator_N)
        return ax
                      
    def plot(self, sigma=2,bins=None,smooth=2,lims=None,colors=None,line_styles=None,
             line_width=None,labels=None,best_values=None,best_values_styles=None,
             ticks_size=None,major_locator_N=None,minor_locator=True,
             minor_locator_N=None,fill_contours=True,legend=False,legend_labels=None,
             legend_size=None,legend_loc=None,bbox_to_anchor=(1,1),show_title=False,title_fontsize=None,
             title_sigma=1,fig_size=None):
        '''
        plot 2-D contours
        
        :param sigma: contour levels, if sigma=2(3), plot with 2-sigma(3-sigma) contour levels
        :param bins: the number of bins
        :param smooth: the number passed to 'scipy.ndimage.gaussian_filter' to smooth the probability density function.
        :param colors: the colors of lines
        :param line_styles: the styles of lines
        :param line_width: the width of lines
        :param labels: labels of the panel
        :param best_values: the best fit values
        :param best_value_styles: the styles of best fit values        
        :param ticks_size: the size of ticks
        :param major_locator_N: the number of major locators
        :param minor_locator: if True(False), show the minor locators
        :param minor_locator_N: the number of minor locators
        :param show_title: if True(False), show(don't show) the best fit values
        :param title_fontsize: the font size of titles
        :param title_sigma: show 1-sigma errors if title_sigma=1, and 2-sigma errors if title_sigma=2
        '''
        if bins is None:
            bins = 100
        if colors is None:
            colors = pl.fiducial_colors[:self.N]
        else:
            colors = pl.makeList(colors)
        if line_styles is None:
            line_styles = pl.fiducial_line_styles
        else:
            line_styles = pl.makeList(line_styles)
        if best_values is not None:
            best_values = pl.makeListList(best_values)
        if line_width is None:
            line_width = 1.618
        if best_values_styles is None:
            best_values_styles = 'o'
        if sigma == 2:
            levels = (0.6826, 0.9544)
        elif sigma == 3:
            levels = (0.6826, 0.9544, 0.9974)
        if ticks_size is None:
            ticks_size = 12
        if legend_labels is None:
            legend_labels = [r'$\rm Line$' for i in range(self.N)]
        if legend_size is None:
            legend_size = ticks_size*(1+0.382) #ticks_size/0.618
        
        # best fit values and 1-sigma & 2-sigma errors
        best_fit = np.ones(2)
        sigma_1l = np.ones(2); sigma_1r = np.ones(2)
        sigma_2l = np.ones(2); sigma_2r = np.ones(2)
        sigma_3l = np.ones(2); sigma_3r = np.ones(2)
        for i in range(2):
            best_fit[i] = _quantile(self.data[0][i], 0.5)
            sigma_1l[i] = _quantile(self.data[0][i], 0.5) - _quantile(self.data[0][i], 0.1587)
            sigma_1r[i] = _quantile(self.data[0][i], 0.8413) - _quantile(self.data[0][i], 0.5)
            sigma_2l[i] = _quantile(self.data[0][i], 0.5) - _quantile(self.data[0][i], 0.0228)
            sigma_2r[i] = _quantile(self.data[0][i], 0.9772) - _quantile(self.data[0][i], 0.5)
            sigma_3l[i] = _quantile(self.data[0][i], 0.5) - _quantile(self.data[0][i], 0.0013)
            sigma_3r[i] = _quantile(self.data[0][i], 0.9987) - _quantile(self.data[0][i], 0.5)
        
        if fig_size is None:
            fig = plt.figure(figsize=(6, 4.5))
        else:
            fig = plt.figure(figsize=fig_size)
        ax = self.__setting(labels,lims,ticks_size,major_locator_N,minor_locator,minor_locator_N)
        if show_title is True:
            if title_fontsize is None:
                title_fontsize = ticks_size*1.5
            if title_sigma == 1:
                ax.set_title(r'%s$=%.3f^{+%.3f}_{-%.3f}$; %s=$%.3f^{+%.3f}_{-%.3f}$'\
                %(labels[0],best_fit[0],sigma_1r[0],sigma_1l[0],labels[1],best_fit[1],sigma_1r[1],sigma_1l[1]),fontsize=title_fontsize)
            elif title_sigma == 2:
                ax.set_title(r'%s$=%.3f^{+%.3f+%.3f}_{-%.3f-%.3f}$; %s$=%.3f^{+%.3f+%.3f}_{-%.3f-%.3f}$'\
                %(labels[0],best_fit[0],sigma_1r[0],sigma_2r[0],sigma_1l[0],sigma_2l[0],
                  labels[1],best_fit[1],sigma_1r[1],sigma_2r[1],sigma_1l[1],sigma_2l[1]),fontsize=title_fontsize)
            elif title_sigma == 3:
                ax.set_title(r'%s$=%.3f^{+%.3f+%.3f+%.3f}_{-%.3f-%.3f-%.3f}$; %s$=%.3f^{+%.3f+%.3f+%.3f}_{-%.3f-%.3f-%.3f}$'\
                %(labels[0],best_fit[0],sigma_1r[0],sigma_2r[0],sigma_3r[0],sigma_1l[0],sigma_2l[0],sigma_3l[0],
                  labels[1],best_fit[1],sigma_1r[1],sigma_2r[1],sigma_3r[1],sigma_1l[1],sigma_2l[1],sigma_3l[1]),fontsize=title_fontsize)
        
        for i in range(self.N):
            _hist2d(self.data[i][0], self.data[i][1], bins=bins,levels=levels,\
            smooth=smooth,line_styles=line_styles[i],color=colors[i],\
            fill_contours=fill_contours,line_width=line_width,legend_label=legend_labels[i])
            if best_values is not None:
                ax.plot(best_values[i][0],best_values[i][1],best_values_styles,color=colors[i])
            if legend is True:
                if legend_loc is None:
                    legend_loc = 'upper right'
                if fill_contours is False:
                    ax.legend(loc=legend_loc,borderaxespad=0.5,fontsize=legend_size)
                elif fill_contours is True:
                    patch = []
                    legend_colors = colors
                    for i in range(len(legend_colors)):
                        patch.append(mpatches.Patch(color=legend_colors[i], label=legend_labels[i]))
                    ax.legend(handles=patch, bbox_to_anchor=bbox_to_anchor, loc=legend_loc,\
                    borderaxespad=0.5, fontsize=legend_size)
        return fig


class Contours:
    ''' Plot 1-D and 2-D contours for several sets of data. '''
    
    def __init__(self, data):
        data = pl.makeList(data)
        # if data.shape = (10000,), one can't plot by using Contours(data).plot(),
        # so, here we need solve this problem by reshapeing the data
        if len(data[0].shape)==1:
            data[0] = data[0].reshape(len(data[0]), 1)
        N = len(data[0][0,:]) # N dimensions
        n = len(data) # n data sets
        self.data = data
        self.N = N
        self.n = n

    def __pdf(self, X, bins, smooth, pdf_method='2'):
        if pdf_method=='1':
            x, P = pdf_1(X, bins, smooth)
        elif pdf_method=='2':
            x, P = pdf_2(X, smooth)
        return x, P
        
    def __lims(self, bins, smooth, pdf_method='2'):
        ''' get the upper and lower limits of the axis '''
        ones_N = np.ones(self.N)
        min_xx = [[m] for m in ones_N]
        max_xx = [[m] for m in ones_N]
        min_PP = [[m] for m in ones_N]
        max_PP = [[m] for m in ones_N]
        for i in range(self.N):
            ones_n = np.ones(self.n)
            min_x = [[m] for m in ones_n]
            max_x = [[m] for m in ones_n]
            min_P = [[m] for m in ones_n]
            max_P = [[m] for m in ones_n]
            for j in range(self.n):
                x, P = self.__pdf(self.data[j][:,i], bins, smooth, pdf_method=pdf_method)
                max_x[j], min_x[j] = max(x), min(x)
                max_P[j], min_P[j] = max(P), min(P)
            min_xx[i], max_xx[i], min_PP[i], max_PP[i] = min(min_x), max(max_x), min(min_P), max(max_P)
        return min_xx, max_xx, min_PP, max_PP
    
    def __plot_1d(self, ax,X,bins,color,line_style,line_width,smooth,ticks_size,
                  legend_label=None, legend=False,legend_colors=None, fill_contours=False,
                  legend_size=None,equal_1d=False,std_P=1,pdf_method='2'):
        
        x, P = self.__pdf(X, bins, smooth, pdf_method=pdf_method)
        if equal_1d:
            P = std_P/max(P) * P
        ax.plot(x, P, line_style, color=color, linewidth=line_width, label=legend_label)
        # add legend
        if legend is True:
            legend_loc = 'upper right'
#            legend_loc = 'center'
            if legend_size is None:
                legend_size = ticks_size*(1+0.382) #ticks_size/0.618
            if fill_contours is False:
                ax.legend(bbox_to_anchor=(1, self.N), loc=legend_loc,\
                borderaxespad=0., fontsize=legend_size)
            elif fill_contours is True:
                patch = []
                for i in range(len(legend_colors)):
                    patch.append(mpatches.Patch(color=legend_colors[i], label=legend_label[i]))
                ax.legend(handles=patch, bbox_to_anchor=(1, self.N), loc=legend_loc,\
                borderaxespad=0., fontsize=legend_size)
        return
    
    
    # All required parameters must be placed before any default arguments.
    # def __plot_2d(self, XY, ..., best_value=None, best_value_style) is wrong
    # def __plot_2d(self, XY, ..., fill_contours=True,best_value=None) is right
    
    def __plot_2d(self, ax,XY, bins, levels, smooth, line_style, line_width, color,
                  best_value_style, best_value_color, fill_contours=True, best_value=None,
                  legend_label=None,legend=False,legend_colors=None,ticks_size=None,legend_size=None):
    
        _hist2d(XY[0], XY[1], bins=bins,levels=levels,smooth=smooth,\
        line_styles=line_style,color=color,fill_contours=fill_contours,\
        line_width=line_width,legend_label=legend_label)
        if best_value is not None:
            ax.plot(best_value[0],best_value[1],best_value_style,color=best_value_color)

        # add legend
        if legend is True:
            legend_loc = 'upper right'
#            legend_loc = 'center'
            if legend_size is None:
                legend_size = ticks_size*(1+0.382) #ticks_size/0.618
            if fill_contours is False:
                ax.legend(bbox_to_anchor=(1, self.N-1), loc=legend_loc,\
                borderaxespad=0., fontsize=legend_size)#
            elif fill_contours is True:
                patch = []
                for i in range(len(legend_colors)):
                    patch.append(mpatches.Patch(color=legend_colors[i], label=legend_label[i]))
                ax.legend(handles=patch, bbox_to_anchor=(1, self.N-1), loc=legend_loc,\
                borderaxespad=0., fontsize=legend_size)
        return
    
    def __intersecting_line(self, ax, point, xy_lim, line_color, line_style=None):
        ''' Draw the intersection lines through the best value points '''
        if line_style is None:
            line_style = '--'
            
        if len(xy_lim) == 4:
            #transverse line
            ax.plot([xy_lim[0],xy_lim[1]], [point[1],point[1]], line_style, color=line_color)
            # vertical line
            ax.plot([point[0],point[0]], [xy_lim[2],xy_lim[3]], line_style, color=line_color)
        elif len(xy_lim) == 2:
            # vertical line
            ax.plot([point[0],point[0]], [xy_lim[0],xy_lim[1]], line_style, color=line_color)
        return            
    
    def plot(self, bins=None,labels=None,colors=None,line_styles=None,
             same_line_styles=True,sigma=2,smooth=2,minor_locator=True,
             major_locator_N=None,minor_locator_N=None,ticks_size=None,
             layout_adjust=None,lims=None,line_width=None,show_titles=False,
             title_fontsize=None,title_sigma=1,fill_contours=True,
             best_values=None,best_value_styles=None,best_value_colors=None,
             show_best_value_lines=False,legend=False,legend_labels=None,
             legend_size=None,equal_1d=True,pdf_method='2'):
        '''
        Plot 1-D and 2-D contours for several sets of data.
        
        :param bins: bins for 1-D and 2-D contours
        :param labels: labels of panels
        :param colors: colors of 1-D and 2-D contours
        :param line_styles: lines styles of 1-D and 2-D contour for a set of data
        :param same_line_styles: if True, 2-D contour have the same line styles with 1-D contours
        :param sigma: contour levels, if sigma=2(3), plot with 2-sigma(3-sigma) contour levels
        :param smooth: the number passed to 'scipy.ndimage.gaussian_filter' to smooth the probability density function
        :param minor_locator: small scale of xticks and yticks
        :param major_locator_N: the number of major scale of xticks and yticks
        :param minor_locator_N: the number of minor locators
        :param ticks_size: the size of xticklabel and yticklabel
        :param layout_adjust: adjust the distance of panels
        :param lims: xlim and ylim for every panels, lims=[[min_x1, min_x2,],[max_x1,max_x2,]]
        :param line_width: the width of lines for 1-D contours
        :param show_titles: if True(False), show(don't show) the best fit values
        :param title_fontsize: the font size of titles
        :param title_sigma: show 1-sigma errors if title_sigma=1, and 2-sigma errors if title_sigma=2
        :param fill_contours: fill contours with colors
        :param best_values: plot the best fit value in the 2-D contour
        :param best_value_styles: the styles of best values, eg: 'o'/'^'
        :param equal_1d: making the maximum of the probability density function of multiple data sets equal
        :param pdf_method: '1' or '2', the method to estimate the probability density function for the given data, see the method pdf_1 and pdf_2
        '''
        if bins is None:
            bins = 100
        if labels is None:
            labels = [r"$x_{%s}$"%i for i in range(self.N)]
        if colors is None:
#            colors = pl.fiducial_colors[:self.n]
            colors = pl.get_fiducial_colors(self.n)
        else:
            colors = pl.makeList(colors)
        if line_styles is None:
#            line_styles = pl.fiducial_line_styles[:self.n]
            line_styles = pl.get_fiducial_line_styles(self.n)
        else:
            line_styles = pl.makeList(line_styles)
        if same_line_styles is True:
            linestyles_2d = line_styles #[[m] for m in line_styles]
        else:
            linestyles_2d = ['-' for m in range(self.n)]
        if layout_adjust is None:
            layout_adjust = [0.025, 0.025]
        smooth_1d = smooth
        smooth_2d = smooth/2.0
        std_P = self.__lims(bins, smooth_1d, pdf_method=pdf_method)[3]#max_P
        max_Y = [i*1.0618 for i in std_P] # max_P = P*1.0618
        if lims is None:
            min_x, max_x = self.__lims(bins, smooth_1d, pdf_method=pdf_method)[:2]
        else:
            min_x, max_x = lims[0], lims[1]
        if line_width is None:
            line_width = 1.618
        if fill_contours is True:
            line_width_2d = line_width/5.0
        else:
            line_width_2d = line_width
        if sigma == 2:
            levels = (0.6826, 0.9544)
        elif sigma == 3:
            levels = (0.6826, 0.9544, 0.9974)
        if best_values is not None:
            best_values = pl.makeListList(best_values)
        if best_value_styles is None:
            best_value_styles = ['o' for i in range(self.n)]
        if best_value_colors is None:
            best_value_colors = colors
        else:
            best_value_colors = pl.makeList(best_value_colors)
        rotation = 45
        if major_locator_N is None:
            major_locator_N = 5.0
        if minor_locator_N is None:
            minor_locator_N = 5.0
        if ticks_size is None:
            ticks_size = 12
        if legend_labels is None:
            legend_labels = [r'$\rm Line$' for i in range(self.n)]
            
        # best fit values and 1-sigma & 2-sigma errors
        best_fit = np.ones(self.N)
        sigma_1l = np.ones(self.N); sigma_1r = np.ones(self.N)
        sigma_2l = np.ones(self.N); sigma_2r = np.ones(self.N)
        sigma_3l = np.ones(self.N); sigma_3r = np.ones(self.N)
        for i in range(self.N):
            best_fit[i] = _quantile(self.data[0][:,i], 0.5)
            sigma_1l[i] = _quantile(self.data[0][:,i], 0.5) - _quantile(self.data[0][:,i], 0.1587)
            sigma_1r[i] = _quantile(self.data[0][:,i], 0.8413) - _quantile(self.data[0][:,i], 0.5)
            sigma_2l[i] = _quantile(self.data[0][:,i], 0.5) - _quantile(self.data[0][:,i], 0.0228)
            sigma_2r[i] = _quantile(self.data[0][:,i], 0.9772) - _quantile(self.data[0][:,i], 0.5)
            sigma_3l[i] = _quantile(self.data[0][:,i], 0.5) - _quantile(self.data[0][:,i], 0.0013)
            sigma_3r[i] = _quantile(self.data[0][:,i], 0.9987) - _quantile(self.data[0][:,i], 0.5)
        
        # plot contours
        fig = plt.figure(figsize=(self.N*2, self.N*2))
        for i in range(self.N):
            for j in range(self.N):
                location = (self.N, self.N, i*self.N+j+1)
                # setting property of 1-D subplots
                if i == j:
                    xy_lims = [ min_x[j], max_x[j], 0, max_Y[i] ]
                    if i == self.N-1:
                        ax = pls.PlotSettings().setting(location=location,lims=xy_lims,\
                        labels=[labels[j],labels[i]],ticks_size=ticks_size,major_locator_length=4,\
                        rotation=rotation,major_locator_N=major_locator_N,\
                        minor_locator=minor_locator,minor_locator_N=minor_locator_N,\
                        show_xticks=True,show_yticks=False,show_xticklabels=True,\
                        show_yticklabels=False,show_xlabel=True,show_ylabel=False,old_ticksStyle=True,rightTopTicks=False)
                        if show_titles is True:
                            if title_sigma == 1:
                                if title_fontsize is None:
                                    title_fontsize = ticks_size*1.5
                                ax.set_title(r'%s$=%.3f^{+%.3f}_{-%.3f}$'\
                                %(labels[j],best_fit[j],sigma_1r[j],sigma_1l[j]),fontsize=title_fontsize)
                            elif title_sigma == 2:
                                if title_fontsize is None:
                                    title_fontsize = ticks_size*1.5
                                ax.set_title(r'%s$=%.3f^{+%.3f+%.3f}_{-%.3f-%.3f}$'\
                                %(labels[j],best_fit[j],sigma_1r[j],sigma_2r[j],sigma_1l[j],sigma_2l[j]),fontsize=title_fontsize)
                            elif title_sigma == 3:
                                if title_fontsize is None:
                                    title_fontsize = ticks_size*1.5
                                ax.set_title(r'%s$=%.3f^{+%.3f+%.3f+%.3f}_{-%.3f-%.3f-%.3f}$'\
                                %(labels[j],best_fit[j],sigma_1r[j],sigma_2r[j],sigma_3r[j],sigma_1l[j],sigma_2l[j],sigma_3l[j]),fontsize=title_fontsize)
                    else:
                        ax = pls.PlotSettings().setting(location=location,lims=xy_lims,\
                        labels=[labels[j],labels[i]],ticks_size=ticks_size,major_locator_length=4,\
                        rotation=rotation,major_locator_N=major_locator_N,\
                        minor_locator=minor_locator,minor_locator_N=minor_locator_N,\
                        show_xticks=True,show_yticks=False,show_xticklabels=False,\
                        show_yticklabels=False,show_xlabel=False,show_ylabel=False,old_ticksStyle=True,rightTopTicks=False)
                        if show_titles is True:
                            if title_sigma == 1:
                                if title_fontsize is None:
                                    title_fontsize = ticks_size*1.5
                                ax.set_title(r'%s$=%.3f^{+%.3f}_{-%.3f}$'\
                                %(labels[j],best_fit[j],sigma_1r[j],sigma_1l[j]),fontsize=title_fontsize)
                            elif title_sigma == 2:
                                if title_fontsize is None:
                                    title_fontsize = ticks_size*1.5
                                ax.set_title(r'%s$=%.3f^{+%.3f+%.3f}_{-%.3f-%.3f}$'\
                                %(labels[j],best_fit[j],sigma_1r[j],sigma_2r[j],sigma_1l[j],sigma_2l[j]),fontsize=title_fontsize)
                            elif title_sigma == 3:
                                if title_fontsize is None:
                                    title_fontsize = ticks_size*1.5
                                ax.set_title(r'%s$=%.3f^{+%.3f+%.3f+%.3f}_{-%.3f-%.3f-%.3f}$'\
                                %(labels[j],best_fit[j],sigma_1r[j],sigma_2r[j],sigma_3r[j],sigma_1l[j],sigma_2l[j],sigma_3l[j]),fontsize=title_fontsize)
                    # plot 1-D contours
                    for k in range(self.n):
                        if legend is False:
                            self.__plot_1d(ax,self.data[k][:,i],bins,colors[k],\
                            line_styles[k],line_width,smooth_1d, ticks_size,equal_1d=equal_1d,std_P=std_P[i],pdf_method=pdf_method)
                        elif legend is True:
                            if i == self.N-1:
                                if fill_contours is False:
                                    self.__plot_1d(ax,self.data[k][:,i],bins,colors[k],\
                                    line_styles[k],line_width,smooth_1d, ticks_size, legend=legend,\
                                    legend_label=legend_labels[k], legend_size=legend_size,equal_1d=equal_1d,std_P=std_P[i],pdf_method=pdf_method)
                                elif fill_contours is True:
                                    self.__plot_1d(ax,self.data[k][:,i],bins,colors[k],\
                                    line_styles[k],line_width,smooth_1d, ticks_size, legend=legend,\
                                    legend_label=legend_labels,legend_colors=colors,legend_size=legend_size,\
                                    fill_contours=fill_contours,equal_1d=equal_1d,std_P=std_P[i],pdf_method=pdf_method)
                            else:
                                self.__plot_1d(ax,self.data[k][:,i],bins,colors[k],\
                                line_styles[k],line_width,smooth_1d, ticks_size,equal_1d=equal_1d,std_P=std_P[i],pdf_method=pdf_method)
                        if show_best_value_lines is True:
                            self.__intersecting_line(ax, [best_values[k][j],best_values[k][i]],\
                            xy_lims[2:], best_value_colors[k])
                # setting property of 2-D subplots
                elif i > j:
                    xy_lims = [ min_x[j], max_x[j], min_x[i], max_x[i] ]
                    if i==self.N-1 and j!=0:
                        ax = pls.PlotSettings().setting(location=location,lims=xy_lims,\
                        labels=[labels[j],labels[i]],ticks_size=ticks_size,major_locator_length=4,\
                        rotation=rotation,major_locator_N=major_locator_N,\
                        minor_locator=minor_locator,minor_locator_N=minor_locator_N,\
                        show_xticks=True,show_yticks=True,show_xticklabels=True,\
                        show_yticklabels=False,show_xlabel=True,show_ylabel=False,old_ticksStyle=True)
                    elif j==0 and i!=self.N-1:
                        ax = pls.PlotSettings().setting(location=location,lims=xy_lims,\
                        labels=[labels[j],labels[i]],ticks_size=ticks_size,major_locator_length=4,\
                        rotation=rotation,major_locator_N=major_locator_N,\
                        minor_locator=minor_locator,minor_locator_N=minor_locator_N,\
                        show_xticks=True,show_yticks=True,show_xticklabels=False,\
                        show_yticklabels=True,show_xlabel=False,show_ylabel=True,old_ticksStyle=True)
                    elif i==self.N-1 and j==0:
                        ax = pls.PlotSettings().setting(location=location,lims=xy_lims,\
                        labels=[labels[j],labels[i]],ticks_size=ticks_size,major_locator_length=4,\
                        rotation=rotation,major_locator_N=major_locator_N,\
                        minor_locator=minor_locator,minor_locator_N=minor_locator_N,\
                        show_xticks=True,show_yticks=True,show_xticklabels=True,\
                        show_yticklabels=True,show_xlabel=True,show_ylabel=True,old_ticksStyle=True)
                    else:
                        ax = pls.PlotSettings().setting(location=location,lims=xy_lims,\
                        labels=[labels[j],labels[i]],ticks_size=ticks_size,major_locator_length=4,\
                        rotation=rotation,major_locator_N=major_locator_N,\
                        minor_locator=minor_locator,minor_locator_N=minor_locator_N,\
                        show_xticks=True,show_yticks=True,show_xticklabels=False,\
                        show_yticklabels=False,show_xlabel=False,show_ylabel=False,old_ticksStyle=True)
                    # plot 2-D contours
                    for k in range(self.n):
                        if best_values is None:
                            self.__plot_2d(ax,[self.data[k][:,j],self.data[k][:,i]], bins,\
                            levels, smooth_2d, linestyles_2d[k], line_width_2d, colors[k],\
                            best_value_styles[k], best_value_colors[k], fill_contours=fill_contours,\
                            best_value=None)
                        else:
                            self.__plot_2d(ax,[self.data[k][:,j],self.data[k][:,i]], bins,\
                            levels, smooth_2d, linestyles_2d[k], line_width_2d, colors[k],\
                            best_value_styles[k], best_value_colors[k], fill_contours=fill_contours,\
                            best_value=[best_values[k][j],best_values[k][i]])
                            if show_best_value_lines is True:
                                self.__intersecting_line(ax, [best_values[k][j],best_values[k][i]],\
                                xy_lims, best_value_colors[k])
        fig.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=layout_adjust[0],hspace=layout_adjust[1])
        return fig


    #to be improved
    def plot_2d(self, bins=None,labels=None,colors=None,line_styles=None,
             same_line_styles=True,sigma=2,smooth=2,minor_locator=True,
             major_locator_N=None,minor_locator_N=None,ticks_size=None,
             layout_adjust=None,lims=None,line_width=None,show_titles=False,
             title_fontsize=None,title_sigma=1,fill_contours=True,
             best_values=None,best_value_styles=None,best_value_colors=None,
             show_best_value_lines=False,legend=False,legend_labels=None,
             legend_size=None,equal_1d=True,pdf_method='2'):
        '''
        Plot 2-D contours for several sets of data.
        
        :param bins: bins for 1-D and 2-D contours
        :param labels: labels of panels
        :param colors: colors of 1-D and 2-D contours
        :param line_styles: lines styles of 1-D and 2-D contour for a set of data
        :param same_line_styles: if True, 2-D contour have the same line styles with 1-D contours
        :param sigma: contour levels, if sigma=2(3), plot with 2-sigma(3-sigma) contour levels
        :param smooth: the number passed to 'scipy.ndimage.gaussian_filter' to smooth the probability density function
        :param minor_locator: small scale of xticks and yticks
        :param major_locator_N: the number of major scale of xticks and yticks
        :param minor_locator_N: the number of minor locators
        :param ticks_size: the size of xticklabel and yticklabel
        :param layout_adjust: adjust the distance of panels
        :param lims: xlim and ylim for every panels, lims=[[min_x1, min_x2,],[max_x1,max_x2,]]
        :param line_width: the width of lines for 1-D contours
        :param show_titles: if True(False), show(don't show) the best fit values
        :param title_fontsize: the font size of titles
        :param title_sigma: show 1-sigma errors if title_sigma=1, and 2-sigma errors if title_sigma=2
        :param fill_contours: fill contours with colors
        :param best_values: plot the best fit value in the 2-D contour
        :param best_value_styles: the styles of best values, eg: 'o'/'^'
        :param equal_1d: making the maximum of the probability density function of multiple data sets equal
        :param pdf_method: '1' or '2', the method to estimate the probability density function for the given data, see the method pdf_1 and pdf_2
        '''
        if bins is None:
            bins = 100
        if labels is None:
            labels = [r"$x_{%s}$"%i for i in range(self.N)]
        if colors is None:
#            colors = pl.fiducial_colors[:self.n]
            colors = pl.get_fiducial_colors(self.n)
        else:
            colors = pl.makeList(colors)
        if line_styles is None:
#            line_styles = pl.fiducial_line_styles[:self.n]
            line_styles = pl.get_fiducial_line_styles(self.n)
        else:
            line_styles = pl.makeList(line_styles)
        if same_line_styles is True:
            linestyles_2d = line_styles #[[m] for m in line_styles]
        else:
            linestyles_2d = ['-' for m in range(self.n)]
        if layout_adjust is None:
            layout_adjust = [0.025, 0.025]
        smooth_1d = smooth
        smooth_2d = smooth/2.0
        std_P = self.__lims(bins, smooth_1d, pdf_method=pdf_method)[3]#max_P
        max_Y = [i*1.0618 for i in std_P] # max_P = P*1.0618
        if lims is None:
            min_x, max_x = self.__lims(bins, smooth_1d, pdf_method=pdf_method)[:2]
        else:
            min_x, max_x = lims[0], lims[1]
        if line_width is None:
            line_width = 1.618
        if fill_contours is True:
            line_width_2d = line_width/5.0
        else:
            line_width_2d = line_width
        if sigma == 2:
            levels = (0.6826, 0.9544)
        elif sigma == 3:
            levels = (0.6826, 0.9544, 0.9974)
        if best_values is not None:
            best_values = pl.makeListList(best_values)
        if best_value_styles is None:
            best_value_styles = ['o' for i in range(self.n)]
        if best_value_colors is None:
            best_value_colors = colors
        else:
            best_value_colors = pl.makeList(best_value_colors)
        rotation = 45
        if major_locator_N is None:
            major_locator_N = 5.0
        if minor_locator_N is None:
            minor_locator_N = 5.0
        if ticks_size is None:
            ticks_size = 12
        if legend_labels is None:
            legend_labels = [r'$\rm Line$' for i in range(self.n)]
            
        # best fit values and 1-sigma & 2-sigma errors
        best_fit = np.ones(self.N)
        sigma_1l = np.ones(self.N); sigma_1r = np.ones(self.N)
        sigma_2l = np.ones(self.N); sigma_2r = np.ones(self.N)
        sigma_3l = np.ones(self.N); sigma_3r = np.ones(self.N)
        for i in range(self.N):
            best_fit[i] = _quantile(self.data[0][:,i], 0.5)
            sigma_1l[i] = _quantile(self.data[0][:,i], 0.5) - _quantile(self.data[0][:,i], 0.1587)
            sigma_1r[i] = _quantile(self.data[0][:,i], 0.8413) - _quantile(self.data[0][:,i], 0.5)
            sigma_2l[i] = _quantile(self.data[0][:,i], 0.5) - _quantile(self.data[0][:,i], 0.0228)
            sigma_2r[i] = _quantile(self.data[0][:,i], 0.9772) - _quantile(self.data[0][:,i], 0.5)
            sigma_3l[i] = _quantile(self.data[0][:,i], 0.5) - _quantile(self.data[0][:,i], 0.0013)
            sigma_3r[i] = _quantile(self.data[0][:,i], 0.9987) - _quantile(self.data[0][:,i], 0.5)
        
        # plot contours
        fig = plt.figure(figsize=((self.N-1)*2, (self.N-1)*2))#
        for i in range(self.N-1):
            for j in range(self.N-1):
                location = (self.N-1, self.N-1, i*(self.N-1)+j+1)
                # setting property of 2-D subplots
                if i >= j:
                    xy_lims = [ min_x[j], max_x[j], min_x[i+1], max_x[i+1] ]
                    if i==self.N-2 and j!=0:
                        ax = pls.PlotSettings().setting(location=location,lims=xy_lims,\
                        labels=[labels[j],labels[i+1]],ticks_size=ticks_size,major_locator_length=4,\
                        rotation=rotation,major_locator_N=major_locator_N,\
                        minor_locator=minor_locator,minor_locator_N=minor_locator_N,\
                        show_xticks=True,show_yticks=True,show_xticklabels=True,\
                        show_yticklabels=False,show_xlabel=True,show_ylabel=False,old_ticksStyle=True)
                    elif j==0 and i!=self.N-2:
                        ax = pls.PlotSettings().setting(location=location,lims=xy_lims,\
                        labels=[labels[j],labels[i+1]],ticks_size=ticks_size,major_locator_length=4,\
                        rotation=rotation,major_locator_N=major_locator_N,\
                        minor_locator=minor_locator,minor_locator_N=minor_locator_N,\
                        show_xticks=True,show_yticks=True,show_xticklabels=False,\
                        show_yticklabels=True,show_xlabel=False,show_ylabel=True,old_ticksStyle=True)
                    elif i==self.N-2 and j==0:
                        ax = pls.PlotSettings().setting(location=location,lims=xy_lims,\
                        labels=[labels[j],labels[i+1]],ticks_size=ticks_size,major_locator_length=4,\
                        rotation=rotation,major_locator_N=major_locator_N,\
                        minor_locator=minor_locator,minor_locator_N=minor_locator_N,\
                        show_xticks=True,show_yticks=True,show_xticklabels=True,\
                        show_yticklabels=True,show_xlabel=True,show_ylabel=True,old_ticksStyle=True)
                    else:
                        ax = pls.PlotSettings().setting(location=location,lims=xy_lims,\
                        labels=[labels[j],labels[i+1]],ticks_size=ticks_size,major_locator_length=4,\
                        rotation=rotation,major_locator_N=major_locator_N,\
                        minor_locator=minor_locator,minor_locator_N=minor_locator_N,\
                        show_xticks=True,show_yticks=True,show_xticklabels=False,\
                        show_yticklabels=False,show_xlabel=False,show_ylabel=False,old_ticksStyle=True)
                    # plot 2-D contours
                    for k in range(self.n):
                        if legend and i==self.N-2 and j==self.N-2:
                            _legend = True
                        else:
                            _legend = False
                        
                        if best_values is None:
                            self.__plot_2d(ax,[self.data[k][:,j],self.data[k][:,i+1]], bins,\
                            levels, smooth_2d, linestyles_2d[k], line_width_2d, colors[k],\
                            best_value_styles[k], best_value_colors[k], fill_contours=fill_contours,best_value=None,\
                            legend_label=legend_labels[k],legend=_legend,legend_colors=colors,ticks_size=ticks_size,legend_size=legend_size)
                        else:
                            self.__plot_2d(ax,[self.data[k][:,j],self.data[k][:,i+1]], bins,\
                            levels, smooth_2d, linestyles_2d[k], line_width_2d, colors[k],\
                            best_value_styles[k], best_value_colors[k], fill_contours=fill_contours,\
                            best_value=[best_values[k][j],best_values[k][i+1]],\
                            legend_label=legend_labels[k],legend=_legend,legend_colors=colors,ticks_size=ticks_size,legend_size=legend_size)
                            if show_best_value_lines is True:
                                self.__intersecting_line(ax, [best_values[k][j],best_values[k][i+1]],\
                                xy_lims, best_value_colors[k])
        fig.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=layout_adjust[0],hspace=layout_adjust[1])
        return fig

