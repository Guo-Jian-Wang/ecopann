# -*- coding: utf-8 -*-

from . import utils
from . import plot_settings as pls
import matplotlib.pyplot as plt
import numpy as np
import math


fiducial_colors = ['#006FED','#E03424','#33b540','#f68712','#1f77b4','#595657','m','#bdbcbc','#1f77b4','#ff7f0e']
fiducial_line_styles = ['-','--',':','-.','--']
markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']

def get_fiducial_colors(num):
    if num < len(fiducial_colors):
        fc = fiducial_colors[:num]
    else:
        fc = list( np.random.choice(fiducial_colors, num) )
    return fc

def get_fiducial_line_styles(num):
    if num < len(fiducial_line_styles):
        fl = fiducial_line_styles[:num]
    else:
        fl = list( np.random.choice(fiducial_line_styles, num) )
    return fl


def makeList(roots):
    """Checks if the given parameter is a list, if not, creates a list with the parameter as an item in it.
    
    Parameters
    ----------
    roots : object
        The parameter to be checked.

    Returns
    -------
    list
        A list containing the parameter.
    """
    if isinstance(roots, (list, tuple)):
        return roots
    else:
        return [roots]

def makeListList(roots):
    """Checks if the element of the given parameter is a list, if not, creates a list with the parameter as an item in it.
    
    Parameters
    ----------
    roots : object
        The parameter to be checked.

    Returns
    -------
    list
        A list containing the parameter.
    """
    if isinstance(roots[0], (list, tuple)):
        return roots
    else:
        return [roots]

#to be improved
class Plots:
    def __init__(self, set_figsize=True):
        self.set_figsize = set_figsize
    
    def plot_lines(self):
        pass
    
    def plot_dots(self):
        pass
    
    def plot_errbars(self):
        pass
    
    def plot_fillBetween(self):
        pass
    
    def plot_hist(self):
        pass
    
    def siglePlot(self, location=None,lims=None,labels=None,ticks_size=None,
                  major_locator_N=None,minor_locator=True,minor_locator_N=None,
                  lines=None,line_labels=None,line_styles=None,line_colors=None,line_width=None,
                  dots=None,dot_labels=None,dot_styles=None,dot_colors=None,
                  errbars=None,errbar_line_width=None,errbar_colors=None,xerr=True,yerr=True,errbars_fmt=None,errbar_labels=None,
                  fill_between=None,fill_between_line_width=None,fill_between_line_styles=None,fill_between_colors=None,fill_between_labels=None,fill_between_alphas=None,
                  hist=None,hist_bins=None,hist_colors=None,hist_alpha=None,hist_labels=None,
                  legend=False,legend_location=None,title=None,title_size=None):
        """Plot dots, lines, error bars, fill between, histogram, etc.
        
        Parameters
        ----------
        location : tuple or list, optional
            The location of the panel, location=(1,2,2) or [1,2,2].
        lims : list, optional
            The limits of X and Y axis: [min_x, max_x, min_y, max_y].
        labels : list, optional
            The labels of the panel. e.g. [r'$x$',r'$y$']
        ticks_size : int, optional
            The font size of ticks.
        major_locator_N : int, optional
            The number of major locators.
        minor_locator : bool, optional
            If True(False), show(don't show) the minor locators.
        minor_locator_N : int, optional
            The number of minor locators.
        
        Returns
        -------
        object
            fig or ax
        """
        if self.set_figsize:
            fig_rate = 1
            fig = plt.figure(figsize=(6*fig_rate, 4.5*fig_rate))
        
        if ticks_size is None:
            ticks_size = 12
        ax = pls.PlotSettings().setting(location=location,lims=lims,labels=labels,\
        ticks_size=ticks_size,major_locator_N=major_locator_N,\
        minor_locator=minor_locator,minor_locator_N=minor_locator_N)
                
        # plot dots
        if dots is not None:
            dots = makeList(dots)
            if dot_styles is None:
                dot_styles = ['.' for i in range(len(dots))]
            else:
                dot_styles = makeList(dot_styles)
            if dot_colors is None:
                dot_colors = fiducial_colors
            else:
                dot_colors = makeList(dot_colors)
            if dot_labels is None:
                dot_labels = ['dot' for i in range(len(dots))]
#                dot_labels = ['' for i in range(len(dots))]
            for i in range(len(dots)):
                ax.plot(dots[i][:,0],dots[i][:,1],dot_styles[i],color=dot_colors[i],\
                label=dot_labels[i])
        
        # plot lines
        if lines is not None:
            lines = makeList(lines)
            if line_colors is None:
                line_colors = fiducial_colors
            else:
                line_colors = makeList(line_colors)
            if line_styles is None:
                line_styles = fiducial_line_styles
            else:
                line_styles = makeList(line_styles)
            if line_width is None:
                line_width = 1.618
            if line_labels is None:
                line_labels = ['line' for i in range(len(lines))]
#                line_labels = ['' for i in range(len(lines))]
            for i in range(len(lines)):
                ax.plot(lines[i][:,0],lines[i][:,1],color=line_colors[i],\
                linestyle=line_styles[i],linewidth=line_width,label=line_labels[i])
        
        # plot error bars
        if errbars is not None:
            if errbars_fmt is None:
                errbars_fmt = ['.' for i in range(len(errbars))]
            if errbar_labels is None:
                errbar_labels = ['errbar' for i in range(len(errbars))]
#                errbar_labels = ['' for i in range(len(errbars))]
            
            errbars = makeList(errbars)
            if errbar_colors is None:
                errbar_colors = fiducial_colors
            else:
                errbar_colors = makeList(errbar_colors)
            if errbar_line_width is None:
                errbar_line_width = 1.618
            for i in range(len(errbars)):
                if xerr is False and yerr is True:
                    ax.errorbar(errbars[i][:,0],errbars[i][:,1],yerr=errbars[i][:,2],\
                    fmt=errbars_fmt[i], color=errbar_colors[i],label=errbar_labels[i])
                elif xerr is True and yerr is False:
                    ax.errorbar(errbars[i][:,0],errbars[i][:,1],xerr=errbars[i][:,2],\
                    fmt=errbars_fmt[i], color=errbar_colors[i],label=errbar_labels[i])
                elif xerr is True and yerr is True:
                    ax.errorbar(errbars[i][:,0],errbars[i][:,1],xerr=errbars[i][:,2],\
                    yerr=errbars[i][:,3],fmt=errbars_fmt[i],color=errbar_colors[i],\
                    linewidth=errbar_line_width,label=errbar_labels[i])
        
        # plot fill between
        if fill_between is not None:
            fill_between = makeList(fill_between)
            if fill_between_line_width is None:
                fill_between_line_width = 1.618
            if fill_between_line_styles is None:
                fill_between_line_styles = fiducial_line_styles
            else:
                fill_between_line_styles = makeList(fill_between_line_styles)
            if fill_between_colors is None:
                fill_between_colors = fiducial_colors
            else:
                fill_between_colors = makeList(fill_between_colors)
            if fill_between_alphas is None:
                fill_between_alphas = [0.3 for i in range(len(fill_between))]
            if fill_between_labels is None:
                fill_between_labels = ['fill between' for i in range(len(fill_between))]
#                fill_between_labels = ['' for i in range(len(fill_between))]
            for i in range(len(fill_between)):
                ax.fill_between(fill_between[i][:,0],fill_between[i][:,1]-fill_between[i][:,2],\
                fill_between[i][:,1]+fill_between[i][:,2],color=fill_between_colors[i],\
                alpha=fill_between_alphas[i],linewidth=0,label='fill between')
                ax.plot(fill_between[i][:,0],fill_between[i][:,1],color=fill_between_colors[i],\
                linestyle=fill_between_line_styles[i], linewidth=fill_between_line_width)
        
        # plot histogram
        if hist is not None:
            hist = makeList(hist)
            if hist_bins is None:
                hist_bins = 30
            if hist_colors is None:
                hist_colors = fiducial_colors
            else:
                hist_colors = makeList(hist_colors)
            if hist_alpha is None:
                hist_alpha = [1 for i in range(len(hist))]
            if hist_labels is None:
                hist_labels = ['histogram' for i in range(len(hist))]
#                hist_labels = ['' for i in range(len(hist))]
            for i in range(len(hist)):
                ax.hist(hist[i][:],bins=hist_bins,color=hist_colors[i],\
                alpha=hist_alpha[i],label=hist_labels[i])
        
        # add legend
        if legend_location is None:
            # legend_location = 'upper(lower) right(left or center)' or 'center right(left)'
            legend_location = 'upper right'
        
        legend_size = ticks_size
        if legend is True:
            ax.legend(loc=legend_location, fontsize=legend_size)
            
        # add title
        if title is not None:
            if title_size is None:
                title_size = ticks_size
            ax.set_title(label=title, fontsize=title_size)
        
        if self.set_figsize is True:
            return fig
        else:
            return ax

def savefig(path, fig_name, fig):
    if path:
        utils.mkdir(path)
        fig.savefig(path + '/' + fig_name, bbox_inches='tight')
    else:
        fig.savefig(fig_name, bbox_inches='tight')


#%% multiple panels
#to be improved
class MultiplePanels(object):
    """Plot a figure with multiple panels.
    
    Parameters
    ----------
    panel_model : object
        An instance that can provide datasets and panel model (it should contain two methods: 'panels_data' and 'panel').
    """
    def __init__(self, panel_model, lat_n=3):
        self.datasets = panel_model.panels_data()
        self.panel_model = panel_model
        self.panel_n = len(self.datasets)
        self._lat_n = lat_n
    
    @property
    def lat_n(self):
        """ The number of panels in latitude (or transverse) direction. """
        if self.panel_n<self._lat_n:
            return self.panel_n
        else:
            return self._lat_n
    
    @property
    def lon_n(self):
        """ The number of panels in longitude (or longitudinal) direction. """
        return int(math.ceil(self.panel_n/float(self.lat_n)))
    
    def plot(self, panel_size=(4, 3), layout_adjust=[0.3, 0.25], ticks_size=12):
        fig = plt.figure(figsize=(panel_size[0]*self.lat_n, panel_size[1]*self.lon_n))
        fig.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=layout_adjust[0],hspace=layout_adjust[1])
        for i in range(self.lon_n):
            for j in range(self.lat_n):
                if i*self.lat_n+j+1 > self.panel_n:
                    break
                data = self.datasets[i*self.lat_n+j]
                pls.PlotSettings().setting(location=[self.lon_n, self.lat_n, i*self.lat_n+j+1], lims=data['lims'], labels=data['labels'],
                                           ticks_size=ticks_size)
                self.panel_model.panel(data)
        return fig
