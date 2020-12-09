# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoMinorLocator


#to be improved
class PlotSettings:
    def __init__(self):
        #move parameters to here
        pass
    
    def set_label(self):
        pass
    
    def set_locator(self):
        pass
    
    def set_size(self):
        pass
    
    def set_tick(self):
        pass
    
    def setting(self, ax=None, location=None,lims=None,set_labels=True,labels=None,ticks_size=None,
                rotation=None,major_locator_N=None,major_locator_length=None,
                minor_locator=True,minor_locator_N=None,show_xticks=True,show_yticks=True,
                show_xticklabels=True,show_yticklabels=True,show_xlabel=True,
                show_ylabel=True, old_ticksStyle=False,rightTopTicks=True):
        ''' 
        set the attribute of panels 
        
        :param lims: [min_x, max_x, min_y, max_y], where min_* or max_* is a real number or None
        :param labels: the label of a panel
        :param ticks_size: the font size of ticks
        :param rotation: the rotation of the label of ticks
        :param major_locator_N: the number of major locators
        :param minor_locator: if the minor_locator is True(False), show(don't show) the minor locators
        :param minor_locator_N: the number of minor locators
        :param show_xticks: if show_xticks is True(False), show(don't show) ticks of X axis
        :param show_yticks: if show_yticks is True(False), show(don't show) ticks of Y axis
        :param show_xticklabels: if True(False), show(don't show) the labels of ticks of X axis
        :param show_yticklabels: if True(False), show(don't show) the labels of ticks of Y axis
        :param show_xlabel: if True(False), show(don't show) the labels of X axis
        :param show_ylabel: if True(False), show(don't show) the labels of Y axis
        '''
        
        if location is None:
            location = (1,1,1)
        if ax is None:
            ax = plt.subplot(location[0],location[1],location[2])
        
        if lims is not None:
            min_x, max_x, min_y, max_y = lims[0], lims[1], lims[2], lims[3]
            if min_x is not None and max_x is not None:
                ax.set_xlim(min_x, max_x)
            if min_y is not None and max_y is not None:
                ax.set_ylim(min_y, max_y)
        
        # set ticks
        if major_locator_length is None:
            major_locator_length = 5.5
        minor_locator_length = major_locator_length*0.5 #618
        ax.tick_params(which='major', length=major_locator_length, width= 0.5, color='k')
        ax.tick_params(which='minor', length=minor_locator_length, width= 0.5, color='k')
        
        if major_locator_N is None:
            major_locator_N = 6.0
        if minor_locator_N is None:
            minor_locator_N = 10.0
        
        if old_ticksStyle:
            if rightTopTicks:
                ax.axes.tick_params(axis='both',which='both',direction='in',right=True,top=True)
            else:
                ax.axes.tick_params(axis='both',which='both',direction='in',right=False,top=False)
        
        if show_xticks is True:
            if minor_locator is True:
                ax.xaxis.set_major_locator( MaxNLocator(major_locator_N) )
                ax.xaxis.set_minor_locator( AutoMinorLocator(minor_locator_N) )
            else:
                ax.xaxis.set_major_locator( MaxNLocator(major_locator_N) )
        else:
            ax.set_xticks([])
        if show_yticks is True:
            if minor_locator is True:
                ax.yaxis.set_major_locator( MaxNLocator(major_locator_N) )
                ax.yaxis.set_minor_locator( AutoMinorLocator(minor_locator_N) )
            else:
                ax.yaxis.set_major_locator( MaxNLocator(major_locator_N) )
        else:
            ax.set_yticks([])
        
        # set ticks label
        if ticks_size is None:
            ticks_size = 12
        if rotation is None:
            rotation = 0
        if show_xticklabels is True:
            ax.tick_params(axis='both', which='major', labelsize=ticks_size)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_rotation(rotation)
        else:
            ax.set_xticklabels([])
        if show_yticklabels is True:
            ax.tick_params(axis='both', which='major', labelsize=ticks_size)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_rotation(rotation)
        else:
            ax.set_yticklabels([])

        # set labels
        if set_labels:
            labels_size = ticks_size/0.618
            if labels is None:
                labels = [r'$x$',r'$y$']
            if show_xlabel is True:
                ax.set_xlabel(r'%s'%labels[0], fontsize=labels_size)
            if show_ylabel is True:
                ax.set_ylabel(r'%s'%labels[1], fontsize=labels_size)
        return ax

