import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

class PlotHandler:
    def __init__(self, cv_collection):
        ''' Handles the plotting of the Cross Validation results 
        
        Args:
            cv_collection(list of CrossValidator instances): used for plotting input
        '''

        self.cv_collection = cv_collection
    
    def plot_all(self):
        ''' Plots the next cv '''

        for itter, cv in enumerate(self.cv_collection):
            self._plot(cv, itter)
    
    def _plot(self, cv, itter):
        ''' Plot a graph of our cross-validated predictions vs actual values.
        
        Args:
            cv(instance of CrossValidator): a single instance of cv to be plotted
        '''

        x = np.arange(len(cv.predictions_cv))

        plt.plot(x, cv.train_targets, 
                marker='', color='blue', linewidth=1.0, linestyle='dashed',label="actual")
        
        plt.plot(x, cv.predictions_cv, 
                marker='', color='red', linewidth=1.0, label='predicted')
        
        plt.savefig('predict'+str(cv.itter)+'.png', dpi=600)
        
        plt.legend()

        plt.show()