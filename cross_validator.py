import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn import metrics

import matplotlib.pyplot as plt

''' all imports '''
import numpy as np
import pandas as pd
#
# Import rdkit stuff
#
import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
import numpy as np

#
# Import the machine learning tools
#
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn import metrics
from sklearn.metrics import explained_variance_score
#
#Import matplot for our figures
#
import matplotlib.pyplot as plt
#
# Load the dataframe from a .csv file, stores the smiles strings for our compound and their PIC50a ctivities
# Load the dataframe from a .csv file, stores the smiles strings for our compound and their PIC50a ctivities
#
''' end all imports '''


class CrossValidator:
    def __init__(self, morgan_matrix, activity, smiles, id, regressor, itter):
        
        self.morgan = morgan_matrix
        self.activity = activity
        self.smiles = smiles
        self.id = id
        self.regressor = regressor
        self.itter = itter
        
        # Yet to be assigned variables:
        self.train_features = None 
        self.test_features = None
        self.train_targets = None
        self.test_targets = None

        # Start the run 
        self._start()

    def _start(self):
        print ('----------------------'+ 'loop '+str(self.itter)+ '----------------------')

        # Splits input data and assigns it to class variables
        train_features, test_features, train_targets, test_targets = train_test_split(
                                np.array(self.morgan),
                                np.array(self.activity),
                                test_size = 0.10,
                                random_state=self.itter,
                                shuffle=True)

        self.train_features = train_features  
        self.test_features = test_features  
        self.train_targets = train_targets  
        self.test_targets = test_targets 

        
        scores = cross_validate(
                    self.regressor, 
                    self.train_features, 
                    self.train_targets,
                    cv=20, 
                    scoring=('r2', 'explained_variance'),
                    return_train_score=True)
        
        self.predictions_cv = cross_val_predict(self.regressor,
                                                self.train_features,
                                                self.train_targets, 
                                                cv=20)

        self._report_cv(scores)
        self._report_diff()

        self._save_diff()
        self._save_targets()

    def _report_cv(self, scores):        
        print('The CV variance for loop '+str(self.itter)+' is:', 
                scores['test_explained_variance'].mean())
        print('The CV sigma variance for loop '+str(self.itter)+' is:', 
                scores['test_explained_variance'].std() * 2)
        print('The CV q2 for loop '+str(self.itter)+' is:', 
                scores['train_r2'].mean())
        print('The CV sigma q2 for loop '+str(self.itter)+' is:',
                scores['train_r2'].std() * 2)

    def _report_diff(self):
        
        errors_cv = abs(self.predictions_cv - self.train_targets)

        print ('Calculating CV Mean Absolute Error for loop '+str(self.itter))
        print('Mean CV-Absolute Error:', round(np.mean(errors_cv), 2), 'units.')
        print('Standard error:',  round(np.std(errors_cv), 2) )
        print ("Largest CV prediction error is", np.max(errors_cv))

    def _save_diff(self):
        
        with open('predictions_'+str(self.itter)+'.csv', 'w') as file:
            file.write('train_targets, predictions, errors\n')

            for i in range(len(self.predictions_cv)):
                file.write( str(self.train_targets[i]) + "," + 
                            str(self.predictions_cv[i]) + "," + 
                            str(abs(self.predictions_cv[i] - self.train_targets[i])))
            file.write(' \n')

    def _save_targets(self):

        with open('test_targets'+str(self.itter)+'.csv', 'w') as file:
            file.write('SMILES, ID, test_features, test_targets\n')
            
            for i in range(len(self.test_targets)):
                file.write(str(self.smiles[i]) + "," +
					   str(self.id[i]) + "," +
					   str(self.test_features[i]) + ","+
					   str(self.test_targets[i]))

            file.write(' \n')

    def _binary_convert(self, data, thresh):

        result = []
        for val in data:
            if val > thresh:
                result.appent(1.0)
            else:
                result.append(0.0)


    def _record_aoc_curve(self):
        ''' 
        Calculate the AUC (Area Under The Curve) of 
        the ROC (Receiver Operating Characteristics...)

        ROC and AUC are reponses of binary hit based on a threshold.
        The results need to be converted to binary.
        '''
        
        roc_auc_thresh = 6
        

        train_targets_bin = self._binary_convert(self.train_argets, roc_auc_thresh)
        predictions_bin   = self._binary_convert(self.predictions_cv, roc_auc_thresh)

        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(
                train_targets_bin,predictions_bin,pos_label=1)
        



    