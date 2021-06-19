import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate

import matplotlib.pyplot as plt
class CrossValidator:
    def __init__(self, morgan_matrix, activity, smiles, id):
        
        self.morgan = morgan_matrix
        self.activity = activity
        self.smiles = smiles
        self.id = id

    def start(self, loop_count):
        for i in range(loop_count):    
            self._loop(itter = i)

def _test_split(self, itter):

    return train_test_split(self.morgan,
                            self.activity,
                            test_size = 0.10,
                            random_state=itter,
                            shuffle=True)

def _report_cv(self, features, targets, itter):
    
    scores = cross_validate(self.regressor, 
                   features, targets,
                   cv=20, scoring=('r2', 'explained_variance'),
                   return_train_score=True)
    
    print('The CV variance for loop '+str(itter)+' is:', 
            scores['test_explained_variance'].mean())
    print('The CV sigma variance for loop '+str(itter)+' is:', 
            scores['test_explained_variance'].std() * 2)
    print('The CV q2 for loop '+str(itter)+' is:', 
            scores['train_r2'].mean())
    print('The CV sigma q2 for loop '+str(itter)+' is:',
            scores['train_r2'].std() * 2)

def _report_diff(self, predictions, targets, itter):
    
    errors_cv = abs(predictions - targets)
    print ('Calculating CV Mean Absolute Error for loop '+str(itter))
    print('Mean CV-Absolute Error:', round(np.mean(errors_cv), 2), 'units.')
    print('Standard error:',  round(np.std(errors_cv), 2) )
    print ("Largest CV prediction error is", np.max(errors_cv))

def _save_diff(self, predictions_cv, train_targets, itter):
    
    with open('predictions_'+str(itter)+'.csv') as file:
        file.write('train_targets, predictions, errors\n')

        for i in range(len(predictions_cv)):
            file.write( str(train_targets[i]) + "," +
                            str(predictions_cv[i]) + "," +
                            str(abs(predictions_cv[i] - train_targets[i])) + 
                            "\n")

def _save_targets(self, test_targets, test_features, itter):

    with open('test_targets'+str(itter)+'.csv') as file:
        file.write('SMILES, ID, test_features, test_targets\n')
        
        for i in range(len(test_targets)):
            file.write(str(self.smiles[i]) + "," +
                    str(self.id[i]) + "," +
                    str(test_features[i]) + "," +
                    str(test_targets[i]))
            file.write(' \n')

def _loop(self, itter):
    print ('----------------------'+ 'loop '+str(itter)+ '----------------------')

    # TODO: Extract to class wide variables and itterate over CV externally, 
    #       instead of internally.

    train_features, test_features, train_targets, test_targets = self._test_split(itter)

    predictions_cv = cross_val_predict(self.regressor,train_features, train_targets, cv=20)

    self._report_cv(train_features, train_targets, itter)
    self._report_diff(predictions_cv, train_targets, itter)

    self._save_diff(predictions_cv, train_targets, itter)
    self._save_targets(test_targets, test_features, itter)

    # TODO: Parts under are not sorted
#---------------DONE----------------------------------------------------------------------------------------------------------
# Plot a a  graph of our cross-validated predictions vs actual values.
#-----------------------------------------------------------------------------------------------------------------------------

    x = np.arange(len(predictions_cv))

    plt.plot( x, train_targets,    marker='', color='blue', linewidth=1.0, linestyle='dashed', label="actual")
    plt.plot( x, predictions_cv, marker='', color='red', linewidth=1.0, label='predicted')
    
    
    plt.savefig('predict'+str(i)+'.png', dpi=600)
    
    plt.legend()

    print('------------------------------------------------------------------------------------------')
    print('Calculating AUC (Area Under The Curve) of the ROC (Receiver Operating Characteristics...)')
    print('------------------------------------------------------------------------------------------')
#---------------DONE----------------------------------------------------------------------------------------------------------
# ROC and AUC are reponses of binary hit based on a threshold (6 in our case), we need to convert out data to ones and zeros.
#-----------------------------------------------------------------------------------------------------------------------------

    for ii, item in enumerate(train_targets):
        if item > 6:
            train_targets[ii] = 1.0
        else:
            train_targets[ii] = 0.0
    
    for ii, item in enumerate(predictions_cv):
        if item > 6:
            predictions_cv[ii] = 1.0
        else:
            predictions_cv[ii] = 0.0

    fpr, tpr, thresholds = metrics.roc_curve(train_targets, predictions_cv, pos_label=1)
        
    print ('The AUC score for loop '+str(i)+'is: ' + str(metrics.auc(fpr, tpr)))
    

    print ('----------------------')
    print ('\n')
    plt.show()


    