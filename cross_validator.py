import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn import metrics



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

        self.predictions_cv = None

        self.fpr = None
        self.tpr = None
        self.thresholds = None
        # Start the run 
        self._start()

    def _start(self):
        print ('----------------------'+ 'loop '+str(self.itter)+ '----------------------')

        # Splits input data and assigns it to class variables
        self.train_features, self.test_features, self.train_targets, self.test_targets = train_test_split(
                                np.array(self.morgan),
                                np.array(self.activity),
                                test_size = 0.10,
                                random_state=self.itter,
                                shuffle=True)
        # Cross validate
        self.cv_scores = cross_validate(
                    self.regressor, 
                    self.train_features, 
                    self.train_targets,
                    cv=20, 
                    scoring=('r2', 'explained_variance'),
                    return_train_score=True)
        
        # Predict cross validation
        self.predictions_cv = cross_val_predict(self.regressor,
                                                self.train_features,
                                                self.train_targets, 
                                                cv=20)
        
        self.cv_errors = abs(self.predictions_cv - self.train_targets)

        self.cv_auc = self._calculate_auc(auc_thresh=6)

        self._report_score()
        self._report_errors()
        self._report_auc()
        
        self._save_diff()
        self._save_targets()

    def _report_auc(self):
        print('The AUC score for loop is: ' + str(self.cv_auc))
    
    def _report_score(self):        
        print('The CV variance for loop '+str(self.itter)+' is:', 
                self.cv_scores['test_explained_variance'].mean())
        print('The CV sigma variance for loop '+str(self.itter)+' is:', 
                self.cv_scores['test_explained_variance'].std() * 2)
        print('The CV q2 for loop '+str(self.itter)+' is:', 
                self.cv_scores['train_r2'].mean())
        print('The CV sigma q2 for loop '+str(self.itter)+' is:',
                self.cv_scores['train_r2'].std() * 2)

    def _report_errors(self):
        
        print ('Calculating CV Mean Absolute Error for loop '+str(self.itter))
        print('Mean CV-Absolute Error:', round(np.mean(self.cv_errors), 2), 'units.')
        print('Standard error:',  round(np.std(self.cv_errors), 2) )
        print ("Largest CV prediction error is", np.max(self.cv_errors))

    def _calculate_auc(self, auc_thresh):
        ''' 
        Calculate the AUC (Area Under The Curve) of 
        the ROC (Receiver Operating Characteristics...)

        ROC and AUC are reponses of binary hit based on a threshold.
        The results need to be converted to binary.
        '''      

        train_targets_bin = self._binary_convert(self.train_targets, auc_thresh)
        predictions_bin   = self._binary_convert(self.predictions_cv, auc_thresh)

        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(
                train_targets_bin,predictions_bin,pos_label=1)
        
        return  metrics.auc(self.fpr, self.tpr)

    def _save_diff(self):
        
        with open('predictions_'+str(self.itter)+'.csv', 'w') as file:
            file.write('morgan, pValue, predictions, errors\n')

            for i in range(len(self.predictions_cv)):
                file.write( str(self.morgan[i]) + "," +
                            str(self.train_targets[i]) + "," + 
                            str(self.predictions_cv[i]) + "," + 
                            str(abs(self.predictions_cv[i] - self.train_targets[i])) + "\n")
           

    def _save_targets(self):

        with open('test_targets'+str(self.itter)+'.csv', 'w') as file:
            file.write('morgan, test_features, pValue\n')
            
            for i in range(len(self.test_targets)):
                file.write(str(self.morgan[i]) + "," + 
					   str(self.test_features[i]) + ","+
					   str(self.test_targets[i]) + "\n")

            

    def _binary_convert(self, data, thresh):

        result = []
        for val in data:
            if val > thresh:
                result.append(1.0)
            else:
                result.append(0.0)
        
        return result
