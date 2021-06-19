import numpy as np


class CrossValidator:
    def __init__(self, morgan_matrix, activity):
        
        self.morgan = morgan_matrix
        self.activity = activity


    def start(self):
        for i in range(10):    
            print ('----------------------'+ 'loop '+str(i)+ '----------------------')
            features_cross = np.array(self.morgan_matrix)
            targets_cross = np.array(self.activity)
            ''' TODO:Update the rest under this line, while introducing vars from qsar '''
            
            train_features, test_features, train_targets, test_targets = train_test_split(features_cross, targets_cross, test_size = 0.10, random_state=i, shuffle=True)

            scores = cross_validate(regressor, train_features, train_targets, cv=20, scoring=('r2', 'explained_variance'), return_train_score=True)

            print('The cross-validated variance for loop '+str(i)+' is:', scores['test_explained_variance'].mean())
            print('The cross-validated sigma variance for loop '+str(i)+' is:', scores['test_explained_variance'].std() * 2)

            print('The cross-validated q2 for loop '+str(i)+' is:', scores['train_r2'].mean())
            print('The cross-validated sigma q2 for loop '+str(i)+' is:',scores['train_r2'].std() * 2)

            predictions_cv=cross_val_predict(regressor,train_features, train_targets, cv=20)

    ### difference between experimental and predicted
            errors_cv = abs(predictions_cv - train_targets)

    ### Save experimental vs predicted data
            with open('predictions_'+str(i)+'.csv') as the_file:
                l=len(predictions_cv)
                the_file.write('train_targets, predictions, errors\n')
                for ii in range(l):
                    the_file.write( str(train_targets[ii]) + "," + str(predictions_cv[ii]) + "," + str(abs(predictions_cv[ii] - train_targets[ii])))
                    the_file.write(' \n')

    ### Save test targets
            with open('test_targets'+str(i)+'.csv') as the_file:
                l=len(test_targets)
                the_file.write('SMILES, ID, test_features, test_targets\n')
                for ii in range(l):
                    the_file.write(str(smiles[ii]) + "," +
                            str(ID[ii]) + "," +
                            str(test_features[ii]) + "," +
                            str(test_targets[ii]))
                    the_file.write(' \n')


            print ('Calculating Cross-Validated Mean Absolute Error...for loop '+str(i))
            print('Mean CV-Absolute Error:', round(np.mean(errors_cv), 2), 'units.',  'Standard error:',  round(np.std(errors_cv), 2) )

            arr_cv=errors_cv
            n_cv=len(arr_cv)
            Ans_cv = largest(arr_cv,n_cv)

            print ("Largest cross-validated prediction error is",Ans_cv)

    #---------------DONE----------------------------------------------------------------------------------------------------------
    # Plotting a a  graph now of our cross-validated predictions vs actual values.
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


    