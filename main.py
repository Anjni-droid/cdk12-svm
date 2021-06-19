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
data = pd.read_csv('dummy_data.csv')
smiles = np.array(data['SMILES'])
ID = np.array(data['ID'])
activity = np.array(data['pValue'])


# Generate Morgan`s circular fingerprint. This family of fingerprints,
#better known as circular fingerprints , is built by applying
#the Morgan algorithm to a set of user-supplied atom invariants. When
#generating Morgan fingerprints, the radius of the fingerprint can be changed to increase precision :
#
def generate_FP_matrix(smiles):
    morgan_matrix = np.zeros((1024))
    l=len(smiles)
#
    for i in range(l):
        try:
            compound = Chem.MolFromSmiles(smiles[i])
            fp = Chem.AllChem.GetMorganFingerprintAsBitVect(compound, 3, nBits = 1024)
            fp = fp.ToBitString()
            matrix_row = np.array ([int(x) for x in list(fp)])
            morgan_matrix = np.row_stack((morgan_matrix, matrix_row))
#
            if i%500==0:
                percentage = np.round(0.1*(i/1),1)
                print ('Calculating fingerprint', percentage,  '% done')
#
        except:
            print ('problem with index', i)
    morgan_matrix = np.delete(morgan_matrix, 0, axis = 0)
#
    print('\n')
    print('Morgan Matrix Dimension is', morgan_matrix.shape)
#
    return morgan_matrix
#
morgan_matrix_feature_cross = generate_FP_matrix(smiles)


#
#
def largest(arr,n):
    # Initialize maximum element
    max = arr[0]
    # Traverse array elements from second
    # and compare every element with
    # current max
    for i in range(1, n):
        if arr[i] > max:
            max = arr[i]
    return max

# Define the regressor to be used in our model. Here a Supporting Vectot Machine
#is chosen. Hyperparameters were optimized ina different step througha grid search.
#
regressor=SVR(kernel='rbf', C = 3, gamma='scale', epsilon = 0.005, max_iter=-1)
#

#-----------------DONE---------------------------------------------------------------------------------------------------------
# Performing the 10-fold cross validation and leav-one-out validation
#------------------------------------------------------------------------------------------------------------------------------
print('------------------------------------------------------------------------------------------')
print('CALCULATING CROSS-VALIDATED MODELS STATISTICS')
print('------------------------------------------------------------------------------------------')
#------------------------------------------------------------------------------------------------------------------------------
print ('Calculating 10-fold cross validation...')
#

# TODO:
# In order of priority for project success
# Upload code to github
# Create a single work space for project
# Get matplotlib to produce results in the format Dr Hurst needs
# Make plt non-blocking
# Setup VS Code
# Organise code in Object Oriented Programming fashion (OOP)
# Handle range input via command linep

print('\n')

for i in range(10):

    print ('----------------------'+ 'loop '+str(i)+ '----------------------')
    features_cross = np.array(morgan_matrix_feature_cross)
    targets_cross = np.array(activity)
	
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
