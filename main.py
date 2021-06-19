import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn import metrics
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt



def generate_fp_matrix(smiles):
    '''
    Generate Morgan`s circular fingerprint. 
    
    Args:
        smiles(np.array): Smiles data used to generate the fp matrix

    This family of fingerprints, better known as circular fingerprints, is built by
    applying the Morgan algorithm to a set of user-supplied atom invariants. 
    When generating Morgan fingerprints, the radius of the fingerprint can be changed
    to increase precision.
    '''
    morgan_matrix = np.zeros((1024))
    l=len(smiles)

    for i in range(l):
        try:
            compound = Chem.MolFromSmiles(smiles[i])
            fp = Chem.AllChem.GetMorganFingerprintAsBitVect(compound, 3, nBits = 1024)
            fp = fp.ToBitString()
            matrix_row = np.array ([int(x) for x in list(fp)])
            morgan_matrix = np.row_stack((morgan_matrix, matrix_row))

            if i%500==0:
                percentage = np.round(0.1*(i/1),1)
                print ('Calculating fingerprint', percentage,  '% done')

        except:
            print ('problem with index', i)
    morgan_matrix = np.delete(morgan_matrix, 0, axis = 0)

    print('\n')
    print('Morgan Matrix Dimension is', morgan_matrix.shape)

    return morgan_matrix

def generate_regressor():
    ''' 
    Define the regressor to be used in our model. 
    
    Here a Supporting Vectot Machine is chosen. 
    Hyperparameters were optimized in a different step through a grid search.

    TODO: Optimise hyperparameters locally
    '''

    return SVR(kernel='rbf', C = 3, gamma='scale', epsilon = 0.005, max_iter=-1)

if __name__ == "__main__":
    
    # TODO:
    # In order of priority for project success
    # Upload code to github
    # Create a single work space for project
    # Get matplotlib to produce results in the format Dr Hurst needs
    # Make plt non-blocking
    # Setup VS Code
    # Organise code in Object Oriented Programming fashion (OOP)
    # Handle range input via command linep
    ''' Load input data '''
    data = pd.read_csv('dummy_data.csv')
    
    ''' Split data into arrays '''
    smiles = np.array(data['SMILES'])
    ID = np.array(data['ID'])
    activity = np.array(data['pValue'])

    morgan_matrix = generate_fp_matrix(smiles)
    regressor = generate_regressor()

    from cross_validator import CrossValidator
    
    cv = CrossValidator(morgan_matrix=morgan_matrix, activity=activity)

    cv.start()

