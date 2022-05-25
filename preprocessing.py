import pandas as pd
from sklearn.utils import resample
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

# convert treatment to num : mono = 0 combo=1


#%% filt NaN
def filtnan(tabmut,filtcoef=0.05): 
    #drop les colonnes ayant plus que (coeff * nb de ligne) de nan (0 très strigent, 1 pas du tout)
    # Autrement dit, garde les colonne avec un proportion de NA inférieur à filtcoef
    filt_nan = len(tabmut)*filtcoef
    return tabmut.loc[:,(tabmut.isna().sum() <= filt_nan)] 


def clean_NA_duplicate_input(df_X,filtcoef=0.05):
    N = df_X.shape[1]
    # print(df_X.shape)
    df_X = filtnan(df_X,filtcoef=filtcoef)
    
    # print("The raw_dataset contains {0} feature (MS) with proportion of NA > {2} over {1} features".format(N-df_X.shape[1], N,filtcoef)) #0 null values
    
    # Removing duplicates if there exist
    N_dupli = sum(df_X.T.duplicated(keep='first'))
    df_X = df_X.T.drop_duplicates(keep='first').T
    # print(df_X.shape)
    # print("The raw_dataset contains {} duplicates".format(N_dupli))

    # Number of samples in the dataset
    N = df_X.shape[0]
    
    # Creating the time and event columns
    time_column = 'ipfs'
    event_column = 'ipfsev'

    # Extracting the features
    features = np.setdiff1d(df_X.columns, [time_column, event_column, 'Row.names', 'Cohort'] ).tolist()
    
    X=df_X[features]
    
    # Iterative imputation
    ## ExtraTree
    imp_tree = IterativeImputer(random_state=0,estimator=ExtraTreesRegressor(n_estimators=10, random_state=0))

    X_tree= pd.DataFrame(imp_tree.fit_transform(X),columns=X.columns,index=X.index)
    df_X = pd.merge(X_tree,df_X[['Cohort',time_column,event_column]],left_index=True,right_index=True)
    return df_X,features



def upsampling(data,size_resample = 1000):
    Nipicol = data[data.Cohort == 'NIPICOL']

    # Separate majority and minority classes
    df_majority = Nipicol[Nipicol.ipfsev==0]
    df_minority = Nipicol[Nipicol.ipfsev==1]


    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=size_resample,    # to match majority class
                                     random_state=123) # reproducible results

    # Upsample majority class
    df_majority_upsampled = resample(df_majority, 
                                     replace=True,     # sample with replacement
                                     n_samples=size_resample,    # to match majority class
                                     random_state=123) # reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority_upsampled, df_minority_upsampled])



    # Create X/E train/test sets for upsample

    Ricki = data[data.Cohort == 'RICKI']

    X_train_upsampled = df_upsampled.drop(['Cohort','ipfs','ipfsev'],axis=1)
    E_train_upsampled = df_upsampled.ipfsev

    X_test_upsampled = Ricki.drop(['Cohort','ipfs','ipfsev'],axis=1)
    E_test_upsampled = Ricki.ipfsev
    
    return X_train_upsampled, E_train_upsampled, X_test_upsampled, E_test_upsampled 