import csv
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split

# Get keys from names file
def getKeysfromNames(filename):
    try:
        if filename[-6:]!=".names": raise NameError('filename must end with ".names"')
        keys = []
        d={}
        with open(filename, 'r') as file_in:
                    reader = csv.reader(file_in, delimiter=':')
                    for line in reader:
                        if len(line)>1 and not ' ignore.' in line:
                            key = line[0]
                            keys.append(key)
                            d[key] = line[1].split(',')
        return keys, d
    except NameError: print ("filename was incorrect"); raise

def subsampling(data, target, split):
    idx = int(split*len(target))
    return data.iloc[:idx, :], target.iloc[:idx]

def load_data(folderpath, filename=None, split=0.33):
    # Load train and test data and generate X,y.
    # If filename is not specified it tries to load all files, conatenate them and perform train-test split
    if filename:
        X_train = pd.read_csv(folderpath + filename + '.csv', header=None)
        print("Train Data Loaded")
        if os.path.isfile(folderpath + 'test_' + filename + '.csv'):
            X_test = pd.read_csv(folderpath + 'test_' + filename + '.csv', header=0)
            y_train = X_train.iloc[:, 0].astype(int)
            X_train.drop(X_train.columns[0], axis=1, inplace=True)
            y_test = X_test.iloc[:, 0].astype(int)
            X_test.drop(X_test.columns[0], axis=1, inplace=True)
            print("Test Data Loaded")
        else:
            print(".test file doesn't exist. Performing Train Test Split")
            Y = X_train.iloc[:, 0].astype(int)
            X_train.drop(0, axis=1, inplace=True)
            X_train, X_test, y_train, y_test = train_test_split(X_train, Y, test_size=split, random_state=42)
            print("Data Splitted")
        if os.path.isfile(folderpath + filename + '.names'):
            keys, d = getKeysfromNames(folderpath + filename + '.names')
            X_train.columns = X_test.columns = keys[:-1]
            print("keys loaded !")
        else:
            print("keys not loaded, because .names file doesn't exist")
    else:
        print("Load all files from folder and concatenate")
        filenames = os.listdir(folderpath)
        print("Found {} files".format(len(filenames)))
        dfs = []
        i = 1
        for filename in filenames:
            dfs.append(pd.read_csv(folderpath + filename, header=None))
            print("loaded {} out of {} files".format(i, len(filenames)), end="\r")
            i += 1
        # Concatenate all data into one DataFrame
        X = pd.concat(dfs, ignore_index=True)
        Y = X.iloc[:, 0]
        X.drop(X.columns[0], axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split, random_state=42)
        print("Data Splitted in Train and Test")

        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        y_train = y_train.fillna(0).astype(int)
        y_test = y_test.fillna(0).astype(int)
        print("Filled NaN with zeros")

    return X_train, X_test, y_train, y_test
