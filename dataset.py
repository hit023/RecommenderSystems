import pandas as pd
import numpy as np

def get_dataset():
    """
        Return:
        ratings(user X items) : all the ratings given by users to movies
    """
    folder = "./ml-100k/"
    fields = ['user_id', 'item_id', 'rating', 'timestamp']
    training = folder + 'u3.base'

    users = 943
    items = 1682

    #r[i][j] : rating given by user 'i' to movie 'j'
    ratings = np.zeros((users,items))

    #read the dataset as a tab-separated file
    dtset = pd.read_csv(training,sep = '\t',names = fields)

    #iterate over all the entries and fill the ratings matrix
    for row in dtset.itertuples():
        ratings[row[1]-1,row[2]-1] = row[3]
    return ratings
