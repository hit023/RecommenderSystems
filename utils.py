import pandas as pd
import numpy as np

def rmse(pred,y):
    """
        Parameter(s):
        pred: The prediction made by the model
        y: The actual value/rating given by the user

        Return:
        err : The root-mean-squered error
    """
    err = 0
    count = 0
    for i,j in zip(pred,y):
        count += 1
        diff = abs(i-j)
        err += (diff * diff)
    err = np.sqrt(err/count)
    return err

def get_means(ratings):
    """
        Parameter(s):
        ratings(user X items) : matrix containing all the ratings

        Return:
        movie_mean(row-vector) : mean of all movie ratings calculated separately for each movie.
        user_mean(row-vector) : mean of all the ratings given by each user
        mean : mean of all the ratngs in the matrix

        Note:
        if x,y are 1-d vectors, np.where(condition,x,y) works like:
            [xv if c else yv for (c,xv,yv) in zip(condition,x,y)]
    """
    #find the required means.
    mean = np.mean(ratings[ratings!=0])
    movie_mean = sum(ratings)/sum(ratings!=0)
    user_mean = sum(ratings.T)/sum(ratings.T!=0)

    #replace NaN values with the 'mean' rating
    movie_mean = np.where(np.isnan(movie_mean),mean,movie_mean)
    user_mean = np.where(np.isnan(user_mean),mean,user_mean)

    return movie_mean,user_mean,mean

def get_sim(ratings,f="users"):
    """
        Parameters(s):
        ratings(user X items) : matrix containing all the ratings
        f : selects whether to find nearest items or movies.

        Return:
        The normalized similarities of items/movies using Pearson correlation coefficient.
    """
    if f == 'users':
        r_copy = ratings.copy()
        _,user_mean,_ = get_means(ratings)
        for i in range(ratings.shape[0]):
            #consider only non-zero entries in the ratings matrix
            non_zero = ratings[i].nonzero()
            r_copy[i][non_zero] = ratings[i][non_zero] - user_mean[i]
        similarity = r_copy.dot(r_copy.T)
    elif f == 'movies':
        r_copy = ratings.copy()
        movie_mean,_,_ = get_means(ratings)
        for i in range(ratings.shape[1]):
            #consider only non-zero entries in the ratings matrix
            non_zero = ratings[:,i].nonzero()
            r_copy[:,i][non_zero] = ratings[:,i][non_zero] - movie_mean[i]
        similarity = r_copy.T.dot(r_copy)
    norm = np.array([np.sqrt(np.diagonal(similarity))])
    return (similarity / norm / norm.T)
