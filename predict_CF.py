import pandas as pd
import numpy as np
import math
from operator import itemgetter
from dataset import get_dataset
from utils import get_means,get_sim,rmse

def predict_neighbours(user,movie,ratings,movie_sim,k=15):
    """
        Parameter(s):
        user : the user whose rating to the movie has to be predicted
        movie : the movie
        k : number of neighbours to consider

        Return:
        estimated_rating : the rating estimated by taking a weighted average of neighbours

        Note: implements only item-item C.F.
    """
    #first, get all nonzero ratings
    non_zero = ratings[user].nonzero()[0]

    #to get the index of the nearest ones, use argsort.
    neighbours = non_zero[movie_sim[movie,non_zero].argsort()[::-1][:k]]

    #prediction : (ratings*similarity)/(sum of similarities)
    estimated_rating = \
    (ratings[user,neighbours]).dot(movie_sim[movie,neighbours])/sum(movie_sim[movie,neighbours])

    #clipping values that go beyond the range.
    if estimated_rating>5:
        estimated_rating = 5
    elif estimated_rating<1:
        estimated_rating = 1
    return estimated_rating

def predict_neighbours_baseline(user,movie,ratings,movie_mean,user_mean,mean,movie_sim,k=15):
    """
        Description:
        Finds the weighted average of the ratings with respect to teh similarities of the movies. The calculation of the final rating can be done using the equation given in the design doc.

        Parameter(s):
        user : the user whose rating to the movie has to be predicted
        movie : the movie
        k : number of neighbours to consider

        Return:
        estimated_rating : the rating estimated by taking a weighted average of neighbours

        Note: implements only item-item C.F.
        baseline = (bias-movie) + (bias-user) + all_mean
                 = (movie_mean - all_mean) + (user_mean - all_mean) + all_mean
                 = movie_mean + user_mean - all_mean
    """
    #get baseline prediction using the formula shown above
    baseline = user_mean[user] + movie_mean - mean

    non_zero = ratings[user].nonzero()[0]

    #to get the index of the nearest ones, use argsort.
    neighbours = non_zero[movie_sim[movie,non_zero].argsort()[::-1][:k]]

    #prediction : (ratings*similarity)/(sum of similarities)
    estimated_rating = (ratings[user,neighbours] -\
    baseline[neighbours]).dot(movie_sim[movie,neighbours])/sum(movie_sim[movie,neighbours]) + \
    baseline[movie]
    if estimated_rating>5:
        estimated_rating = 5
    elif estimated_rating<1:
        estimated_rating = 1
    return estimated_rating

def test_basic():
    """
        Description:
        Predict using collaborative filtering without baseline model; using nearest neighbours only.

        Return:
        pred : prediction of ratings using only nearest neighbours
        pred_base : prediction of ratings using baseline along with basic C.F.
        y : actual predictions
        users_items : users corresponding to the predictions made; useful for p@k
    """
    pred = []
    y = []
    users = []
    items = []
    users_items = []
    folder = "./ml-100k/"
    testing = folder + 'u3.test'
    fields = ['user_id', 'item_id', 'rating', 'timestamp']
    #fetch the dataset
    dtset = pd.read_csv(testing,sep = '\t',names = fields)
    count = 0
    um = {}
    #save all the values like mean, similarities etc in local variables to avoid redundant calling of the same functions.
    ratings = get_dataset()
    movie_mean,user_mean,mean = get_means(ratings)
    movie_sim = get_sim(ratings,f="movies")
    for row in dtset.itertuples():
        pred.append(predict_neighbours(row[1]-1,row[2]-1,ratings,movie_sim))
        y.append(row[3]-1)
        users.append(row[1]-1)
        items.append(row[2]-1)
    final_pred = []
    final_y = []
    for i in range(len(pred)):
        #to avoid getting 'nan' values for the final evaluation results, select only those that are not 'nan'.
        if not (math.isnan(y[i]) or y[i]==0 or math.isnan(pred[i])):
            final_pred.append(pred[i])
            final_y.append(y[i])
            users_items.append([users[i],items[i]])
    return final_pred,final_y,users_items

def test_base():
    """
        Description:
        Predict using collaborative filtering using baseline model along with nearest neighbours.

        Return:
        pred : prediction of ratings using only nearest neighbours
        pred_base : prediction of ratings using baseline along with basic C.F.
        y : actual predictions
        users_items : users corresponding to the predictions made; useful for p@k
    """
    pred_base = []
    y = []
    users = []
    items = []
    users_items = []
    folder = "./ml-100k/"
    testing = folder + 'u3.test'
    fields = ['user_id', 'item_id', 'rating', 'timestamp']
    dtset = pd.read_csv(testing,sep = '\t',names = fields)
    count = 0
    um = {}
    ratings = get_dataset()
    movie_mean,user_mean,mean = get_means(ratings)
    movie_sim = get_sim(ratings,f="movies")
    for row in dtset.itertuples():
        pred_base.append(predict_neighbours_baseline(row[1]-1,row[2]-1,ratings,movie_mean,user_mean,mean,movie_sim))
        y.append(row[3]-1)
        users.append(row[1]-1)
        items.append(row[2]-1)
    final_pred_base = []
    final_y = []
    for i in range(len(pred_base)):
        if not (math.isnan(y[i]) or y[i]==0 or math.isnan(pred_base[i])):
            final_pred_base.append(pred_base[i])
            final_y.append(y[i])
            users_items.append([users[i],items[i]])
    return final_pred_base,final_y,users_items

def evaluate(pred,y,users_items,k):
    """
        Parameter(s):
        pred : predicted ratings
        y : actual ratings given
        users_items : users for whom ratings have been predicted
        k : value for p@k

        Return:
        rmse : Root-mean squared error
        p@k : precision at top-k
        sman : Spearman's Correlation; rho = 1-6*((_sigma_)(di)**2)/n(n^2-1)

        Note:
        relevant items : items with actual rating greater or equal to 3.5.
        Recommended item: has a predicted rating >= 3.5
        p@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
    """
    #find out the root-mean-squared-error
    re = rmse(pred,y)

    #find out p@k
    pk = {}
    p={}
    out = set()
    for i in users_items:
        out.add(i[0])
    for i in out:
        p[i] = []

    for i in range(len(users_items)):
        p[users_items[i][0]].append([users_items[i][1],pred[i],y[i]])
    for i in out:
        p[i] = sorted(p[i],key=itemgetter(1))
        #collect relevant and recommended movies using the criterion described in the design desc.
        relevant = []
        for j in p[i]:
            if j[2] > 3.5:
                relevant.append(j[0])
        recommended = p[i][-k:]
        relevant_count = 0
        recommended_count = 0
        for j in recommended:
            if j[0] in relevant:
                relevant_count += 1
            if j[1] > 3.5:
                recommended_count += 1
        if recommended_count!=0:
            l = relevant_count/recommended_count
            pk[i] = l

    avg_pk = 0
    for k in pk.keys():
        avg_pk += pk[k]
    avg_pk = avg_pk/len(pk)

    #calculate spearman correlation
    n = len(pred)
    square = np.sum(np.square( np.abs( np.asarray(pred) - np.asarray(y) ) ))
    sman = 1-(6*square)/(n*(n**2-1))

    return re,avg_pk,sman

if __name__ == "__main__":
    pred_basic,y,users_items = test_basic()
    rms,pk,sman = evaluate(pred_basic,y,users_items,10)
    print("For classical C.F. prediction: ")
    print("rmse: {0}, pk: {1}, sman: {2}".format(rms,pk,sman))
    pred_base,y,users_items = test_base()
    rms,pk,sman = evaluate(pred_base,y,users_items,10)
    print("For C.F. prediction with baseline: ")
    print("rmse: {0}, pk: {1}, sman: {2}".format(rms,pk,sman))
