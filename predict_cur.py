import pandas as pd
from utils import rmse
from operator import itemgetter
from dataset import get_dataset
from scipy.linalg import norm
import numpy as np
import math
from predict_svd import svd

def get_prob():
    """
        Parameter(s):

        Return:
        prow,pcol : the probabilities of the corresponding rows and cols using the Frobenius
                    norm.
    """
    ratings = get_dataset()
    mat_sum = np.sum(ratings**2)
    prow = []
    pcol = []
    for i in range(ratings.shape[0]):
        tp = ratings[i,:]
        prow.append(np.sum(tp**2)/mat_sum)
    for i in range(ratings.shape[1]):
        tp = ratings[:,i]
        pcol.append(np.sum(tp**2)/mat_sum)
    return prow,pcol

def cur(k,no_dup):
    """
        Parameter(s):
        k : The number of columns and rows to sample based on the probability distribution.
        no_dup : 1 if duplications in the row and column matrices have to be avoided.

        Return:
        C,U,R : The factors obtained after CUR decomposition.

        Note:
        Replaces the missing values in the ratings matrix with the mean of the corresponding column.
    """
    ratings = get_dataset()
    prow,pcol = get_prob()
    #sample rows and columns based on their probabilities.
    rows_sampled = np.random.choice(len(prow),k,p=prow)
    cols_sampled = np.random.choice(len(pcol),k,p=pcol)
    #resample in case of suplication
    if no_dup==1:
        rows = set()
        for k in rows_sampled:
            rows.add(k)
        cols = set()
        for k in cols_sampled:
            cols.add(k)
        while len(rows)<k:
            f=1
            repl = np.random.choice(len(prow),1,p=prow)[0]
            while f==1:
                if repl in rows:
                    repl = np.random.choice(len(prow),1,p=prow)[0]
                else:
                    f=0
            rows.add(repl)
        while len(cols) < k:
            f=1
            repl = np.random.choice(len(pcol),1,p=pcol)[0]
            while f==1:
                if repl in cols:
                    repl = np.random.choice(len(pcol),1,p=pcol)[0]
                else:
                    f=0
            cols.add(repl)
        rows_sampled = np.asarray(list(rows))
        cols_sampled = np.asarray(list(cols))
    r = np.zeros((k,len(pcol)))
    c = np.zeros((len(prow),k))
    for i in range(k):
        #find the denominator which acts like a scaling value.
        denom = math.sqrt(k*prow[rows_sampled[i]])
        r_mean = sum(ratings[rows_sampled[i],:])/sum(ratings[rows_sampled[i],:]!=0)
        for j in range(k):
            if ratings[rows_sampled[i],j]==0:
                ratings[rows_sampled[i],j] = r_mean
        r[i,:] = ratings[rows_sampled[i],:]/denom
        denom = math.sqrt(k*pcol[cols_sampled[i]])
        c_mean = sum(ratings[:,cols_sampled[i]])/sum(ratings[:,cols_sampled[i]]!=0)
        for j in range(k):
            if ratings[j,cols_sampled[i]]==0:
                ratings[j,cols_sampled[i]] = c_mean
        c[:,i] = ratings[:,cols_sampled[i]]/denom
    w = np.zeros((k,k))
    #find the intersection of C and R
    for i in range(k):
        for j in range(k):
            w[i,j] = ratings[rows_sampled[i],cols_sampled[j]]
    u,s,vt = svd(w,0)
    v = vt.T
    #find the reciprocal of the singular values to yield the pseudoinverse.
    for i in range(s.shape[0]):
        if s[i,i]!=0:
            s[i,i] = 1/s[i,i]
    U = np.dot(v,s)
    U = np.dot(U,u.T)
    return c,U,r

def predict_cur(k,no_dup=0):
    """
        Description:
        Predicts movie ratings for users from the test set by reconstructing the ratings matrix.

        Parameter(s):
        k : The number of columns and rows to sample based on the probability distribution.
        no_dup : 1 if duplications in the row and column matrices have to be avoided.

        Return:
        reconstruct: The reconstructed ratings matrix
        pred : A list of lists with each list of the form : [userid, prediction,itemid]
        y : The actual ratings given by the users in the test set
    """
    c,u,r = cur(k,no_dup)
    folder = "./ml-100k/"
    testing = folder + 'u3.test'
    fields = ['user_id', 'item_id', 'rating', 'timestamp']
    dtset = pd.read_csv(testing,sep = '\t',names = fields)
    #reconstruct the ratings matrix to predict.
    reconstruct = np.dot(c,np.dot(u,r))
    users = {}
    for row in dtset.itertuples():
        if (row[1]-1) not in users.keys():
            users[row[1]-1] = []
        users[row[1]-1].append([row[2]-1,row[3]])
    pred = []
    y = []
    for user in users.keys():
        #prod = np.dot(U[user,:],right_term)
        for i in users[user]:
            pred.append([user,reconstruct[user,i[0]],i[0]])
            y.append(i[1])
    return reconstruct,pred,y

def evaluate(pred,y,k):
    """
        Description:
        Evaluates the technique using for recommending along the following algorithms:
        1) Root-mean-squared error
        2) Precision at top-k
        3) Spearman Coefficient

        Parameter(s):
        pred : predicted ratings
        y : actual ratings given
        users_items : users for whom ratings have been predicted
        k : value for p@k

        Return:
        rmse : Root-mean squared error
        p@k : precision at top-k; this is a map which gives the precision at top-k for each user.
        sman : Spearman's Correlation; rho = 1-6*((_sigma_)(di)**2)/n(n^2-1)

        Note:
        relevant items : items with actual rating greater or equal to 3.5.
        Recommended item: has a predicted rating >= 3.5
        p@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
    """
    #find out the root-mean-squared-error
    pred_new = []
    for i in pred:
        pred_new.append(i[1])
    rms = rmse(pred_new,y)

    users = {}
    count = 0
    pk = {}
    for i in pred:
        if i[0] not in users.keys():
            users[i[0]] = []
        users[i[0]].append([i[2],i[1],y[count]])
        count += 1
    for i in users.keys():
        users[i] = sorted(users[i],key = itemgetter(1))
        #collect relevant and recommended movies using the criterion described in the design desc.
        relevant = []
        for j in users[i]:
            if j[2] > 3.5:
                relevant.append(j[0])
        recommended = users[i][-k:]
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
    n = len(pred_new)
    square = np.sum(np.square( np.abs( np.asarray(pred_new) - np.asarray(y) ) ))
    sman = 1-(6*square)/(n*(n**2-1))

    return rms,avg_pk,sman

if __name__ == "__main__":
    #no_dup = int(input("Should duplications be allowed while sampling in CUR? (0/1) : "))
    rec,pred,y = predict_cur(50,0)
    rms,pk,sman = evaluate(pred,y,5)
    ratings = get_dataset()
    print("reconstruction error : ",norm(ratings - rec))
    print("rms: {},pk: {},spearman correlation: {}".format(rms,pk,sman))
