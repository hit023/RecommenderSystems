import os
import pickle
from numpy import linalg as LA
import pandas as pd
import numpy as np
from dataset import get_dataset
from operator import itemgetter
import math as mt
from utils import rmse

def svd(r,flag):
    """
        Description:
        Performs SVD decomposition of the given matrix using eigen-pairs.

        Parameter(s):
        r : The matrix to decompose
        flag : 1 if 'r' is the ratings matrix

        Return:
        u,sigma,v : the decomposed elements after performing SVD
    """
    if flag==1:
        if os.path.isfile("./ratings_U.pkl"):
            with open('./ratings_U.pkl', 'rb') as f:
                U = pickle.load(f)
            with open('./ratings_S.pkl', 'rb') as f:
                S = pickle.load(f)
            with open('./ratings_Vt.pkl', 'rb') as f:
                Vt = pickle.load(f)
            return U,S,Vt
    ratings = r
    u_eval, u_ev = LA.eig(np.dot(ratings,ratings.T))
    v_eval, v_ev = LA.eig(np.dot(ratings.T,ratings))

    eigen_values = []
    for i in u_eval:
        if abs(i)!=0:
            eigen_values.append(round(i,2))
    eigen_values = sorted(eigen_values)[::-1]

    u = np.zeros((ratings.shape[0],len(eigen_values)))
    v = np.zeros((ratings.shape[1],len(eigen_values)))

    for i in range(len(eigen_values)):
        for j in range(len(u_ev[:,i])):
            u[j][i] = u_ev[:,i][j]
        for j in range(len(v_ev[:,i])):
            v[j][i] = v_ev[:,i][j]

    v = v.T

    #find the sqrt of the eigen_values to get the singular values.
    sigma = np.zeros((len(eigen_values),len(eigen_values)))
    for i in range(len(eigen_values)):
        sigma[i][i] = eigen_values[i]**0.5

    for i in range(len(sigma)):
        temp = np.dot(ratings,np.matrix(v[i]).T)
        temp_U = np.matrix(u[:,i]).T
        flag=False
        for j in range(len(temp)):
            if temp_U[j]!=0.0:
                if temp[j]/temp_U[j] < 0.0:
                    flag = True
                    break
        if flag:
            for k in range(len(u[:,i])):
                u[k][i] = (-1) * u[k][i]
    if flag==1:
        if not os.path.isfile("./ratings_U.pkl"):
            with open('./ratings_U.pkl','wb') as f:
                pickle.dump(u,f)
            with open('./ratings_S.pkl','wb') as f:
                pickle.dump(sigma,f)
            with open('./ratings_Vt.pkl','wb') as f:
                pickle.dump(v,f)
    return u,sigma,v

def predict_svd_90():
    """
        Description :
        Retains 90% of the energy in terms of singular values of the 'sigma' matrix obtained after
        SVD decomposition and performs reconstruction of the original matrix using these singular
        values.

        Parameter(s):

        Return:
        pred : A list of lists with each list of the form : [userid, prediction,itemid]
        y : The actual ratings given by the users in the test set
    """
    ratings = get_dataset()
    U,S,Vt = svd(ratings,1)
    print(U.shape,S.shape,Vt.shape)
    total = 0
    for i in range(S.shape[0]):
        total += S[i,i]*S[i,i]
    so_far = 0
    ind = 0
    for i in range(S.shape[0]):
        so_far += S[i,i]*S[i,i]
        if so_far/total > 0.9:
            ind = i
            break

    U = U[:,:(ind+1)]
    Vt = Vt[:(ind+1)]
    S = S[:(ind+1)]
    S = S[:,:(ind+1)]
    print(U.shape,S.shape,Vt.shape)
    pred,y = predict_svd(U,S,Vt)
    return pred,y

def predict_svd(U,S,Vt):
    """
        Description :
        Reconstructs the rank matrix using the pieces obtained from SVD decomposition and predicts
        the missing ratings

        Parameter(s):
        U,S,Vt : The decomposed pieces of the ratings matrix itself

        Return:
        pred : A list of lists with each list of the form : [userid, prediction,itemid]
        y : The actual ratings given by the users in the test set
    """
    folder = "./ml-100k/"
    testing = folder + 'u3.test'
    fields = ['user_id', 'item_id', 'rating', 'timestamp']
    dtset = pd.read_csv(testing,sep = '\t',names = fields)
    right_term = np.dot(S,Vt)
    users = {}
    for row in dtset.itertuples():
        if (row[1]-1) not in users.keys():
            users[row[1]-1] = []
        users[row[1]-1].append([row[2]-1,row[3]])
    pred = []
    y = []
    for user in users.keys():
        #load into 'prod', the projection of the user into the latent-space.
        prod = np.dot(U[user,:],right_term)
        for i in users[user]:
            pred.append([user,prod[i[0]],i[0]])
            y.append(i[1])
    return pred,y

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

    #print("calculating spearman coefficient...")
    #calculate spearman correlation
    n = len(pred_new)
    square = np.sum(np.square( np.abs( np.asarray(pred_new) - np.asarray(y) ) ))
    sman = 1-(6*square)/(n*(n**2-1))

    return rms,avg_pk,sman

if __name__ == "__main__":
    ratings = get_dataset()
    U,S,Vt = svd(ratings,1)
    print("Selecting all singular values: ")
    pred,y = predict_svd(U,S,Vt)
    rms,pk,sman = evaluate(pred,y,5)
    print("rms: {},pk: {},spearman correlation: {}".format(rms,pk,sman))

    print("Retaining 90% energy: ")
    pred,y = predict_svd_90()
    rms,pk,sman = evaluate(pred,y,5)
    print("rms: {},pk: {},spearman correlation: {}".format(rms,pk,sman))
