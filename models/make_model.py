from collections import Counter
import numpy as np
import math
from ML.encoders import OneHotEncoder
from pandas import *  
class RegressionError(Exception):
    pass
class linearRegression:
    def fit(self,data:dict):
        if not isinstance(data, dict) or "X" not in data or "y" not in data:
            raise RegressionError("Input must be a dictionary with keys 'X' and 'y'")
        assert len(data["X"]) == len(data["y"]),"the data must be of same length"
        feature_matrix = np.array(data["X"])
        n_samples = feature_matrix.shape[0]
        bias_column = np.ones((n_samples,1))
        feature_matrix = np.hstack([bias_column,feature_matrix])
        target_vector = np.array(data["y"]).reshape(-1,1)
        transpose = feature_matrix.transpose()
        mul = transpose @ feature_matrix
        inverse = np.linalg.pinv(mul)
        feature_ans_mul = transpose @ target_vector
        final = inverse @ feature_ans_mul
        self.intercept = final.flatten().tolist()[0]
        self.weights = final.flatten().tolist()[1:]
    def predict (self,value):
        prediction = 0
        for i,j in zip(value,self.weights):
            prediction += i*j
        prediction += self.intercept
        return prediction
def mean(v):
    return sum(v)/len(v)
def std(p):
    m = mean(p)
    dif = []
    dif_sqr = []
    for i in p:
        dif.append(i-m)
    for j in dif:
        dif_sqr.append(j**2)
    r = sum(dif_sqr)/len(p)
    final = math.sqrt(r)
    return final if r!=0 else 1e-8
class logisticregression:
    def fit(self,data:dict,epoches=100):
        if not isinstance(data, dict) or "X" not in data or "y" not in data:
            raise ValueError("Input must be a dictionary with keys 'X' and 'y'")
        weights =[]
        intercept =0
        each_feature =[]
        main = []
        for i in zip(*data["X"]):
            each_feature.append(i)
        for j in each_feature :
            weights.append(0)
        for k in each_feature:
            main.append([(x-mean(k))/std(k)for x in k])
        main = list(zip(*main))
        learning_rate =  0.1
        for n in range(epoches):
            for i,j in zip(main,data["y"]):
                prediction = 0
                for x,w in zip(i,weights):
                    prediction += x*w
                prediction += intercept
                p = 1/(1+math.e**-prediction)
                error = p - j
                for q in range(len(weights)):
                    weights[q] -= learning_rate*error*i[q]
                intercept = intercept - learning_rate  * error
        self.std = [std(value)for value in each_feature]
        self.mean = [mean(d) for d in each_feature]
        self.weights = weights
        self.intercept = intercept
    def predict(self,value):
        z_scored = []
        for i,j,k in zip(value,self.mean,self.std):
            z_scored.append((i-j)/k)
        self.predicion = 0
        for y,z in zip(z_scored,self.weights):
            self.predicion += y*z
        self.predicion += self.intercept
        self.p = 1/(1+math.exp(-self.predicion))
        return self.p
class LogisticRegressionMultipleCat:
    def fit(self,X,y):
        encoder = OneHotEncoder()
        encoder.fit(y,dtype=int) #type:ignore
        encoded = encoder.get_encoded()
        models = {}
        for i in encoded:
            models.update({i:logisticregression()})
        for j in encoded:
            models[j].fit(
                {
                    "X":X,
                    "y":encoded[j]
                }
            )
        self.models = models
    def predict(self,value):
        self.votes = {}
        count = []
        for i in self.models:
            self.votes.update({i:self.models[i].predict(value)})
        for j in self.votes.values():
            count.append(j)
        winner = max(self.votes,key=self.votes.get) #type:ignore
        return winner 
class KNN:
    def fit(self,X,y):
        self.X   = X
        self.y = y
    def predict(self,value,K=2,method="Distance",mode="categerical"):
        D = []
        for i,p in zip(self.X,self.y):
            e = 0
            for y,z in zip(i,value):
                e += (y-z)**2
            d = math.sqrt(e)
            D.append([d,p])
        D.sort()
        K_neighbours = []
        if method == "Distance":
            for i in D:
                if i[0]<=K:
                    K_neighbours.append(i)
        elif method == "Top":
            K_neighbours = D[:K]
        else:
            for i in D:
                if i[0]<=K:
                    K_neighbours.append(i)
        if not K_neighbours:
            return None
        answers = [q[1]for q in K_neighbours]
        if mode == "categerical":
            w = Counter(answers)
            winner = w.most_common()
            return winner[0]
        elif mode == "Regression":
            d = sum(answers)/len(answers)
            return d
        else:
            w = Counter(answers)
            winner = w.most_common()
            return winner[0]
