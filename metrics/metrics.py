import math
class RegressionMetrics:
    @staticmethod
    def Mean_squared_error(y_true,y):
        sub_sqr = []
        for i,j in zip(y_true,y):
            sub_sqr.append((i-j)**2)
        MSE = sum(sub_sqr)/len(sub_sqr)
        return MSE
    @staticmethod
    def Mean_absoulte_error(y_true,y):
        absolute = []
        for i,j in zip(y_true,y):
            absolute.append(abs(i-j))
        MAE = sum(absolute)/len(absolute)
        return MAE
    @staticmethod
    def Root_mean_squared_error(y_true,y):
        sub_sqr = []
        for i,j in zip(y_true,y):
            sub_sqr.append((i-j)**2)
        MSE = sum(sub_sqr)/len(sub_sqr)
        RMSE = MSE ** 0.5
        return RMSE
    @staticmethod
    def Mean_bias_error(y_true,y):
        error =[]
        for i,j in zip(y_true,y):
            error.append(i-j)
        MBE = sum(error)/len(error)
        return MBE
    @staticmethod
    def Mean_absolute_percentage_error(y_true,y):
        try:
            per_error = []
            for i,j in zip(y_true,y):
                per_error.append(abs(i-j)/i)
            MAPE = (sum(per_error)/len(y_true))*100
            return MAPE
        except ZeroDivisionError:
            return "cant divide by zero make sure your actual data doesnt have any 0"
    @staticmethod
    def R_squared(y_true,y):
        v =[]
        m = sum(y_true)/len(y_true)
        for i in y_true:
            v.append((i-m)**2)
        t = sum(v)
        r = []
        for j,k in zip(y_true,y):
            r.append((j-k)**2)
        re = sum(r)
        div = re/t
        R2 = 1 - div
        return R2
class ClassificationMetrics:
    def _cal_(self,y_true,y):
        TP = 0 
        TF = 0
        FP =0
        FN = 0
        for i,j in zip(y_true,y):
            if i==1 and j==1:
                TP+=1
            if i==0 and j==0:
                TF += 1
            if i==1 and j==0:
                FN += 1
            if i==0 and j==1:
                FP += 1
        d = {"TP":TP,"TN":TF,"FP":FP,"FN":FN}
        return d
    def accuracy(self,y_true,y):
        total = len(y_true) if len(y_true)==len(y) else None
        if not total:
            raise ValueError("malformed inupt")
        c = self._cal_(y_true,y)
        tp = c.get("TP")
        tn = c.get("TN")
        return (tp+tn)/total #type:ignore
    def precision(self,y_true,y):
        c = self._cal_(y_true,y)
        tp = c.get("TP")
        fp = c.get("FP")
        if tp+fp == 0:#type:ignore
            return 0
        else:
            return tp/(tp+fp) #type:ignore
    def recall(self,y_true,y):
        c = self._cal_(y_true,y)
        tp = c.get("TP")
        fn = c.get("FN")
        if tp+fn==0:#type:ignore
            return 0
        else:
            return tp/(tp+fn) #type:ignore
    def specificity(self,y_true,y):
        c = self._cal_(y_true,y)
        tn = c.get("TN")
        fp = c.get("FP")
        if tn+fp==0:#type:ignore
            return 0
        else:
            return tn/(tn+fp)#type:ignore
    def f1_score(self,y_true,y):
        p_mul_r = self.precision(y_true,y)*self.recall(y_true,y)
        p__add_r = self.precision(y_true,y)+self.recall(y_true,y)
        if p_mul_r==0 or p__add_r==0:
            return 0
        else:
            s = p_mul_r/p__add_r
            return 2*s
    def balanced_accuracy(self,y_true,y):
        rec = self.recall(y_true,y)
        spec = self.specificity(y_true,y)
        return (rec+spec)/2
    def G_mean(self,y_true,y):
        rec = self.recall(y_true,y)
        spec = self.specificity(y_true,y)
        return math.sqrt(rec*spec)
    def False_discovery_rate(self,y_true,y):
        c = self._cal_(y_true,y)
        fp = c.get("FP")
        tp = c.get("TP")
        if tp+fp==0: #type:ignore
            return 0
        else:
            return fp/(tp+fp) #type:ignore
    def False_ommision_rate(self,y_true,y):
        c = self._cal_(y_true,y)
        tn = c.get("TN")
        fn = c.get("FN")
        if tn+fn==0:#type:ignore
            return 0
        else:
            return fn/(tn+fn) #type:ignore
    def Negative_predictive_value(self,y_true,y):
        c = self._cal_(y_true,y)
        tn = c.get("TN")
        fn = c.get("FN")
        if tn+fn==0:#type:ignore
            return 0
        else:
            return tn/(tn+fn)#type:ignore
    def Threat_score(self,y_true,y):
        c = self._cal_(y_true,y)
        tp = c.get("TP")
        fp = c.get("FP")
        fn = c.get("FN")
        if tp+fn+fp==0:#type:ignore
            return 0
        else:
            return tp/(tp+fn+fp)#type:ignore
    def Mathews_correlation_coefficient(self,y_true,y):
        #(TP×TN - FP×FN) / sqrt((TP+FP)×(TP+FN)×(TN+FP)×(TN+FN))
        c = self._cal_(y_true,y)
        tp = c.get("TP")
        tn = c.get("TN")
        fp = c.get("FP")
        fn = c.get("FN")
        d = (tp*tn)-(fp*fn)#type:ignore
        e = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))#type:ignore
        if e==0:
            return 0
        else:
            return d/e 
    def Youden_J_index(self,y_true,y):
        return (self.recall(y_true,y)+self.specificity(y_true,y))-1
    
