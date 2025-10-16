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