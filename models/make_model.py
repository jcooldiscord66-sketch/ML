
from collections import Counter
import math
class linearregression:
    def train(self, n):
        data = dict(n)
        x_diff = []
        y_diff = []
        xy_product = []
        x_diff_squared = []
        x_mean = sum(data["X"]) / len(data["X"])
        y_mean = sum(data["y"]) / len(data["y"])
        for i in data["X"]:
            x_diff.append(i - x_mean)
        for j in data["y"]:
            y_diff.append(j - y_mean)
        for i, j in zip(x_diff, y_diff):
            xy_product.append(i * j)
        for z in x_diff:
            x_diff_squared.append(z ** 2)
        sum_xy_product = sum(xy_product)
        sum_x_diff_squared = sum(x_diff_squared)
        self.m = sum_xy_product / sum_x_diff_squared
        self.intercept = y_mean - self.m * x_mean
    def out(self, a):
        return self.intercept + (self.m * a)


class logisticregression:
    def train(self, n: dict):
        slope = 1
        intercept = 1
        learning_rate = 0.1
        for i, j in zip(n["X"], n["y"]):
            z = slope * i + intercept
            pred = 1 / (1 +  2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274
** (-z))
            error = pred - j
            slope = slope - learning_rate * error * i
            intercept = intercept - learning_rate * error
        self.slope = slope
        self.intercept = intercept

    def predict(self, a):
        z = self.slope * a + self.intercept
        return 0 if 1 / (1 + 2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274
 ** (-z)) < 0.5 else 1
class KNN:
    def one_d(self, data: dict):
        self.data = data
    def predict_oned(self, K):
        distance = []
        label_distance = []
        for i in self.data["X"]:
            distance.append(abs(K - i))
        for i, n in zip(distance, self.data["y"]):
            label_distance.append([i, n])
        sort = sorted(label_distance, key=lambda x: x[0])
        k_neighbour = sort[:K]
        vote_count = [label for _, label in k_neighbour]
        winner = Counter(vote_count).most_common()
        return winner[0][0]