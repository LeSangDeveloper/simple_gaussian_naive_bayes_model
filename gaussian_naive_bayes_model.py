import numpy as np
from math import sqrt
from math import exp
from math import pi

class GaussianNB():
    def __init__(self):
        print("Call init")
    
    def __call__(self, X):
        print("call __call__()")
        results = []
        for x in X:
            result = dict()
            for class_name, _ in self.cls_number.items():
                i = 0
                result_value = self.prior[class_name]
                for xi in x:
                    result_value *= self.calculate_probability(xi, self.mean[class_name][i], self.var[class_name][i])
                    i += 1
                result[class_name] = result_value
            
            total = 0
            for _, value in result.items():
                total += value
            for key, value in result.items():
                result[key] = value / total

            results.append(result)
        return results

    def fit(self, X, y):
        print("call fit()")
        # identify classes
        classes = np.unique(y)
        self.cls_number = dict()
        i = 0
        for class_name in classes:
            self.cls_number[class_name] = i
            i += 1

        # run loop for each class
        self.doc_in_class = dict()
        self.prior = dict()
        self.mean = dict()
        self.var = dict()

        for class_name, _ in self.cls_number.items():
            self.doc_in_class[class_name] = np.count_nonzero(y == class_name)
            self.prior[class_name] = self.doc_in_class[class_name] / len(y)
            X_c = []
            for x, label in zip(X, y):
                if label == class_name:
                    X_c.append(x)
            self.mean[class_name] = self.mean_fn(X_c)
            self.var[class_name] = self.stdev(X_c)

        return self

    def calculate_probability(self, x, mean, stdev):
        exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent

    def mean_fn(self, X):
        return np.sum(X, axis=0)/float(len(X))

    def stdev(self, X):
        avg = self.mean_fn(X)
        variance = sum([(x - avg)**2 for x in X]) / float(len(X) - 1)
        return np.sqrt(variance)