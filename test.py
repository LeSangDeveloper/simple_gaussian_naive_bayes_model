from gaussian_naive_bayes_model import GaussianNB

model = GaussianNB()
model.fit([2, 2], [1, 1])
model([1, 2, 3])