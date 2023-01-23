from gaussian_naive_bayes_model import GaussianNB
import numpy as np

# Test summarizing by class
dataset = [[3.393533211,2.331273381],
 [3.110073483,1.781539638],
 [1.343808831,3.368360954],
 [3.582294042,4.67917911],
 [2.280362439,2.866990263],
 [7.423436942,4.696522875],
 [5.745051997,3.533989803],
 [9.172168622,2.511101045],
 [7.792783481,3.424088941],
 [7.939820817,0.791637231]]
labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

train_data = np.array(dataset)
label = np.array(labels)

model = GaussianNB()
model.fit(dataset, label)
print(model(np.array([[3.393533211,2.331273381], [7.939820817,0.791637231]])))