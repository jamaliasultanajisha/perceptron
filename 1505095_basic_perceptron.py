import numpy as np


numOfClass, numOfFeatures, datasetLength = 0,0,0

dataset = []

count = 0

file = open("trainLinearlySeparable.txt")

lines = file.readlines()

for line in lines:
    if(count == 0):
        var = line.split()
        numOfFeatures = int(var[0])
        numOfClass = int(var[1])
        datasetLength = int(var[2])
    else:
        var = line.split()
        size = len(var)
        data = []
        index = 0
        for i in var:
            if(index == size - 1):
                data.append(int(i))
            else:
                data.append(float(i))
            index = index + 1
        dataset.append(data)
    count = count + 1
    
# print(numOfClass, numOfFeatures, datasetLength)
# print(dataset)

# class_wise_dataset = []

# classes = set()

# for data in dataset:
#     classes.add(data[numOfFeatures])
    
# print(classes)

np.random.seed(95)
weight = np.random.uniform(-15, 15, numOfFeatures+1)

learning_rate = 0.25

t = 0

for iteration in range(1000):
    Y = []
    del_x = []
    correction_vector = np.zeros(numOfFeatures+1)

    for i in range(datasetLength):
        x = np.array(dataset[i])
        grp = x[numOfFeatures]
        x[numOfFeatures] = 1
        x = x.reshape(numOfFeatures+1,1)
        dot_product = np.dot(weight, x)[0]
        if(grp == 2 and dot_product>0):
            Y.append(x)
            del_x.append(1)
        if(grp ==1 and dot_product<0):
            Y.append(x)
            del_x.append(-1)

    for i in range(len(Y)):
        correction_vector += del_x[i]*Y[i].transpose()[0]
    
    weight = weight-(learning_rate*correction_vector)

    if len(Y) == 0:
        break
print('weight: ', weight)

test_dataset = []

test_file = open("testLinearlySeparable.txt")

test_line = test_file.readlines()

for line in test_line:
    var = line.split()
    size = len(var)
    data = []
    idx = 0
    for i in var:
        if(idx == size - 1):
            data.append(int(i))
        else:
            data.append(float(i))
        idx = idx + 1
    test_dataset.append(data)
    
# print("test dataset")
# print(test_dataset)

accuracy = 0.0
sample = 0

for test_data in test_dataset:
    x = np.array(test_data)
    grp = x[numOfFeatures]
    x[numOfFeatures] = 1
    x = np.array(x)
    x = x.reshape(numOfFeatures+1, 1)
    dot_product = np.dot(weight, x)[0]
    predicted = -1
    if dot_product >= 0:
        predicted = 1
    else:
        predicted = 2
    
    if predicted == grp:
        accuracy += 1
        sample += 1
    else:
        sample += 1
        print(sample, test_data, predicted)

test_accuracy = (accuracy/len(test_dataset))*100

print("Accuracy: ", test_accuracy)
