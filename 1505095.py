import numpy as np


numOfClass, numOfFeatures, datasetLength = 0,0,0

dataset = []

count = 0

file = open("trainLinearlyNonSeparable.txt")

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
###########basic perceptron algo
print("basic perceptron algorithm\n")

for iteration in range(1000):
# while True:
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

test_file = open("testLinearlyNonSeparable.txt")

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

print("Accuracy for basic perceptron: ", test_accuracy)

###################reward and punishment algo
print("\n\n\nreward and punishment algorithm\n")

np.random.seed(42)
weight = np.random.rand(numOfFeatures+1, 1)  ## np.random.random_sample()
# weight = np.zeros(numOfFeatures+1)
# print("random weight ", weight)

learning_rate = 0.25
count = 0

############ reward and punishment algo starts here#######
for epoch in range(1000):
# while True:
    count = 0
    for i in dataset:
        x = np.array(i)
        grp = x[numOfFeatures]
        x[numOfFeatures] = 1
        x = x.reshape(numOfFeatures+1,1)
        # print("shape of weight "+ str(weight.shape))
        # print("shape of x "+ str(x.shape))
        dot_product = np.dot(weight.T, x)[0]
        if(dot_product <= 0 and grp == 1):
            weight = weight+(learning_rate*x)
            # count += 1
            # print("shape of weight "+ str(weight.shape))
            # print(weight)
        elif(dot_product > 0 and grp == 2):
            weight = weight-(learning_rate*x)
            # print(weight)
            # count += 1
        else:
            count += 1
        
    # else:
        # count += 1
        # continue
    if count == datasetLength:
        break
    # count += 1
# if count == 0:
#     break
# if count == datasetLength:
#     break
# print('weight: ', weight)

test_dataset = []

test_file = open("testLinearlyNonSeparable.txt")

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
    dot_product = np.dot(weight.T, x)[0]
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

print("Accuracy for reward and punishment algorithm: ", test_accuracy)

#############pocket algo
print("\n\n\npocket algorithm\n")

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

#def test_perceptron(test_dataset, weight):
np.random.seed(42)
weight_pocket = np.random.rand(numOfFeatures+1, 1)
weight = weight_pocket

accuracy = 0.0

for data in dataset:
    x = np.array(data)
    grp = x[numOfFeatures]
    x[numOfFeatures] = 1
    x = np.array(x)
    x = x.reshape(numOfFeatures+1, 1)
    dot_product = np.dot(weight.T, x)[0]
    predicted = -1
    if dot_product >= 0:
        predicted = 1
    else:
        predicted = 2
    
    if predicted == grp:
        accuracy += 1

test_accuracy = (accuracy/len(dataset))

print("Accuracy: ", test_accuracy*100)

#return test_accuracy

# def basic_perceptron():
#     np.random.seed(95)
#     wp = np.random.uniform(-15, 15, numOfFeatures+1)
#     weight = wp

#     learning_rate = 0.025

#     t = 0

#     misclassification = test_perceptron(dataset, weight)*len(dataset)
#     print("misclassification ", misclassification)

#     for i in range(datasetLength):
#         Y = []
#         del_x = []
#         correction_vector = np.zeros(numOfFeatures+1)
#         count = 0

#         for i in range(datasetLength):
#             x = np.array(dataset[i])
#             grp = x[numOfFeatures]
#             x[numOfFeatures] = 1
#             x = x.reshape(numOfFeatures+1,1)
#             dot_product = np.dot(weight, x)[0]
#             if(grp == 2 and dot_product>0):
#                 Y.append(x)
#                 del_x.append(1)
#                 count += 1
#             if(grp ==1 and dot_product<0):
#                 Y.append(x)
#                 del_x.append(-1)
#                 count += 1

#         for i in range(len(Y)):
#             correction_vector += del_x[i]*Y[i].transpose()[0]
        
#         weight = weight-(learning_rate*correction_vector)

#         if count < misclassification:
#             misclassification = count
#             wp = weight
        
#         if len(Y) == 0:
#             break
#     print('weight: ', wp)
#     return wp


#def pocket_algo():
#np.random.seed(42)
#weight = np.random.rand(numOfFeatures+1, 1)  ## np.random.random_sample()
# weight = np.random.uniform(-15, 15, numOfFeatures+1)

learning_rate = 0.025
# misclassification = test_perceptron(dataset, weight)*datasetLength
misclassification = test_accuracy*datasetLength
print("misclassification ", misclassification)

for i in range(datasetLength):
    count = 0 
    for i in dataset:
        x = np.array(i)
        grp = x[numOfFeatures]
        x[numOfFeatures] = 1
        x = x.reshape(numOfFeatures+1,1)
        dot_product = np.dot(weight.T,x)[0]
        if dot_product<=0 and grp == 1:
            # count += 1
            weight = weight + learning_rate*x
        elif dot_product >0 and grp == 2:
            weight = weight - learning_rate*x
            # count += 1
        else:
            count += 1
    
    if count<misclassification:
        misclassification = count
        weight_pocket = weight
    
    if count == 0:
        break
print(weight_pocket)

accuracy = 0.0
sample = 0

for test_data in test_dataset:
    x = np.array(test_data)
    grp = x[numOfFeatures]
    x[numOfFeatures] = 1
    x = np.array(x)
    x = x.reshape(numOfFeatures+1, 1)
    dot_product = np.dot(weight.T, x)[0]
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

test_accuracy = (accuracy/len(test_dataset))

print("Accuracy for pocket algorithm: ", test_accuracy*100)