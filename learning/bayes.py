import csv
import numpy as np
import math
import matplotlib.pyplot as plt


## Extracts our dataset from the csv
def extract_dataset(dataset):
    with open(dataset, mode='r', newline='') as csv_file:
        file = csv.reader(csv_file)
        data = list(file)
    return data # as a list

## Returns the seperated version of our data set
def seperate_dataset(dataset):
    length = len(dataset) - 1

    setosa = []
    versicolor = []
    virginica = []

    for i in range(0, length):
        if dataset[i][4] == "Iris-setosa": # is num 1
            setosa.append([float(dataset[i][0]), float(dataset[i][1]), float(dataset[i][2]), float(dataset[i][3]), 1])
        elif dataset[i][4] == "Iris-versicolor": # is num 2
            versicolor.append([float(dataset[i][0]), float(dataset[i][1]), float(dataset[i][2]), float(dataset[i][3]), 2])
        elif dataset[i][4] == "Iris-virginica": # is num 3
            virginica.append([float(dataset[i][0]), float(dataset[i][1]), float(dataset[i][2]), float(dataset[i][3]), 3])

    return setosa, versicolor, virginica, length

def calculate_stats(setosa, versicolor, virginica):
    len_setosa = len(setosa); len_versi = len(versicolor); len_virginica = len(virginica)
    len_data = len_setosa + len_versi + len_virginica

    setosa = np.array(setosa, np.float)
    versicolor = np.array(versicolor, np.float)
    virginica = np.array(virginica, np.float)

    # prior probabilities
    PrProb_setosa = len_setosa/len_data
    PrProb_versi = len_versi/len_data
    PrProb_virginica = len_virginica/len_data

    # class condititional probabilities
    mean_setosa = np.mean(setosa, axis=0)
    mean_versi = np.mean(versicolor, axis=0)
    mean_virginca = np.mean(virginica, axis=0)

    std_setosa = np.std(setosa, axis=0)
    std_versi = np.std(versicolor, axis=0)
    std_virginica = np.std(virginica, axis=0)

    return [PrProb_setosa, PrProb_versi, PrProb_virginica], [mean_setosa, mean_versi, mean_virginca],  [std_setosa, std_versi, std_virginica]

def calculate_discriminant(x, PriorProb, means, standard_devs):

    mean_setosa = means[0]
    mean_versi = means[1]
    mean_virginica = means[2]

    std_setosa = standard_devs[0]
    std_versi = standard_devs[1]
    std_virginica = standard_devs[2]

    CondProb_setosa = []
    CondProb_versi = []
    CondProb_virginica = []

    PostProb_setosa = []
    PostProb_versi = []
    PostProb_virginica = []

    discriminant = []

    # Probabilities
    for i in range(0, 4):

        CondProb_setosa.append( (1 / (math.sqrt(2 * math.pi) * mean_setosa[i])) * math.exp( (-0.5) * ((x[i] - mean_setosa[i]) / std_setosa[i]) ** 2) )
        CondProb_versi.append( (1 / (math.sqrt(2 * math.pi) * mean_versi[i])) * math.exp( (-0.5) * ((x[i] - mean_versi[i]) / std_versi[i]) ** 2) )
        CondProb_virginica.append( (1 / (math.sqrt(2 * math.pi) * mean_virginica[i])) * math.exp( (-0.5) * ((x[i] - mean_virginica[i]) / std_virginica[i]) ** 2) )

        PostProb_setosa.append( PriorProb[0] * CondProb_setosa[i] / (PriorProb[0] * CondProb_setosa[i] + PriorProb[1] * CondProb_versi[i] + PriorProb[1] * CondProb_virginica[i]) )
        PostProb_versi.append( PriorProb[1] * CondProb_versi[i] / (PriorProb[0] * CondProb_setosa[i] + PriorProb[1] * CondProb_versi[i] + PriorProb[1] * CondProb_virginica[i]) )
        PostProb_virginica.append( PriorProb[2] * CondProb_virginica[i] / (PriorProb[0] * CondProb_setosa[i] + PriorProb[1] * CondProb_versi[i] + PriorProb[1] * CondProb_virginica[i]) )

    print(CondProb_setosa)
    # Discriminant
    val = 0
    for i in range(0, 4):
        for j in range(0,3):
            if j == 0: # s vs v
                val = PostProb_setosa[i] - PostProb_versi[i]
            elif j == 1:# s vs vi
                val = PostProb_setosa[i] - PostProb_virginica[i]
            else:# v vs vi
                val = PostProb_versi[i] - PostProb_virginica[i]

            discriminant.append(val)

    return discriminant

def plots(setosa, versicolor, virginica):
    # Plot sepal len vs width
    plt.figure(1)
    plt.scatter(np.array(setosa)[:, 0], np.array(setosa)[:, 1])
    plt.scatter(np.array(versicolor)[:, 0], np.array(versicolor)[:, 1])
    plt.scatter(np.array(virginica)[:, 0], np.array(virginica)[:, 1])
    plt.ylabel("Sepal Length")
    plt.xlabel("Sepal Width")
    plt.title("Sepal len vs width")
    plt.legend(["Setosa", "Versicolor", "Virginica"])

    plt.figure(2)
    plt.scatter(np.array(setosa)[:, 2], np.array(setosa)[:, 3])
    plt.scatter(np.array(versicolor)[:, 2], np.array(versicolor)[:, 3])
    plt.scatter(np.array(virginica)[:, 2], np.array(virginica)[:, 3])
    plt.ylabel("Pedal Length")
    plt.xlabel("Pedal Width")
    plt.title("Pedal len vs width")
    plt.legend(["Setosa", "Versicolor", "Virginica"])

    plt.show()

    return

def main():
    dataset_location = 'iris.data'
    training_set = .70

    setosa, versicolor, virginica, data_length = seperate_dataset(extract_dataset(dataset_location))

    # splitting the dataset in two
    # A training set
    setosa_training_set = setosa[0: math.floor(training_set * len(setosa))]
    versicolor_training_set = versicolor[0: math.floor(training_set * len(versicolor))]
    virginica_training_set = virginica[0: math.floor(training_set * len(virginica))]

    # A testing set
    setosa_test_set = setosa[math.ceil(training_set * len(setosa)) : len(setosa)]
    versicolor_test_set = versicolor[math.ceil(training_set * len(versicolor)) : len(versicolor)]
    virginica_test_set = virginica[math.ceil(training_set * len(virginica)) : len(virginica)]

    prior_prob, mean, standard_dev = calculate_stats(setosa_training_set, versicolor_training_set, virginica_training_set)

    #testing on setosa
    out = calculate_discriminant(setosa_test_set[3], prior_prob, mean, standard_dev)
    print(out)

    return

main()