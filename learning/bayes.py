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
    mean_setosa = np.mean(setosa, axis=1)
    mean_versi = np.mean(versicolor, axis=1)
    mean_virginca = np.mean(virginica, axis=1)

    std_setosa = np.std(setosa, axis=1)
    std_versi = np.std(versicolor, axis=1)
    std_virginica = np.std(virginica, axis=1)

    return [PrProb_setosa, PrProb_versi, PrProb_virginica], mean_setosa, mean_versi, mean_virginca, std_setosa, std_versi, std_virginica

def calculate_discriminant(x, PriorProb, mean_setosa, mean_versi, mean_virginca, std_setosa, std_versi, std_virginica):
    CondProb_setosa = []
    CondProb_versi = []
    CondProb_virginica = []

    PostProb_setosa = []
    PostProb_versi = []
    PostProb_virginica = []

    discriminant = []

    for i in range(0, 3):
        CondProb_setosa[i] = (1 / (math.sqrt(2 * math.pi) * mean_setosa[i])) * math.exp( -.5 * ((x - mean_setosa[i]) / std_setosa[i]) ^ 2)
        CondProb_versi[i] = (1 / (math.sqrt(2 * math.pi) * mean_versi[i])) * math.exp( -.5 * ((x - mean_versi[i]) / std_versi[i]) ^ 2)
        CondProb_virginica[i] = (1 / (math.sqrt(2 * math.pi) * mean_virginca[i])) * math.exp( -.5 * ((x - mean_virginca[i]) / std_virginica[i]) ^ 2)

        PostProb_setosa[i] = PriorProb[0] * CondProb_setosa[i] / (PriorProb[0] * CondProb_setosa[i] + PriorProb[1] * CondProb_versi[i] + PriorProb[1] * CondProb_virginica[i])

def main():
    dataset_location = 'iris.data'
    training_set = .70
    testing_set = 1 - training_set

    setosa, versicolor, virginica, data_length = seperate_dataset(extract_dataset(dataset_location))

    print(setosa)
    print(versicolor)
    print(virginica)

    # Plot sepal len vs width
    plt.figure(1)
    plt.scatter(np.array(setosa)[:,0], np.array(setosa)[:,1])
    plt.scatter(np.array(versicolor)[:,0], np.array(versicolor)[:,1])
    plt.scatter(np.array(virginica)[:,0], np.array(virginica)[:,1])
    plt.ylabel("Sepal Length")
    plt.xlabel("Sepal Width")
    plt.title("Sepal len vs width")
    plt.legend(["Setosa", "Versicolor", "Virginica"])

    plt.figure(2)
    plt.scatter(np.array(setosa)[:,2], np.array(setosa)[:,3])
    plt.scatter(np.array(versicolor)[:,2], np.array(versicolor)[:,3])
    plt.scatter(np.array(virginica)[:,2], np.array(virginica)[:,3])
    plt.ylabel("Pedal Length")
    plt.xlabel("Pedal Width")
    plt.title("Pedal len vs width")
    plt.legend(["Setosa", "Versicolor", "Virginica"])


    plt.show()

    return

main()