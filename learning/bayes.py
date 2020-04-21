import csv

## Extracts our dataset from the csv
def extract_dataset(dataset):
    with open(dataset, mode='r', newline='') as csv_file:
        file = csv.reader(csv_file)
        data = list(file)
    return data # as a dictionary

## Returns the seperated version of our data set
def seperate_dataset(dataset):
    length = len(dataset)

    setosa = []
    versicolor = []
    virginica = []

    for i in dataset:
        if i[4] == "Iris-setosa":
            setosa.append([i[0], i[1], i[2], i[3]])
        elif i[4] == "Iris-versicolor":
            versicolor.append([i[0], i[1], i[2], i[3]])
        elif i[4] == "Iris-virginica":
            virginica.append([i[0], i[1], i[2], i[3]])

    return setosa, versicolor, virginica, length

def main():
    dataset_location = 'iris.data'
    setosa, versicolor, virginica, data_length = seperate_dataset(extract_dataset(dataset_location))

    return

