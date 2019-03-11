import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt

# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

def getFiles():
    rows = []
    with open('fer2013.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            rows.append(row)

    train = []
    test = []
    testPrivate = []

    for i in range(1, len(rows)):
        a = rows[i][1]
        b = []
        while(a.find(" ") != -1):
            b.append(int(a[:a.find(" ")]))
            a = a[a.find(" ") + 1:]
        b.append(int(a))
        rows[i][0] = np.array(int(rows[i][0]))
        rows[i][1] = np.reshape(np.array(b), (48,48))
        if(rows[i][2] == "Training"):
            train.append([rows[i][0], rows[i][1]])
        elif(rows[i][2] == "PublicTest"):
            test.append([rows[i][0], rows[i][1]])
        else:
            testPrivate.append([rows[i][0], rows[i][1]])
        if(i % 100 == 0):
            print(100 * (i/len(rows)), i)

    with open('train', 'wb') as f:
        pickle.dump(train, f)

    with open('test', 'wb') as f:
        pickle.dump(test, f)

    with open('testPrivate', 'wb') as f:
        pickle.dump(testPrivate, f)



def main():
    getFiles()

main()





