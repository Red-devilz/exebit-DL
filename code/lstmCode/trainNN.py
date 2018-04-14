#!/usr/bin/env python3
from __future__ import print_function
import json
import re
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from nltk.stem.porter import PorterStemmer


def printDataSummary(xSeq, ySeq):

    flatSeq = set([item for sublist in xSeq for item in sublist])
    seqLen = [len(sublist) for sublist in xSeq]

    print("  * Loaded data from files ...... ")

    print("  * Data Summary")
    print("    - Number of sequences : " + str(len(xSeq)))
    print("    - Number of sequences classes : " + str(len(set(ySeq))))
    print("    - Number of words : " + str(len(flatSeq)))
    print("    - Average sequence length: " + str(sum(seqLen) / len(seqLen)))

    print("  * Printing some samples")
    for s in range(5):
        i = random.randint(0, 4500)  # Sample point
        print("        Sample No. : " + str(i))
        print("          Brand :", end='')
        print(str(ySeq[i]))
        print("          Sequence :", end='')
        print(xSeq[i])

    #  raw_input("\n\nPress enter to continue......")
    input("\n\nPress enter to continue......")


def loadMaps(seqDat, labDat):
    wordMap = "./data/idToWordMap.json"
    labelMap = "./data/idToBranddMap.json"

    #  Load the data from files
    with open(wordMap, "r") as file1:
        idToWord = json.load(file1)

    with open(labelMap, "r") as file1:
        brandToWord = json.load(file1)

    #  Print some samples
    print("\n  * Printing some samples with words")
    for s in range(10):
        i = random.randint(0, 4500)  # sample point
        #  Convert to words
        seqWords = [idToWord[str(w)] for w in seqDat[i]]
        brandWord = brandToWord[str(labDat[i])]
        print("        " + str(i) + ")")
        print("          Brand :", end='')
        print(str(brandWord))
        print("          Sequence :", end='')
        print(seqWords)
    input("\n\nPress enter to continue......")
    #  raw_input("\n\nPress enter to continue......")

    return idToWord, brandToWord


def loadSequences():
    seqData = "./data/sequences.json"
    seqLabel = "./data/sequenceBrand.json"

    #  Load the data from files
    with open(seqData, "r") as file1:
        xSeq = json.load(file1)

    with open(seqLabel, "r") as file1:
        ySeq = json.load(file1)

    printDataSummary(xSeq, ySeq)

    return xSeq, ySeq


def makeOneHot(ydat):
    one_hot_y = []
    for y in ydat:
        a = [0 for i in range(9)]
        a[y] = 1
        one_hot_y.append(a)
    return one_hot_y


def get_text(input_text):
    #  Get text,
    #  Convert to lower case
    #  Remove all non alphabet letter
    input_text = input_text.lower()
    input_text = re.sub('[^0-9a-zA-Z]+', ' ', input_text)
    input_text = re.sub('[0-9]+', '', input_text)
    input_text = input_text.split()
    return input_text


def proc_inp(text, xmap, ymap):
    x_data = []

    st = PorterStemmer()
    cur_x = []
    wrd = get_text(text)
    for w in wrd:
        stem_w = st.stem(w)
        if(stem_w in xmap.values()):
            for x in xmap.keys():
                if xmap[x] == stem_w:
                    cur_x.append(x)

    x_data.append(cur_x)

    return x_data


def brand_check(model, xmap, ymap, text):
    #  Get Data
    xdat = proc_inp(text, xmap, ymap)
    print("-----------")
    print(text)
    X_pred = sequence.pad_sequences(
        xdat, maxlen=31, dtype='int32', padding='pre', truncating='pre', value=2781)

    ans = model.predict(X_pred, verbose=0)

    pred = []
    for y in ymap.keys():
        pred.append((ans[0][int(y)], ymap[y]))

    pred.sort(key=lambda z: -z[0])
    return pred[0:3]


def main():
    seq, seqLabel = loadSequences()
    wordMap, labelMap = loadMaps(seq, seqLabel)

    DATSIZE = 4500
    VOCAB = 2782
    MAX_LEN = 31

    ind = np.random.permutation(DATSIZE)
    indTrain = ind[:4000]
    indTest = ind[4000:]

    X_vals = sequence.pad_sequences(
        seq, maxlen=31, dtype='int32', padding='pre', truncating='pre', value=2781)
    y_vals = np.asmatrix(makeOneHot(seqLabel))

    X_train = X_vals[indTrain]
    X_test = X_vals[indTest]

    y_train = y_vals[indTrain]
    y_test = y_vals[indTest]

    #  Try different optimizers, dimensions
    EMBED = 300
    opt = 'adam'
    batchSize = 10

    model = Sequential()
    model.add(Embedding(VOCAB, EMBED, input_length=MAX_LEN))
    model.add(LSTM(EMBED))
    model.add(Dense(9, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    print(model.summary())
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              batch_size=batchSize,
              epochs=1)

    print("\n  * Finished model training")
    print("\n\n ===== Model testing ======")

    #  https://www.amazon.com/s/ref=sr_in_s_p_89_7499?fst=as%3Aoff&rh=n%3A172282%2Cp_89%3ASupershieldz&bbn=172282&ie=UTF8&qid=1492120349&rnid=2528832011
    print(brand_check(
        model, wordMap, labelMap,
        "[2-Pack] iPhone 7 Tempered Glass Screen Protector, Supershieldz Anti-Scratch, Anti-Fingerprint, Bubble Free [3D Touch Compatible]"))

    #  https://www.amazon.com/s/ref=sr_in_s_p_89_7499?fst=as%3Aoff&rh=n%3A172282%2Cp_89%3ASupershieldz&bbn=172282&ie=UTF8&qid=1492120349&rnid=2528832011
    print(brand_check(
        model, wordMap, labelMap,
        "[2-Pack] iPhone 7 Tempered Glass Screen Protector, Anti-Fingerprint, Bubble Free [3D Touch Compatible]"))

    print(brand_check(
        model, wordMap, labelMap,
        "Genuine Xerox High Capacity Black Toner Cartridge for the Phaser 6600 or WorkCentre 6605, 106R02228"))

    #  https://www.amazon.com/s/ref=sr_pg_5?fst=as%3Aoff&rh=n%3A172282%2Cp_89%3AApple&page=5&bbn=172282&ie=UTF8&qid=1492120481
    print(brand_check(
        model, wordMap, labelMap,
        "Apple iPhone SE Unlocked Phone - 64 GB Retail Packaging - Space Gray"))


if __name__ == '__main__':
    main()
