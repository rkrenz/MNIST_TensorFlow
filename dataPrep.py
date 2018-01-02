import numpy
import matplotlib.pyplot as plt
import os
import sys

def getRawDataFromInternet():
    # fetch MNIST data from Internet and create feature and label numpy arrays
    print("fetching raw MNIST data from Internet...")
    print()
    from sklearn.datasets import fetch_mldata
    dictionary = fetch_mldata('MNIST original')
    rawFeatures = dictionary['data'].astype(int)
    rawLabels = dictionary['target'].astype(int)
    directory = 'rawData'
    if not os.path.exists(directory):
        os.makedirs(directory)
    numpy.save(directory + "/rawFeatures.npy", rawFeatures, allow_pickle=False)
    numpy.save(directory + "/rawLabels.npy", rawLabels, allow_pickle=False)
    return rawFeatures, rawLabels
    
def getRawDataFromFiles():
    # fetch MNIST data from previously saved files
    print("fetching raw MNIST data from files...")
    print()
    try:
        directory = 'rawData'
        rawFeatures = numpy.load(directory + "/rawFeatures.npy", mmap_mode=None, allow_pickle=False)
        rawLabels = numpy.load(directory + "/rawLabels.npy", mmap_mode=None, allow_pickle=False)
        return rawFeatures, rawLabels
    except:
        print("ERROR: no raw data found.  did you fetch once from Internet first?")
        sys.exit(1)

def partitionData(rawFeatures, rawLabels, testPercent):
    assert len(rawFeatures) == len(rawLabels)
    index = int(len(rawFeatures)*(1-testPercent))
    rawTrainFeatures = rawFeatures[0:index]
    rawTrainLabels = rawLabels[0:index]
    rawTestFeatures = rawFeatures[index:]
    rawTestLabels = rawLabels[index:]
    return rawTrainFeatures, rawTrainLabels, rawTestFeatures, rawTestLabels
    
def normalizeArray(array, divisor):
    if divisor == 0:
        return array
    return array / divisor

def vectorizeArray(array, vectorWidth):
    vectorizedArray = numpy.zeros(shape=(len(array), vectorWidth))
    for count in range(len(array)):
        vector = numpy.array([0.0] * vectorWidth)
        vector[array[count]] = 1.0
        vectorizedArray[count] = vector
    return vectorizedArray

def randomizeTrainingData(trainFeatures, trainLabels):
    assert len(trainFeatures) == len(trainLabels)
    permutation = numpy.random.permutation(len(trainFeatures))
    return trainFeatures[permutation], trainLabels[permutation]

def saveProcessedDataToFiles(trainFeatures, trainLabels, testFeatures, testLabels):
    directory = 'processedData'
    if not os.path.exists(directory):
        os.makedirs(directory)
    numpy.save(directory + "/trainFeatures.npy", trainFeatures, allow_pickle=False)
    numpy.save(directory + "/trainLabels.npy", trainLabels, allow_pickle=False)
    numpy.save(directory + "/testFeatures.npy", testFeatures, allow_pickle=False)
    numpy.save(directory + "/testLabels.npy", testLabels, allow_pickle=False)

# get numpy arrays containing features and labels
rawFeatures, rawLabels = getRawDataFromInternet()
#rawFeatures, rawLabels = getRawDataFromFiles()

# print some interesting stats about the data
print("%d raw features total" % len(rawFeatures))
print("%d raw labels total" % len(rawLabels))
print()

# separate data into separate training and testing arrays
rawTrainFeatures, rawTrainLabels, rawTestFeatures, rawTestLabels = partitionData(rawFeatures, rawLabels, 10000/70000)

# print some interesting stats about the partitioned data
print("unprocessed feature/label data:")
print("%d train feature samples" % rawTrainFeatures.shape[0])
print("%d train label samples" % rawTrainLabels.shape[0])
print("%d test feature samples" % rawTestFeatures.shape[0])
print("%d test label samples" % rawTestLabels.shape[0])
print("%d features per sample" % (1 if len(rawTrainFeatures.shape) == 1 else rawTrainFeatures.shape[1]))
print("%d labels per sample" % (1 if len(rawTrainLabels.shape) == 1 else rawTrainLabels.shape[1]))
print("%d min feature value" % min(rawTrainFeatures.min(), rawTestFeatures.min()))
print("%d max feature value" % max(rawTrainFeatures.max(), rawTestFeatures.min()))
print("%d min label value" % min(rawTrainLabels.min(), rawTestLabels.min()))
print("%d max label value" % max(rawTrainLabels.max(), rawTestLabels.max()))
print()

# perform necessary transformations on feature and label data
# 1. normalize feature data to a max of 1 and convert to float
# 2. one-hot vectorize label data and convert to float
# 3. randomize the order of the training data
divisor = max(rawTrainFeatures.max(), rawTestFeatures.max())
trainFeatures = normalizeArray(rawTrainFeatures, divisor)
testFeatures = normalizeArray(rawTestFeatures, divisor)

vectorWidth = max(rawTrainLabels.max(), rawTestLabels.max()) - min(rawTrainLabels.min(), rawTestLabels.min()) + 1
trainLabels = vectorizeArray(rawTrainLabels, vectorWidth)
testLabels = vectorizeArray(rawTestLabels, vectorWidth)

trainFeatures, trainLabels = randomizeTrainingData(trainFeatures, trainLabels)

# print some interesting stats about the partitioned data
print("processed feature/label data:")
print("%d train feature samples" % trainFeatures.shape[0])
print("%d train label samples" % trainLabels.shape[0])
print("%d test feature samples" % testFeatures.shape[0])
print("%d test label samples" % testLabels.shape[0])
print("%d features per sample" % (1 if len(trainFeatures.shape) == 1 else trainFeatures.shape[1]))
print("%d labels per sample" % (1 if len(trainLabels.shape) == 1 else trainLabels.shape[1]))
print("%d min feature value" % min(trainFeatures.min(), testFeatures.min()))
print("%d max feature value" % max(trainFeatures.max(), testFeatures.min()))
print("%d min label value" % min(trainLabels.min(), testLabels.min()))
print("%d max label value" % max(trainLabels.max(), testLabels.max()))
print()

print("sample data:")
print("features:", trainFeatures[0])
print("labels:", trainLabels[0])

# saved processed data for model training phase
saveProcessedDataToFiles(trainFeatures, trainLabels, testFeatures, testLabels)

# visualize sample data
print("please close graphical display to continue...")
plt.imshow(trainFeatures[0].reshape(28,28), interpolation="nearest")
plt.axis("off")
plt.show()
