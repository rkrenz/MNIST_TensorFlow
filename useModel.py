import numpy
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

def getProcessedDataFromFiles():
    # fetch processed MNIST data from previously saved files
    print("fetching processed MNIST data from files...")
    print()
    try:
        directory = 'processedData'
        trainFeatures = numpy.load(directory + "/trainFeatures.npy", mmap_mode=None, allow_pickle=False)
        trainLabels = numpy.load(directory + "/trainLabels.npy", mmap_mode=None, allow_pickle=False)
        testFeatures = numpy.load(directory + "/testFeatures.npy", mmap_mode=None, allow_pickle=False)
        testLabels = numpy.load(directory + "/testLabels.npy", mmap_mode=None, allow_pickle=False)
        return trainFeatures, trainLabels, testFeatures, testLabels
    except:
        print("ERROR: no processed data found.  did you run main_dataPrep first?")
        sys.exit(1)
        
def randomizeTestingData(testFeatures, testLabels):
    assert len(testFeatures) == len(testLabels)
    permutation = numpy.random.permutation(len(testFeatures))
    return testFeatures[permutation], testLabels[permutation]
        
# get processed numpy arrays containing features and labels (test data only)
_, _, testFeatures, testLabels = getProcessedDataFromFiles()    

# randomize test data
testFeatures, testLabels = randomizeTestingData(testFeatures, testLabels)

print("constructing model from saved file...")

# we need a session in which we can execute our model
session = tf.InteractiveSession()

# read the model files to create the model
directory = 'modelExport'
modelSaver = tf.train.import_meta_graph(directory + '/MNISTmodel.meta')
modelSaver.restore(session, tf.train.latest_checkpoint(directory + '/'))

# get the model graph object
graph = tf.get_default_graph()

# get some tensor handles
#for op in graph.get_operations():
#    print(str(op.name))
features = graph.get_tensor_by_name("features:0")
predictedLabels = graph.get_tensor_by_name("Softmax:0")

# give test data to model
feed_dictionary = {features: testFeatures}

print("processing test data...")

modelOutput = tf.argmax(session.run(predictedLabels, feed_dictionary), 1).eval()

session.close()

# see how the model performed
for count in range(len(testFeatures)):
    print("model says: %d" % modelOutput[count])
    plt.imshow(testFeatures[count].reshape(28,28), interpolation="nearest")
    plt.axis("off")
    plt.show()

print("done!")