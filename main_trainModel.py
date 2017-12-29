import numpy
import tensorflow as tf
import random
import os
import sys

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

def randomTrainingDataSubset(trainFeatures, trainLabels, numbSubsets):
        numbFeatures = len(trainFeatures)
        featuresPerSubset = int(numbFeatures/numbSubsets)
        index = random.randint(0, numbSubsets - 1)
        startIndex = index * featuresPerSubset
        endIndex = startIndex + featuresPerSubset
        return trainFeatures[startIndex:endIndex], trainLabels[startIndex:endIndex]

# get processed numpy arrays containing features and labels
trainFeatures, trainLabels, testFeatures, testLabels = getProcessedDataFromFiles()

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

print("constructing model...")

# we are gonna train with random subsets of training data, to speed up gradient descent
numFeatureBatches = 20

featureVectorLength = len(trainFeatures[0])
labelVectorLength = len(trainLabels[0])

# create the TensorFlow feature tensor
# (None is used because training and testing data are of different sizes)
features = tf.placeholder(tf.float32, [None, featureVectorLength], name='features')

# create the TensorFlow label tensor
# (None is used because training and testing data are of different sizes)
labels = tf.placeholder(tf.float32, [None, labelVectorLength], name='labels')

# each weight initialized to zero and will be adjusted as algorithm learns.  
weights = tf.Variable(tf.zeros([featureVectorLength, labelVectorLength]), name='weights')

# each bias initialized to zero and will be adjusted as algorithm learns.
biases = tf.Variable(tf.zeros([labelVectorLength], name='biases'))
    
# define how our predicted outputs are calculated using our model
predictedLabels = tf.nn.softmax(tf.matmul(features, weights) + biases)

# cross entropy tells us how well our predicted probability distribution matches the true probability distribution (e.g. correct answers)
# (1e-10 us added to prevent NANs)
crossEntropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(predictedLabels + 1e-10), axis=[1]))

print("training model...")

# we want to be able to save the results of our trained model
modelSaver = tf.train.Saver()

# we need a session in which we can execute our model
session = tf.InteractiveSession()

# we need to initialize variables we defined before running the model
tf.global_variables_initializer().run()

# perform the algorithm training (need to batch for really large data sets but we aren't here)
learningRate = 0.5
accuracyGoal = .90
for count in range(0, 1001):
    training = tf.train.GradientDescentOptimizer(learningRate).minimize(crossEntropy)
    batchTrainFeatures, batchTrainLabels = randomTrainingDataSubset(trainFeatures, trainLabels, numFeatureBatches)
    session.run([training], feed_dict={features: batchTrainFeatures, labels: batchTrainLabels})
    
# check the accuracy of our model (on test data) as we iterate
    if count%25 == 0:
        correctPrediction = tf.equal(tf.argmax(predictedLabels,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        accuracy = session.run(accuracy, feed_dict={features: testFeatures, labels: testLabels})
        print("iteration %4d: training accuracy = %5.4f" % (count, accuracy))
        
# quit training if we think we are sufficiently accurate
        if accuracy >= accuracyGoal:
            print("accuracy %5.4f meets our goal of %5.4f, terminating training" % (accuracy, accuracyGoal))
            break 

# save trained model to disk
directory = "modelExport"
if not os.path.exists(directory):
    os.makedirs(directory)
modelSaver.save(session, 'modelExport/MNISTmodel')

session.close()

