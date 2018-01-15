import numpy
import tensorflow as tf
import random
import os
import sys
import math
import time

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

def tensorboardVariableSummaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    #    with tf.name_scope('stddev'):
    #        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    #    tf.summary.scalar('stddev', stddev)
    #    tf.summary.scalar('max', tf.reduce_max(var))
    #    tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

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

# we are gonna train with random subsets of training data, to speed up model optimization (1000 images/batch)
numFeatureBatches = 25

# how wide is each layer of our neural network
featureVectorLength = len(trainFeatures[0])
intermediateLayer1VectorLength = 200
intermediateLayer2VectorLength = 100
intermediateLayer3VectorLength = 60
intermediateLayer4VectorLength = 30
labelVectorLength = len(trainLabels[0])

# create the TensorFlow feature tensor
# (None is used because training and testing data are of different sizes)
features = tf.placeholder(tf.float32, [None, featureVectorLength], name='features')

# create the TensorFlow inner/hidden layer tensors
innerLayer1 = tf.placeholder(tf.float32, [featureVectorLength, intermediateLayer1VectorLength], name='innerLayer1')
innerLayer2 = tf.placeholder(tf.float32, [intermediateLayer1VectorLength, intermediateLayer2VectorLength], name='innerLayer2')
innerLayer3 = tf.placeholder(tf.float32, [intermediateLayer2VectorLength, intermediateLayer3VectorLength], name='innerLayer3')
innerLayer4 = tf.placeholder(tf.float32, [intermediateLayer3VectorLength, intermediateLayer4VectorLength], name='innerLayer4')

# create the TensorFlow label tensor
# (None is used because training and testing data are of different sizes)
labels = tf.placeholder(tf.float32, [None, labelVectorLength], name='labels')

# each weight initialized to random and will be adjusted as algorithm learns.  
weights1 = tf.Variable(tf.truncated_normal([featureVectorLength, intermediateLayer1VectorLength]), name='weights1')
weights2 = tf.Variable(tf.truncated_normal([intermediateLayer1VectorLength, intermediateLayer2VectorLength]), name='weights2')
weights3 = tf.Variable(tf.truncated_normal([intermediateLayer2VectorLength, intermediateLayer3VectorLength]), name='weights3')
weights4 = tf.Variable(tf.truncated_normal([intermediateLayer3VectorLength, intermediateLayer4VectorLength]), name='weights4')
weights5 = tf.Variable(tf.truncated_normal([intermediateLayer4VectorLength, labelVectorLength]), name='weights5')
    
# each bias initialized to small (non-zero) number and will be adjusted as algorithm learns.
biases1 = tf.Variable(tf.ones([intermediateLayer1VectorLength])/10, name='biases1')
biases2 = tf.Variable(tf.ones([intermediateLayer2VectorLength])/10, name='biases2')
biases3 = tf.Variable(tf.ones([intermediateLayer3VectorLength])/10, name='biases3')
biases4 = tf.Variable(tf.ones([intermediateLayer4VectorLength])/10, name='biases4')
biases5 = tf.Variable(tf.ones([labelVectorLength])/10, name='biases5')
    
# define how our predicted outputs are calculated using our model
innerLayer1 = tf.nn.relu(tf.matmul(features, weights1) + biases1)
innerLayer2 = tf.nn.relu(tf.matmul(innerLayer1, weights2) + biases2)
innerLayer3 = tf.nn.relu(tf.matmul(innerLayer2, weights3) + biases3)
innerLayer4 = tf.nn.relu(tf.matmul(innerLayer3, weights4) + biases4)
predictedLogits = tf.matmul(innerLayer4, weights5) + biases5
predictedLabels = tf.nn.softmax(predictedLogits)

# cross entropy tells us how well our predicted probability distribution matches the true probability distribution (e.g. correct answers)
crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predictedLogits, labels = labels))

# create a learning rate tensor so we can decay learning rate during training    
learningRate = tf.placeholder(tf.float32, name='learningRate')

# create tensors for trainAccuracy and testAccuracy so we can store in tensorboard
trainAccuracy = tf.placeholder(tf.float32, name='trainAccuracy')
testAccuracy = tf.placeholder(tf.float32, name='testAccuracy')
    
# track data for TensorBoard
with tf.name_scope('weights1'):
    tensorboardVariableSummaries(weights1)
with tf.name_scope('weights2'):
    tensorboardVariableSummaries(weights2)
with tf.name_scope('weights3'):
    tensorboardVariableSummaries(weights3)
with tf.name_scope('weights4'):
    tensorboardVariableSummaries(weights4)
with tf.name_scope('weights5'):
    tensorboardVariableSummaries(weights5)
with tf.name_scope('biases1'):
    tensorboardVariableSummaries(biases1)
with tf.name_scope('biases2'):
    tensorboardVariableSummaries(biases2)
with tf.name_scope('biases3'):
    tensorboardVariableSummaries(biases3)
with tf.name_scope('biases4'):
    tensorboardVariableSummaries(biases4)
with tf.name_scope('biases5'):
    tensorboardVariableSummaries(biases5)
with tf.name_scope('crossEntropy'):
    tensorboardVariableSummaries(crossEntropy)
with tf.name_scope('learningRate'):
    tensorboardVariableSummaries(learningRate)
with tf.name_scope('trainAccuracy'):
    tensorboardVariableSummaries(trainAccuracy)
with tf.name_scope('testAccuracy'):
    tensorboardVariableSummaries(testAccuracy)

print("training model...")

# we want to be able to save the results of our trained model
modelSaver = tf.train.Saver()

# we need a session in which we can execute our model
session = tf.InteractiveSession()

# prepare to save TensorBoard metrics to files
mergedSummaryData = tf.summary.merge_all()
directoryPath = os.getcwd() + "\\tensorBoard"
trainWriter = tf.summary.FileWriter(directoryPath, session.graph, filename_suffix="training")

# perform the algorithm training on random sub batches of training data each iteration
maxTrainingIterations = 30001
maxLearningRate = 0.004
minLearningRate = 0.0001
learningRateDecaySpeed = maxTrainingIterations
accuracyGoal = .99

training = tf.train.AdamOptimizer(maxLearningRate).minimize(crossEntropy)

# we need to initialize variables we defined before running the model (but after defining our optimizer since it also has variables)
initialize = tf.global_variables_initializer()
session.run(initialize)

trainingTimeHours = 0.0

for count in range(0, maxTrainingIterations):
    startTrainingTime = time.time()
    
    learningRateValue = minLearningRate + (maxLearningRate - minLearningRate) * math.exp(-count/learningRateDecaySpeed) 
    batchTrainFeatures, batchTrainLabels = randomTrainingDataSubset(trainFeatures, trainLabels, numFeatureBatches)
    session.run(training, feed_dict={features: batchTrainFeatures, labels: batchTrainLabels, learningRate: learningRateValue})
            
    endTrainingTime = time.time()
    trainingTimeHours += (endTrainingTime - startTrainingTime)/(60.0*60.0)
        
    if count%100 == 0:    
        # check the accuracy of our model (on test data) as we iterate      
        correctPrediction = tf.equal(tf.argmax(predictedLabels,1), tf.argmax(labels,1))
        
        trainDataAccuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        trainDataAccuracy = session.run(trainDataAccuracy, feed_dict={features: trainFeatures, labels: trainLabels})
        
        testDataAccuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        testDataAccuracy = session.run(testDataAccuracy, feed_dict={features: testFeatures, labels: testLabels})
        
        # save TensorBoard data
        trainSummary = session.run(mergedSummaryData, feed_dict={features: batchTrainFeatures, labels: batchTrainLabels, learningRate: learningRateValue, trainAccuracy: trainDataAccuracy, testAccuracy: testDataAccuracy})
        trainWriter.add_summary(trainSummary, count)
        
        print("iteration %4d: training accuracy=%5.4f, testing accuracy=%5.4f, learningRate=%5.4f, trainingTimeHours=%5.2f" % (count, trainDataAccuracy, testDataAccuracy, learningRateValue, trainingTimeHours))
        
        # quit training if we think we are sufficiently accurate
        if testDataAccuracy >= accuracyGoal:
            print("test data accuracy %5.4f meets our goal of %5.4f, terminating training" % (testDataAccuracy, accuracyGoal))
            break 

trainWriter.close()

# save trained model to disk
directory = "modelExport"
if not os.path.exists(directory):
    os.makedirs(directory)
modelSaver.save(session, 'modelExport/MNISTmodel')

session.close()

