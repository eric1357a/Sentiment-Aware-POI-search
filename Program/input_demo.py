import naive_bayes_classifier

sentence = raw_input("Please enter a sentence for classifying : ")

data = {0: [sentence]}

trainingDataFile = 'data/full_training_dataset.csv'
classifierDumpFile = 'data/test/naivebayes_test_model.pickle'
trainingRequired = 0
print 'Started to instantiate Naive Bayes Classifier'

nb = naive_bayes_classifier.NaiveBayesClassifier(data, "no", "today",\
                                  trainingDataFile, classifierDumpFile, trainingRequired)
print 'Classifying naive bayes'
nb.classify()