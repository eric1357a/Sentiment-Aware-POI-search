import get_twitter_data
import naive_bayes_classifier
import json,sys,pickle

keyword = 'bitcoin'
time = 'today'
twitterData = get_twitter_data.TwitterData()
tweets = twitterData.getTwitterData(keyword, time)


trainingDataFile = 'data/full_training_dataset.csv'
# print "tweets : %s" % (tweets)
classifierDumpFile = 'data/test/naivebayes_test_model.pickle'
trainingRequired = 0
print 'Started to instantiate Naive Bayes Classifier'
sys.stdout.flush()

nb = naive_bayes_classifier.NaiveBayesClassifier(tweets, keyword, time,\
                                  trainingDataFile, classifierDumpFile, trainingRequired)
    #nb.classify()
print 'Classifying naive bayes'
nb.classify()
# print 'Computing Accuracy naive bayes'
# nb.accuracy()
html = nb.getHTML()
f = open("/var/www/html/FYP/result.html", "w+")
f.write(html.encode('utf-8').strip())
f.close
