import nltk
import string
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords


class ReviewSentiment():
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.negation_words = {'no', 'not', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never', 'hardly',
                               'scarcely', 'barely', "n't"}
        self.labeled_data = []
        self.unigram = []
        self.bigrams = []
        self.test_set = []
        self.train_set = []

    def generate_unigram(self, raw):
        unigram_raw = []
        for sent in nltk.sent_tokenize(raw):
            negation = ''
            for word in nltk.word_tokenize(sent):
                if word not in self.stop_words:
                    if word not in string.punctuation:
                        unigram_raw.append(negation + word)
                if word.lower() in self.negation_words:
                    negation = 'NEG-'
        return unigram_raw

    def generate_bigrams(self, unigram_raw):
        return nltk.bigrams(unigram_raw)

    def _create_labeled_data(self):
        # create labeled data
        self.labeled_data = [(movie_reviews.raw(fileids=fileid), movie_reviews.categories(fileid)[0])
                             for fileid in movie_reviews.fileids()]

    def _create_feature_sets(self):
        # create feature sets
        self._create_labeled_data()
        unigram_raw = [unigram
                       for raw, category in self.labeled_data
                       for unigram in self.generate_unigram(raw)]
        self.unigram = [word for word, times in nltk.FreqDist(unigram_raw).most_common() if times > 5]
        feature_sets = [(self.wsd_features_unigram(raw), sentiment)
                        for raw, sentiment in self.labeled_data]
        half = len(feature_sets) // 2
        random.shuffle(feature_sets)
        self.train_set = feature_sets[:half]
        self.test_set = feature_sets[half:]

    def wsd_features_unigram(self, raw):
        features = {}
        tokens = set(self.generate_unigram(raw))
        for token in self.unigram:
            if token in tokens:
                features[token] = 1
            else:
                features[token] = 0
        return features

    def train_classifier(self):
        # create the classifier
        self._create_feature_sets()
        return nltk.NaiveBayesClassifier.train(self.train_set)

    def evaluate_classifier(self, classifier):
        # get the accuracy and print it
        print('Accuracy of the model on test_set is %.2f' % (nltk.classify.accuracy(classifier, self.test_set)))
        print()
'''
    def run_classifier(self, classifier):
        emma = nltk.corpus.gutenberg.sents('austen-emma.txt')
        instances = [(" ".join(sentence), make_instance(nltk.pos_tag(sentence))) for sentence in emma if 'interest' in sentence]
        predictions = [(instance[0], classifier.classify(wsd_features(instance[1]))) for instance in instances]
        print("Total number of satisfied records in 'austen-emma.txt' is %d" % (len(predictions)))
        print()
        print('Prediction results:')
        for prediction in predictions:
            print("\t%s | %s" % (prediction[1], prediction[0]))
'''

if __name__ == '__main__':
    rs = ReviewSentiment()
    classifier = rs.train_classifier()
    rs.evaluate_classifier(classifier)