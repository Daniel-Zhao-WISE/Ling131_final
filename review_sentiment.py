import nltk
import string
import pickle
import numpy as np
import os
from nltk.corpus import stopwords
from sklearn.model_selection import KFold


class ReviewSentiment:
    def __init__(self, labeled_data, features=[], train_set=[], test_set=[], train_size=1000):
        self.stop_words = set(stopwords.words('english'))
        self.negation_words = {'no', 'not', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never', 'hardly',
                               'scarcely', 'barely', "n't"}
        self.labeled_data = labeled_data
        self.features = features
        self.train_set = train_set
        self.test_set = test_set
        self.train_size = train_size
        if not self.features:
            self._create_feature_sets()

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
        return [word1 + ' ' + word2 for word1, word2 in nltk.bigrams(unigram_raw)]

    def generate_pos(self, raw):
        PoS_raw = []
        for sent in nltk.sent_tokenize(raw):
            negation = ''
            for word, PoS in nltk.pos_tag(nltk.word_tokenize(sent)):
                if word not in self.stop_words:
                    if word not in string.punctuation:
                        PoS_raw.append((negation + word, PoS))
                if word.lower() in self.negation_words:
                    negation = 'NEG-'
        return PoS_raw

    def _generate_features(self):
        unigram_raw = []
        bigrams_raw = []
        for unigram in [self.generate_unigram(raw) for raw, sentiment in self.labeled_data]:
            unigram_raw.extend(unigram)
            bigrams_raw.extend(self.generate_bigrams(unigram))
        self.features.append(
            [word for word, times in nltk.FreqDist(unigram_raw).most_common() if times > 5])
        print('unigram # of features: %d' % (len(self.features[0])))
        self.features.append([word for word, times in nltk.FreqDist(bigrams_raw).most_common() if times > 3])
        print('bigrams # of features: %d' % (len(self.features[1])))
        unigram_bigrams = self.features[0].copy()
        unigram_bigrams.extend(self.features[1])
        self.features.append(unigram_bigrams)
        print('unigram + bigrams # of features: %d' % (len(self.features[2])))
        PoS_raw = [item
                   for raw, sentiment in self.labeled_data
                   for item in self.generate_pos(raw)]
        self.features.append(
            [word + '-' + PoS for (word, PoS), times in nltk.FreqDist(PoS_raw).most_common() if times > 5])
        print('unigram + PoS # of features: %d' % (len(self.features[3])))
        adj_raw = [word for word, PoS in PoS_raw if PoS == 'JJ']
        self.features.append([word for word, times in nltk.FreqDist(adj_raw).most_common() if times > 5])
        print('adj # of features: %d' % (len(self.features[4])))
        self.features.append([word for word, times in nltk.FreqDist(unigram_raw).most_common(len(self.features[4]))])
        print('most%d # of features: %d' % (len(self.features[4]), len(self.features[5])))

    def _create_feature_sets(self):
        # create feature sets
        if not self.features:
            self._generate_features()

        feature_sets = [[] for i in range(6)]

        for raw, sentiment in self.labeled_data:
            feature_set = self.mr_features(raw)
            for i in range(6):
                feature_sets[i].append((feature_set[i], sentiment))

        for i in range(6):
            self.train_set.append(feature_sets[i][:self.train_size])
            self.test_set.append(feature_sets[i][self.train_size:])

    def mr_features(self, instance):
        unigram = self.generate_unigram(instance)
        bigrams = set(self.generate_bigrams(unigram))
        unigram = set(unigram)
        unigram_bigrams = set(unigram)
        unigram_bigrams.update(bigrams)
        unigram_pos = set(self.generate_pos(instance))
        feature_data = [unigram, bigrams, unigram_bigrams,
                        {word + '-' + PoS for word, PoS in unigram_pos},
                        {word for word, PoS in unigram_pos if PoS == 'JJ'},
                        unigram]
        return [self._create_features(i, feature_data) for i in range(6)]

    def _create_features(self, i, feature_data):
        features = {}
        for token in self.features[i]:
            if token in feature_data[i]:
                features[token] = 1
            else:
                features[token] = 0
        return features

    def train_classifier(self, classifier, i, n):
        # create the classifier
        kf = KFold(n_splits=n)
        sum = 0
        classifiers = Classifiers(n)
        index = 0
        for train, test in kf.split(self.train_set[i]):
            train_data = np.array(self.train_set[i])[train]
            test_data = np.array(self.train_set[i])[test]
            classifiers.set_classifer(index, classifier.train(train_data))
            sum += nltk.classify.accuracy(classifiers.get_classifer(index), test_data)
            index += 1
        average = sum / 3
        print('Average %d-fold cross validation accuracy of the model of case %d is %.2f'
              % (n, i + 1, 100 * average))
        return classifiers

    def evaluate_classifer(self, classifiers, i):
        sum = 0
        for classifier in classifiers.classifiers:
            sum += nltk.classify.accuracy(classifier, self.test_set[i])
        average = sum / 3
        print('Average accuracy of the model on test_set of case %d on test_set is %.2f'
              % (i + 1, 100 * average))

    def predict(self, raw, classifiers, i):
        classifiers.classify(self.mr_features(raw)[i])

    def save(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + 'features.dat', 'wb') as f:
            pickle.dump(self.features, f, True)
        with open(dir + 'train_set.dat', 'wb') as f:
            pickle.dump(self.train_set, f, True)
        with open(dir + 'test_set.dat', 'wb') as f:
            pickle.dump(self.test_set, f, True)


class Classifiers:
    def __init__(self, n):
        self.classifiers = [None for i in range(n)]

    def set_classifer(self, i, classifier):
        self.classifiers[i] = classifier

    def get_classifer(self, i):
        return self.classifiers[i]

    def classify(self, feature_set):
        print('Respective classify results for each classifier:', ' ')
        pos = 0
        neg = 0
        for classifier in self.classifiers:
            prediction = classifier.classify(feature_set)
            print(prediction, '\t')
            if prediction == 'pos':
                pos += 1
            else:
                neg += 1
        if pos > neg:
            sentiment = 'positive'
        else:
            sentiment = 'negative'
        print()
        print('model prediction: ' + sentiment)
