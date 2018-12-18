import nltk
import pickle
import random
import os
from nltk.corpus import movie_reviews
from sklearn.svm import LinearSVC
from review_sentiment import ReviewSentiment


def train(rs):
    print()
    print('Naive Bayes')
    nb_classifiers = [rs.train_classifier(nltk.NaiveBayesClassifier, i, 3) for i in range(6)]
    print()
    print('SVM')
    svm_classifiers = [rs.train_classifier(nltk.classify.SklearnClassifier(LinearSVC()), i, 3) for i in range(6)]
    print()
    return [nb_classifiers, svm_classifiers]


def evaluate(rs, classifiers):
    print('Naive Bayes test accuracy')
    for i, classifier in enumerate(classifiers[0]):
        rs.evaluate_classifer(classifier, i)
    print()
    print("SVM test accuracy")
    for i, classifier in enumerate(classifiers[1]):
        rs.evaluate_classifer(classifier, i)


if __name__ == '__main__':
    labeled_data = [(movie_reviews.raw(fileids=fileid), movie_reviews.categories(fileid)[0])
                    for fileid in movie_reviews.fileids()]
    random.seed(1234)
    random.shuffle(labeled_data)
    try:
        with open('cache/data/features.dat', 'rb') as f:
            features = pickle.load(f)
        with open('cache/data/train_set.dat', 'rb') as f:
            train_set = pickle.load(f)
        with open('cache/data/test_set.dat', 'rb') as f:
            test_set = pickle.load(f)
        rs = ReviewSentiment(labeled_data, features, train_set, test_set)
    except FileNotFoundError:
        rs = ReviewSentiment(labeled_data)
        rs.save('cache/data/')

    try:
        classifiers = []
        with open('cache/classifier/NB.dat', 'rb') as f:
            classifiers.append(pickle.load(f))
        with open('cache/classifier/SVM.dat', 'rb') as f:
            classifiers.append(pickle.load(f))
    except IOError:
        classifiers = train(rs)
        dir = 'cache/classifier/'
        if not os.path.exists():
            os.makedirs(dir)
        with open(dir + 'NB.dat', 'wb') as f:
            pickle.dump(classifiers[0], f, True)
        with open(dir + 'SVM.dat', 'wb') as f:
            pickle.dump(classifiers[1], f, True)
    evaluate(rs, classifiers)
