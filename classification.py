import nltk
import pickle
from sklearn.svm import LinearSVC
from review_sentiment import ReviewSentiment

if __name__ == '__main__':
    try:
        with open('cache/review_sentiment/labeled_data.dat', 'rb') as f:
            labeled_data = pickle.load(f)
        with open('cache/review_sentiment/features.dat', 'rb') as f:
            features = pickle.load(f)
        with open('cache/review_sentiment/train_set.dat', 'rb') as f:
            train_set = pickle.load(f)
        with open('cache/review_sentiment/test_set.dat', 'rb') as f:
            test_set = pickle.load(f)
        rs = ReviewSentiment(labeled_data, features, train_set, test_set)
    except FileNotFoundError:
        rs = ReviewSentiment()
        rs.save()

    try:
        with open('cache/classifier/NB.dat', 'rb') as f:
            NB_classifiers = pickle.load(f)
        with open('cache/classifier/SVM.dat', 'rb') as f:
            SVM_classifiers = pickle.load(f)
    except IOError:
        print()
        print('Naive Bayes')
        NB_classifiers = [rs.train_classifier(nltk.NaiveBayesClassifier, i, 3) for i in range(6)]
        print()
        print('SVM')
        SVM_classifiers = [rs.train_classifier(nltk.classify.SklearnClassifier(LinearSVC()), i, 3) for i in range(6)]
        print()
        with open('cache/classifier/NB.dat', 'wb') as f:
            pickle.dump(NB_classifiers, f, True)
        with open('cache/classifier/SVM.dat', 'wb') as f:
            pickle.dump(SVM_classifiers, f, True)

    print('Naive Bayes test accuracy')
    for i, classifier in enumerate(NB_classifiers):
        rs.evaluate_classifer(classifier, i)
    print()
    print("SVM test accuracy")
    for i, classifier in enumerate(SVM_classifiers):
        rs.evaluate_classifer(classifier, i)
