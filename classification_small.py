import nltk
import random
from nltk.corpus import movie_reviews
from sklearn.svm import LinearSVC
from review_sentiment import ReviewSentiment
import classification


if __name__ == '__main__':
    labeled_data = [(movie_reviews.raw(fileids=fileid), movie_reviews.categories(fileid)[0])
                    for fileid in movie_reviews.fileids()]
    random.seed(1234)
    random.shuffle(labeled_data)
    labeled_data = labeled_data[:100]
    rs = ReviewSentiment(labeled_data, train_size=50)
    classification.evaluate(rs, classification.train(rs))
