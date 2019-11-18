import pandas as pd
import numpy as np

yelp_reviews = pd.read_csv("/content/drive/My Drive/Colab Datasets/yelp_review_short.csv")

bins = [0,2,5]
review_names = ['negative', 'positive']

yelp_reviews['reviews_score'] = pd.cut(yelp_reviews['stars'], bins, labels=review_names)

yelp_reviews.head()