import re       
import os       
import functools
import numpy as np
import pandas as pd
import seaborn as sns
from nntplib import NNTP
from sklearn import tree
from pydoc import describe
from sklearn.svm import SVC
from sysconfig import get_path
from string import punctuation
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from factor_analyzer import FactorAnalyzer  # pip install factor_analyzer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo


############################################################################################################################################
## Get Data 
# Raw Data reader 
# Store all in csv file
############################################################################################################################################
def get_paths(path=False, ftypes=False):
    if path:
        result = [i for i in os.path(path) if ftypes == i[-4:] ]
    else:
        cwd = os.path.dirname(os.path.realpath(__file__))
        folder = 'reviews'
        dir_path = cwd + '\\' + folder + '\\'
    paths = os.listdir(dir_path)
    L = [(dir_path + i) for i in paths if ftypes == i[-3:] ]
    return L

def read_file(path):
    result = ''
    with open(path, 'r', encoding='utf8') as file:
        result = file.read()
    return result 

def parse_review(content, delimiter=''):
    content = content.split('Review:')
    result = []
    if delimiter[-2:] != ':\n':
        delimiter += ':\n'
    for i in content:
        rating = re.findall(r'Rating:\n(\d)*', i)
        text = re.findall(r'text:\n(.*)', i)
        if rating and text:
            text = text[0].strip()
            if text:
                tup = (rating[0], '\"' + text.replace('\"', "\'") + '\"')
                result.append(tup) 
    return result

def save_to_csv(path, D):
    with open(path, 'w', encoding='utf-8') as file:
        columns =  'rating,review_text\n'
        file.write(columns)
        for n in D:
            for k in range(len(D[n])):
                input = n + ',' + D[n][k] + '\n'
                file.write(input)
    return

def convert_store_raw_data(out_path, count=0):
    paths = get_paths(ftypes='txt')
    D = {'1':[], '2':[], '3':[], '4':[], '5':[]}
    for p in paths:
        content = read_file(p)
        reviews = parse_review(content, 'Rating')
        for r in reviews:
            if len(D[r[0]]) < count or count == 0:
                D[r[0]].append(r[1])
    # for i in D:
    #     print(i, D[i][0])
    save_to_csv(out_path, D)
    return 

############################################################################################################################################
## Clean Text 
# Normalizing case
# Remove extra line breaks
# Tokenize
# Remove stop words and punctuations
############################################################################################################################################
# def normalize_case():
#     reviews_list = reviews1.review_text.values
#     len(reviews_list)
#     reviews_lower = [txt.lower() for txt in reviews_list]
#     return 

# #### Tokenize
# def tokenize():
#     print(word_tokenize(reviews_lower[0]))
#     reviews_tokens = [word_tokenize(sent) for sent in reviews_lower]
#     print(reviews_tokens[0])
#     return

# ### Remove stop words and punctuations
# stop_nltk = stopwords.words("english")
# stop_punct = list(punctuation)
# print(stop_nltk)
# stop_nltk.remove("no")
# stop_nltk.remove("not")
# stop_nltk.remove("don")
# stop_nltk.remove("won")
# "no" in stop_nltk
# stop_final = stop_nltk + stop_punct + ["...", "``","''", "====", "must"]
# def del_stop(sent):
#     return [term for term in sent if term not in stop_final]
# del_stop(reviews_tokens[1])
# reviews_clean = [del_stop(sent) for sent in reviews_tokens]
# reviews_clean = [" ".join(sent) for sent in reviews_clean]
# reviews_clean[:2]




############################################################################################################################################
## Choose Features
# Select 'Vanilla' Features 
# Set whether Sentiment Fatures or not
# Set Ratings for Testing { (0,1), (1,2,3), (1,2,3,4,5),... }
############################################################################################################################################
def one_hot(df, columns):
    for i in columns:
        # df[i] = df[i].str.replace(r'unknown', 'unkn', regex=True)
        if i not in ['month', 'day_of_week']:
            df[i] = df[i].apply(lambda x: i + '_' + x)
    # Create one-hot encoding of the different categories.
    ohe = OneHotEncoder()
    feature_array = ohe.fit_transform(df[columns]).toarray()
    feature_labels = functools.reduce(lambda a,b : a+b, [list(i) for i in ohe.categories_])
    ndf = pd.DataFrame(feature_array, columns=feature_labels)
    df = pd.concat([df, ndf], axis=1)
    return df

def ordinal_encoder(df, columns):
    for key in columns:
        df[key] = df[key].map(columns[key])
        df[key] = df[key].fillna(0)
    return df

def select_columns(df, selected: list):
    if not selected:
        return df
    L_DF = []
    for s in selected:
        if type(s) is int:
            L_DF.append(df.iloc[:, [s] ])
        elif type(s) is str:
            L_DF.append(df.loc[:, [s] ])
        elif type(s) is tuple:
            A = s[0]
            B = s[1]
            if type(A) is str:
                A = df.columns.get_loc(A)
            if type(B) is str:
                B = df.columns.get_loc(B)
            if A == 0:
                span = df.iloc[:, : B ] 
            elif B == 0:
                span = df.iloc[:, A : ] 
            else:
                span = df.iloc[:, A : B ] 
            L_DF.append(span)
        else:
            return df
    df = pd.concat(L_DF, axis=1 )
    return df

def select_rows_by_rating(df, num_selected, selected=[1,2,3,4,5]):
    # color_or_shape = df.loc[(df['Color'] == 'Green') | (df['Shape'] == 'Rectangle')]



    return 



############################################################################################################################################
## Descriptive/Exploratory Data Analysis 
# Document term matrix using TfIdf
############################################################################################################################################


############################################################################################################################################
## Machine Learning Models
# Separate X and Y and perform train test split, { (70-30), (80,20) }
# Parameter Tuning 
# Neural Network -- MLP, RNN, CNN
# Transformer Model
# Support Vector Machine
# Random Forest
# Logistic Regression
# KNN
# Naive Bayes
# QDA (Quadratic Discriminate Analysis)
############################################################################################################################################


############################################################################################################################################
## Using the best estimator to make predictions on the test set
############################################################################################################################################


############################################################################################################################################
## Identifying mismatch cases
############################################################################################################################################




############################################################################################################################################
## Main
############################################################################################################################################
if __name__ == "__main__":
    path_all = r'all_reviews.csv'
    path_all_valid = r'all_valid_reviews.csv'
    path_450K = r'450K_reviews.csv'
    # convert_store_raw_data(path_450K, 90_000)
    df = pd.read_csv(path_450K)
    x = df['rating'].value_counts()
    print(x)

    print('Ready for next step...')







