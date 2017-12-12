import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from datetime import datetime


# Function to read data
def read_data():
  
    data = pd.read_csv('deals_tweet_2016.csv')
    df = pd.DataFrame(data)
# Excluding the other categories
    df = df[df['category'] != 'Other']
    return df
    
def create_ind_var(df, lb):
    
# Transforming the independent variables and creating an Input X
    new_month = lb.fit_transform(df['month'])
    new_month = pd.DataFrame(new_month, columns=lb.classes_, index=df.index)

    new_day = lb.fit_transform(df['day'])
    new_day = pd.DataFrame(new_day, columns=['day'], index=df.index)

    new_date = lb.fit_transform(pd.DatetimeIndex(df.date).normalize())
    new_date = pd.DataFrame(new_date, columns=lb.classes_, index=df.index)

    new_category = lb.fit_transform(df['category'])
    new_category = pd.DataFrame(new_category, columns=lb.classes_, index=df.index)
    
    X = new_month.join(new_day).join(new_category).join(df['pic_ind'])
    X['intercept'] = 1 # Adding intercept
    return X
    


def plot_data(df,param):
    
    months = set(df['month'])
    mlikes = []
    for m in months:
    	sub = df[df['month'] == m]
    	mlikes.append(sub[param])
    
    plt.close('all')
    plt.boxplot(mlikes)
    plt.xlabel('Months')
    plt.ylabel(param)
    plt.show()
    plt.close('all')
    
    category = set(df['category'])
    clikes = []
    for c in category:
    	sub = df[df['category'] == c]
    	clikes.append(sub[param])
        
    plt.boxplot(clikes)
    plt.xlabel('Categories')
    plt.ylabel(param)
    plt.show()
    

  
def multiple_lr(y, df, subX,filename):
    print('Running for ',filename)
    mod = sm.OLS(y, subX)
    res = mod.fit()
    with open(filename, "w") as text_file:
        print(res.summary(), file=text_file)
        
        
# This function will split the tweets based on the quantiles of like and retweet
# Add a Popular tweet column to df, its value is 1 for popular tweet else 0'

def split_tweets():
    df = read_data()
    # Check the size of tweets
    total_tweets = df.shape[0]
    print(total_tweets)
    # Checking other statistics for retweet
    retweet_col = df['retweet']
    retweet_col.describe()
    
    # Checking other statistics for like
    like_col = df['like']
    like_col.describe()
    
    # selecting the needed columns and add the popular tweet column
    df = df [['id','retweet','like','date','month','day','char_length','tweet_txt','pic_ind','category']]
    # Set this column to 0 for all the tweets
    df['popular_tweet'] = 0
    
    # Make the id as index and delete the id column  
    df.index = df['id']
    del df['id']
    
    # Check the value of retweet for 98th percentile and like for 97th percentile
    retweet_col.quantile(0.98)
    like_col.quantile(0.97)
    
    # Considering that if a retweet or like is in above 98th and 97th percentile, it is popular
    # rest other tweets are non popular which is 0
    df_popular_tweets = df[(df['retweet'] >=  retweet_col.quantile(0.98)) | (df['like'] >= like_col.quantile(0.97))]
    df.loc[df_popular_tweets.index,"popular_tweet"] = 1
          
    popular_tweet_group = df.groupby("popular_tweet")
    print(popular_tweet_group['popular_tweet'].agg("count"))
    
    print('Random Baseline - ', 523183 /(523183 + 24644))
    return df


def tweet_transform_split(df):
    tv = TfidfVectorizer(lowercase=True,stop_words=ENGLISH_STOP_WORDS,max_df=0.95,min_df=0.009)
    X = tv.fit_transform(df['tweet_txt'])
    y = df['popular_tweet']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3057)
    return X_train, X_test, y_train, y_test


def logistic_lr(X_train, y_train, X_test, y_test):
    print('Logistic Regression')
    classifier = LogisticRegression()
    # Storing the scores of 5 cross validation sets
    crv_scores = cross_val_score(classifier, X_train, y_train, scoring = "accuracy",cv = 5)
    
    # Printing the accuracies and the mean of 5 fold cross validation
    print(50*'#')
    print("Cross Validations Accuracies for 5 validation sets")
    print(crv_scores)
    print(50*'#')
    print("Mean of accuracies")
    print(np.mean(crv_scores))
    
    
    # Instantiated a new instance of logistic regression
    clf_new = LogisticRegression()
    
    # Fit the classifier on X and y
    clf_new.fit(X_train,y_train)
    
    # Did the prediction
    y_prediction = clf_new.predict(X_test)
    
    # Measured the accuracy of the logistic regression classifier
    print(50*'#')
    print('Accuracy score of logistic regression')
    print(accuracy_score(y_test, y_prediction))
    return y_prediction
    
    
   
def support_vector(X_train, y_train, X_test, y_test):
    startTime= datetime.now()
    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)  
    y_pred = clf.predict(X_test)
    print('Accuracy score of Support Vectoe Machine')
    print(accuracy_score(y_test, y_pred))
    timeElapsed=datetime.now()-startTime 
    print('Time elpased for SVM (hh:mm:ss.ms) {}'.format(timeElapsed))
    return y_pred
        
def popular_cat(df, y_test, y_pred):
    y_test_df = pd.DataFrame(y_test)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.index = y_test_df.index
    y_data = y_test_df.join(y_pred_df)
    y_data.columns = ['popular_tweet_test','popular_tweet_predicted']
    predicted_popular_tweets = y_data[y_data['popular_tweet_predicted'] == 1]
    print(df.loc[predicted_popular_tweets.index, "category"].unique())
    
def words_wt(df, category_name):
    df = df[df['category'] == category_name]
    X=df['tweet_txt']
    y=df['popular_tweet']
    cv = CountVectorizer(lowercase=True, stop_words="english", max_df=0.9, min_df=3)    
    X=cv.fit_transform(X).toarray()
    y=y.values
    
    clf1=LogisticRegression()
    clf1.fit(X,y)
    coef_list=[]
    for key,value in cv.vocabulary_.items():
        coef_list.append((key, clf1.coef_[0][value]))
    
    print('##############################################')
    print('List of weights for category', category_name)
    
    sort_coef=sorted(coef_list, key=lambda x: x[1])
    print("\n 10 words with Lowest weights\n", sort_coef[0:10])
    
    sort1_coef=sorted(coef_list, key=lambda x: x[1],reverse=True)
    print("\n 10 words with Highest weights\n", sort1_coef[:10])


# main function
def main():
    df = read_data()
    plot_data(df,'like')
    plot_data(df,'retweet')
    # Label Binarizer 
    lb = LabelBinarizer()
    X = create_ind_var(df, lb)
    # removing one category from each dummy variable
    subX = X.drop(['Jan', 'Book'], axis=1)
    
    ######################################
    # Approach 1 
    ######################################
    
    # considering like as dependent variable
    yl = df['like']
    multiple_lr(yl, df, subX,"like_multiple_linear_regression.txt")
    yl_log = np.log(yl)
    multiple_lr(yl_log, df, subX,"log_of_like_multiple_linear_regression.txt")
    
    #Considering retweet as dependent variable
    yr = df['retweet']
    multiple_lr(yr, df, subX,"retweet_multiple_linear_regression.txt")
    yr_log = np.log(yr)
    multiple_lr(yr_log, df, subX,"log_of_retweet_multiple_linear_regression.txt")
    
    
    ########################################
    # Approach 2
    ########################################
    df = split_tweets()
    X_train, X_test, y_train, y_test = tweet_transform_split(df)
    # Logistic Regression
    print('Running Logistic Regression..')
    y_pred = logistic_lr(X_train, y_train, X_test, y_test)
    
    # Identify the popular categories based on the predicted popular tweets
    print('Popular tweet categories based on logistic regression')
    popular_cat(df, y_test, y_pred)
    
    # Support Vector Machine
    print('Running Support Vector Machine...')
    y_pred = support_vector(X_train, y_train, X_test, y_test)
    print('Popular tweet categories based on Support Vector machine')
    popular_cat(df, y_test, y_pred)
    
    # Print the weights of the words for each category
    words_wt(df,'Health & Beauty')
    words_wt(df,'Clothing')
    words_wt(df,'Electronics')
    words_wt(df,'Home')
    
    
    
if __name__ == "__main__":
    startTime= datetime.now()
    main()
    timeElapsed=datetime.now()-startTime 
    print('Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))
