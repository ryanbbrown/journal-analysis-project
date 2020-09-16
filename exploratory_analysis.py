### EXPLORATORY DATA ANALYSIS

############################################################################
# imports + reading data
import pandas as pd
import numpy as np
import datetime
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import seaborn as sns

corpus = pd.read_pickle('corpus.pkl')
clean_data = pd.read_pickle('clean_data.pkl')
dtm = pd.read_pickle('dtm.pkl')

############################################################################
# yearly dtm without common words
yearly_dtm = dtm.groupby([dtm.YEAR]).sum()
yearly_dtm = yearly_dtm.transpose()

top_dict = {}
for c in yearly_dtm.columns:
    top = yearly_dtm[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))
    
from collections import Counter

words = []
for year in yearly_dtm.columns:
    top = [word for (word, count) in top_dict[year]]
    for t in top:
        words.append(t)
            
Counter(words).most_common()

add_stop_words = [word for word, count in Counter(words).most_common() if count > 2]

from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer

stop_words = stopwords.words('english') + add_stop_words  # add new stop words

    # Recreate document-term matrix
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(clean_data['entry'])
dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
dtm.index = clean_data.index

    # add grouping columns
dtm['YEAR'] = clean_data['year']
dtm['MONTH'] = clean_data['month']
dtm['DAY'] = clean_data['day']

    # make yearly
yearly_dtm = dtm.groupby([dtm.YEAR]).sum()
yearly_dtm.head()

    # pickle
import pickle
pickle.dump(cv, open("cv_stop.pkl", "wb"))
yearly_dtm.to_pickle("yearly_dtm_stop.pkl")


############################################################################
# word cloud
from wordcloud import WordCloud
from matplotlib import pyplot as plt

dtm = pd.read_pickle("yearly_dtm_stop.pkl")
dtm = dtm.transpose()

wc = WordCloud(max_words=75, height=500, width=1000,
              background_color="white", colormap="Dark2",
               random_state=42)

indexes = [0,1,2,3,4]
names = [2016,2017,2018,2019,2020]

for index, name in zip(indexes, names):
    word_dict = pd.Series(dtm.iloc[:,index].values, index=dtm.index).to_dict()
    wc.generate_from_frequencies(word_dict)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(name)
    plt.show()
    

############################################################################
# sentiment analysis
from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

corpus['polarity'] = corpus['entry'].apply(pol)
corpus['subjectivity'] = corpus['entry'].apply(sub)

monthly = corpus['polarity'].groupby([corpus.year, corpus.month]).mean()
yearly = corpus['polarity'].groupby([corpus.year]).mean()

    # polarity over time, monthly
import seaborn as sns
sns.set(style='darkgrid')
sns.lineplot(x=(range(0,len(monthly.values))), y=monthly.values)
plt.xlabel('Months since Jan 2016')
plt.ylabel('Polarity')
plt.title('Polarity over time (monthly)', fontsize=15, fontweight='bold')
plt.show()

    # polarity over time, yearly
sns.lineplot(x=(range(2016,2021)), y=yearly.values)
plt.xlabel('Year')
plt.ylabel('Polarity')
plt.title('Polarity over time (yearly)', fontsize=15, fontweight='bold')
plt.xticks(range(2016,2021))
plt.show()


############################################################################
# length analysis
daily_length = pd.DataFrame(columns=['# words'])

def daily_words(text):
    alltext = list(text['entry'])

    numwords = []
    vocab = []
    
    for item in alltext:
        numwords.append(len(item.split()))
        vocab.append(len(set(item.split())))
        
    daily_length['# words'] = numwords
    
daily_words(corpus)

    # word count vs. polarity (monthly)
sns.scatterplot(x=monthlydf['polarity'].values, y=lengthdf['words per day'].values)
plt.xlabel('Polarity')
plt.ylabel('Average words per day')
plt.title('Average WPD vs. polarity (monthly)', fontsize=15, fontweight='bold')
plt.show()

    # word count vs. polarity (daily)
sns.scatterplot(x=corpus['polarity'].values, y=daily_length['# words'].values)
plt.xlabel('Polarity')
plt.ylabel('Words')
plt.title('Word count vs. polarity (daily)', fontsize=15, fontweight='bold')
plt.show()


############################################################################
# length analysis cont.
    # yearly words per day vs. sentiment
def yearly_words(text):
    alltext = list(text['entry'].groupby(text.year).sum().values)
    
    numwords = []
    vocab = []
    
    for item in alltext:
        numwords.append(len(item.split()))
        vocab.append(len(set(item.split())))
        
    yearly_length['# words'] = numwords
    yearly_length['words per day'] = yearly_length['# words']/yearly_length['# days']
    yearly_length['# unique words'] = vocab
    
yearly_length = pd.DataFrame(columns=['year','# days','# words','words per day','# unique words'])
yearly_length.year=[2016,2017,2018,2019,2020]
yearly_length['# days']=[366,365,365,365,140]
yearly_words(corpus)

fig,ax1 = plt.subplots()

ax1.plot((range(2016,2021)), yearly.values)
ax1.set_title('Average WPD and polarity (yearly)', fontsize=15, fontweight='bold')
ax1.set_ylabel('Polarity')
ax1.set_xlabel('Year')
ax2 = ax1.twinx()
ax2.plot((range(2016,2021)), yearly_length['words per day'].values, color='tab:red')
ax2.set_ylabel('Average words per day')

plt.xticks(range(2016,2021))
ax2.grid(None)

plt.show

    # monthly words per day vs. sentiment
def monthly_words(text):
    alltext = list(text['entry'].groupby([text.year, text.month]).sum().values)
    
    numwords = []
    vocab = []
    
    for item in alltext:
        numwords.append(len(item.split()))
        vocab.append(len(set(item.split())))
        
    monthly_length['# words'] = numwords
    monthly_length['words per day'] = monthly_length['# words']/monthly_length['# days']
    monthly_length['# unique words'] = vocab
    
monthly_length = pd.DataFrame(columns=['year','month','# days','# words','words per day','# unique words'])
monthly_length.year=[2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,
               2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,
               2018,2018,2018,2018,2018,2018,2018,2018,2018,2018,2018,2018,
               2019,2019,2019,2019,2019,2019,2019,2019,2019,2019,2019,2019,
               2020,2020,2020,2020,2020]
monthly_length.month=['January','February','March','April','May','June','July','August','September','October','November','December',
               'January','February','March','April','May','June','July','August','September','October','November','December',
               'January','February','March','April','May','June','July','August','September','October','November','December',
               'January','February','March','April','May','June','July','August','September','October','November','December',
               'January','February','March','April','May']
monthly_length['# days']=[31,29,31,30,31,30,31,31,30,31,30,31,
                   31,28,31,30,31,30,31,31,30,31,30,31,
                   31,28,31,30,31,30,31,31,30,31,30,31,
                   31,28,31,30,31,30,31,31,30,31,30,31,
                   31,28,31,30,19]

monthly_words(corpus)

import seaborn

fig,ax1 = plt.subplots()

ax1.plot((range(0,len(monthly.values))), monthly.values)
ax1.set_title('Average WPD and polarity (monthly)', fontsize=15, fontweight='bold')
ax1.set_ylabel('Polarity')
ax1.set_xlabel('Months since Jan 2016')
ax2 = ax1.twinx()
ax2.plot((range(0,len(monthly.values))), monthly_length['words per day'].values, color='tab:red')
ax2.set_ylabel('Average words per day')

ax2.grid(None)
plt.show
