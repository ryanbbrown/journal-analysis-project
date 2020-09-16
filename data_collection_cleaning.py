### DATA COLLECTION & CLEANING

############################################################################
# imports
import pandas as pd
import numpy as np
import datetime
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


############################################################################
# 2016 journals
files = ['2016 Text Files/1. January 2016.txt','2016 Text Files/2. February 2016.txt',
        '2016 Text Files/3. March 2016.txt','2016 Text Files/4. April 2016.txt',
        '2016 Text Files/5. May 2016.txt','2016 Text Files/6. June 2016.txt',
        '2016 Text Files/7. July 2016.txt','2016 Text Files/8. August 2016.txt',
        '2016 Text Files/9. September 2016.txt','2016 Text Files/10. October 2016.txt',
        '2016 Text Files/11. November 2016.txt','2016 Text Files/12. December 2016.txt']

months = [1,2,3,4,5,6,7,8,9,10,11,12]

names = ['January','February','March','April','May','June','July',
        'August','September','October','November','December']

year = 2016

df2016 = pd.DataFrame(columns=['year','month','day','date','entry'])

for file, month, name in zip(files, months, names):
    df = pd.read_csv(file, sep='\n')
    
    monthlist1 = []
    monthlist2 = []
    
    for i in range(0,32):
        monthlist1.append(f'{name} {i}')

    for item in monthlist1:
        monthlist2.append(item + ' ')

    finalmonthlist = monthlist1 + monthlist2
    
    newdf = pd.DataFrame(columns=['year','month','day','date','entry'])
    
    newdf['entry'] = df[~df[name].isin(finalmonthlist) == True][name].tolist()
    newdf['year'] = year
    newdf['month'] = month
    newdf['day'] = newdf.index+1
    newdf['date'] = pd.to_datetime(newdf[['year','month','day']])

    df2016 = df2016.append(newdf, ignore_index=True)
    
    
############################################################################
# 2017 journals
files = ['2017 Text Files/1. January 2017.txt','2017 Text Files/2. February 2017.txt',
        '2017 Text Files/3. March 2017.txt','2017 Text Files/4. April 2017.txt',
        '2017 Text Files/5. May 2017.txt','2017 Text Files/6. June 2017.txt',
        '2017 Text Files/7. July 2017.txt','2017 Text Files/8. August 2017.txt',
        '2017 Text Files/9. September 2017.txt','2017 Text Files/10. October 2017.txt',
        '2017 Text Files/11. November 2017.txt','2017 Text Files/12. December 2017.txt']

months = [1,2,3,4,5,6,7,8,9,10,11,12]

names = ['January','February','March','April','May','June','July',
        'August','September','October','November','December']

year = 2017

df2017 = pd.DataFrame(columns=['year','month','day','date','entry'])

for file, month, name in zip(files, months, names):
    df = pd.read_csv(file, sep='\n')
    
    monthlist1 = []
    monthlist2 = []
    
    for i in range(0,32):
        monthlist1.append(f'{name} {i}')

    for item in monthlist1:
        monthlist2.append(item + ' ')

    finalmonthlist = monthlist1 + monthlist2
    
    newdf = pd.DataFrame(columns=['year','month','day','date','entry'])
    
    newdf['entry'] = df[~df[name].isin(finalmonthlist) == True][name].tolist()
    newdf['year'] = year
    newdf['month'] = month
    newdf['day'] = newdf.index+1
    newdf['date'] = pd.to_datetime(newdf[['year','month','day']])

    df2017 = df2017.append(newdf, ignore_index=True)
    
    
############################################################################
# 2018 journals
files = ['2018 Text Files/1. January 2018.txt','2018 Text Files/2. February 2018.txt',
        '2018 Text Files/3. March 2018.txt','2018 Text Files/4. April 2018.txt',
        '2018 Text Files/5. May 2018.txt','2018 Text Files/6. June 2018.txt',
        '2018 Text Files/7. July 2018.txt','2018 Text Files/8. August 2018.txt',
        '2018 Text Files/9. September 2018.txt','2018 Text Files/10. October 2018.txt',
        '2018 Text Files/11. November 2018.txt','2018 Text Files/12. December 2018.txt']

months = [1,2,3,4,5,6,7,8,9,10,11,12]

names = ['January','February','March','April','May','June','July',
        'August','September','October','November','December']

year = 2018

df2018 = pd.DataFrame(columns=['year','month','day','date','entry'])

for file, month, name in zip(files, months, names):
    df = pd.read_csv(file, sep='\n')
    
    monthlist1 = []
    monthlist2 = []
    
    for i in range(0,32):
        monthlist1.append(f'{name} {i}')

    for item in monthlist1:
        monthlist2.append(item + ' ')

    finalmonthlist = monthlist1 + monthlist2
    
    newdf = pd.DataFrame(columns=['year','month','day','date','entry'])
    
    newdf['entry'] = df[~df[name].isin(finalmonthlist) == True][name].tolist()
    newdf['year'] = year
    newdf['month'] = month
    newdf['day'] = newdf.index+1
    newdf['date'] = pd.to_datetime(newdf[['year','month','day']])

    df2018 = df2018.append(newdf, ignore_index=True)
    

############################################################################
# 2019 journals
files = ['2019 Text Files/1. January 2019.txt','2019 Text Files/2. February 2019.txt',
        '2019 Text Files/3. March 2019.txt','2019 Text Files/4. April 2019.txt',
        '2019 Text Files/5. May 2019.txt','2019 Text Files/6. June 2019.txt',
        '2019 Text Files/7. July 2019.txt','2019 Text Files/8. August 2019.txt',
        '2019 Text Files/9. September 2019.txt','2019 Text Files/10. October 2019.txt',
        '2019 Text Files/11. November 2019.txt','2019 Text Files/12. December 2019.txt']

months = [1,2,3,4,5,6,7,8,9,10,11,12]

names = ['January','February','March','April','May','June','July',
        'August','September','October','November','December']

year = 2019

df2019 = pd.DataFrame(columns=['year','month','day','date','entry'])

for file, month, name in zip(files, months, names):
    df = pd.read_csv(file, sep='\n')
    
    monthlist1 = []
    monthlist2 = []
    
    for i in range(0,32):
        monthlist1.append(f'{name} {i}')

    for item in monthlist1:
        monthlist2.append(item + ' ')

    finalmonthlist = monthlist1 + monthlist2
    
    newdf = pd.DataFrame(columns=['year','month','day','date','entry'])
    
    newdf['entry'] = df[~df[name].isin(finalmonthlist) == True][name].tolist()
    newdf['year'] = year
    newdf['month'] = month
    newdf['day'] = newdf.index+1
    newdf['date'] = pd.to_datetime(newdf[['year','month','day']])

    df2019 = df2019.append(newdf, ignore_index=True)

    
############################################################################
# 2020 journals
files = ['2020 Text Files/1. January 2020.txt','2020 Text Files/2. February 2020.txt',
        '2020 Text Files/3. March 2020.txt','2020 Text Files/4. April 2020.txt',
        '2020 Text Files/5. May 2020.txt',]

months = [1,2,3,4,5]

names = ['January','February','March','April','May']

year = 2020

df2020 = pd.DataFrame(columns=['year','month','day','date','entry'])

for file, month, name in zip(files, months, names):
    df = pd.read_csv(file, sep='\n')
    
    monthlist1 = []
    monthlist2 = []
    
    for i in range(0,32):
        monthlist1.append(f'{name} {i}')

    for item in monthlist1:
        monthlist2.append(item + ' ')

    finalmonthlist = monthlist1 + monthlist2
    
    newdf = pd.DataFrame(columns=['year','month','day','date','entry'])
    
    newdf['entry'] = df[~df[name].isin(finalmonthlist) == True][name].tolist()
    newdf['year'] = year
    newdf['month'] = month
    newdf['day'] = newdf.index+1
    newdf['date'] = pd.to_datetime(newdf[['year','month','day']])

    df2020 = df2020.append(newdf, ignore_index=True)
    

############################################################################
# corpus creation
corpus = pd.DataFrame(columns=['year','month','day','date','entry'])

dataframes = [df2016, df2017, df2018, df2019, df2020]

for df in dataframes:
    corpus = corpus.append(df)
    
corpus = corpus.reset_index()

corpus.to_pickle('corpus.pkl')


############################################################################
# corpus cleaning
import re
import string

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, 
    remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    return text

round1 = lambda x: clean_text_round1(x)

clean_data = corpus.copy()
clean_data['entry'] = clean_data.entry.apply(round1)

clean_data.to_pickle('clean_data.pkl')


############################################################################
# document-term matrix
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = stopwords.words('english')

cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(clean_data['entry'])
dtm_df = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
dtm_df.index = clean_data.index

dtm_df['YEAR'] = clean_data['year']
dtm_df['MONTH'] = clean_data['month']
dtm_df['DAY'] = clean_data['day']

dtm_df.to_pickle('dtm.pkl')
