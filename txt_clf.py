from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pandas as pd
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from string import punctuation

russian_stopwords = stopwords.words("russian")

#создаем обучающую выборку
df_train = pd.read_excel('data.xlsx', 'train')
#добавляем рандомные слова с меткой "не распознано"
df_add = pd.read_csv('random_phrases_train.csv', on_bad_lines='skip', names=['Запрос'], nrows=150)
df_add = df_add.replace(to_replace=r'\t.+', value='', regex=True)
df_add["Тема"]="не распознано"
df_train = df_train.append(df_add, ignore_index=True)

#очищаем текст
def remove_punct(text):
    table = {33: ' ', 34: ' ', 35: ' ', 36: ' ', 37: ' ', 38: ' ', 39: ' ', 40: ' ', 41: ' ', 42: ' ', 43: ' ', 44: ' ', 45: ' ', 46: ' ', 47: ' ', 58: ' ', 59: ' ', 60: ' ', 61: ' ', 62: ' ', 63: ' ', 64: ' ', 91: ' ', 92: ' ', 93: ' ', 94: ' ', 95: ' ', 96: ' ', 123: ' ', 124: ' ', 125: ' ', 126: ' '}
    return text.translate(table)

df_train['Запрос'] = df_train['Запрос'].map(lambda x: x.lower())
df_train['Запрос'] = df_train['Запрос'].map(lambda x: remove_punct(x))
df_train['Запрос'] = df_train['Запрос'].map(lambda x: x.split(' '))
df_train['Запрос'] = df_train['Запрос'].map(lambda x: [token for token in x if token not in russian_stopwords\
                                                                  and token != " " \
                                                                  and token.strip() not in punctuation])
df_train['Запрос'] = df_train['Запрос'].map(lambda x: ' '.join(x))

#для удобства
X_train = df_train['Запрос']
y_train = df_train['Тема']

text_clf = Pipeline([
                     ('tfidf', TfidfVectorizer()),
                      ('clf', SGDClassifier(random_state=1))])
 					#Линейный классификатор показал лучший результат на тестовых данных (0.97 средний precision)
text_clf.fit(X_train, y_train)

#input
while(True):
	print('Введите фразу, которую нужно классифицировать: ')
	print(text_clf.predict([input()]))
