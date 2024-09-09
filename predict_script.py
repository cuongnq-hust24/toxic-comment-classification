import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

loaded_model = joblib.load('model/toxic_comment_model.pkl')
loaded_vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Code to create submission to kaggle competition
'''
test_data = pd.read_csv('kaggle/test.csv')
test_comments_tfidf = loaded_vectorizer.transform(test_data['comment_text'])

prediction = loaded_model.predict(test_comments_tfidf)
result_df = pd.DataFrame(prediction, columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

result_df.insert(0, 'id', test_data['id'])
result_df.to_csv('submission.csv', index=False)
'''
# Code to test a new comment
test_comments = ['fuck you bitch', 'Hi, I am a harmless comment']
test_vector = loaded_vectorizer.transform(test_comments)

prediction = loaded_model.predict(test_vector)
print(prediction)
