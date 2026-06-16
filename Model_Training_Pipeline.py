import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from bhho_optimizer import BHHO, create_fitness_function

os.makedirs('models', exist_ok=True)

# 1. Email
print("Loading Email Datasets...")
email_df1 = pd.read_csv('AVN_Basic.csv', on_bad_lines='skip').dropna(subset=['body', 'label'])
email_df2 = pd.read_csv('AVN_Corpus.csv', on_bad_lines='skip').dropna(subset=['body', 'label'])
email_df = pd.concat([email_df1, email_df2], ignore_index=True)
email_df['text'] = email_df['subject'].fillna('') + " " + email_df['body']

sample_size = min(5000, len(email_df))
email_df = email_df.sample(n=sample_size, random_state=42)

X_email_text = email_df['text']
y_email = email_df['label']

X_train_text, X_test_text, y_train, y_test = train_test_split(X_email_text, y_email, test_size=0.2, random_state=42)

print("Extracting features using TF-IDF...")
email_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X_train = email_vectorizer.fit_transform(X_train_text).toarray()
X_test = email_vectorizer.transform(X_test_text).toarray()

joblib.dump(email_vectorizer, 'models/email_vectorizer.pkl')

print("Starting BHHO Feature Selection for Email...")
fitness_func = create_fitness_function(X_train, y_train, alpha=0.99)
optimizer = BHHO(fitness_func, num_agents=5, max_iter=10, dim=X_train.shape[1])
best_features_mask, best_fitness = optimizer.optimize()

selected_indices = np.where(best_features_mask == 1)[0]
X_train_sel = X_train[:, selected_indices]
X_test_sel = X_test[:, selected_indices]

print("Training Final Random Forest on Selected Features...")
email_rf = RandomForestClassifier(n_estimators=100, random_state=42)
email_rf.fit(X_train_sel, y_train)

preds = email_rf.predict(X_test_sel)
print("Email Model Accuracy:", accuracy_score(y_test, preds))
joblib.dump(best_features_mask, 'models/email_feature_mask.pkl')
joblib.dump(email_rf, 'models/email_model.pkl')


# 2. SMS
print("Loading SMS Datasets...")
sms_df1 = pd.read_csv('Dataset_5971 SMS Kaggle.csv').dropna(subset=['TEXT', 'LABEL'])
sms_df1 = sms_df1.rename(columns={'TEXT': 'text', 'LABEL': 'label'})

sms_df2 = pd.read_csv('SMS_Dataset_10191.csv').dropna(subset=['TEXT', 'LABEL'])
sms_df2 = sms_df2.rename(columns={'TEXT': 'text', 'LABEL': 'label'})

sms_df3 = pd.read_csv('SMS_composite_test.csv', on_bad_lines='skip').dropna(subset=['text', 'label'])

sms_df = pd.concat([sms_df1, sms_df2, sms_df3], ignore_index=True)

# Sample the SMS dataset to a manageable size (e.g., 8000) so training doesn't take forever, since the new file is 17MB.
sample_size = min(8000, len(sms_df))
sms_df = sms_df.sample(n=sample_size, random_state=42)

X_sms_text = sms_df['text']
# Convert label to 0 or 1. (Handles both 'ham'/'spam' strings and 0/1 integers)
y_sms = sms_df['label'].apply(lambda x: 0 if str(x).strip().lower() in ['ham', '0', '0.0'] else 1)

X_train_text, X_test_text, y_train, y_test = train_test_split(X_sms_text, y_sms, test_size=0.2, random_state=42)

print("Extracting features using TF-IDF...")
sms_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X_train = sms_vectorizer.fit_transform(X_train_text).toarray()
X_test = sms_vectorizer.transform(X_test_text).toarray()

joblib.dump(sms_vectorizer, 'models/sms_vectorizer.pkl')

print("Starting BHHO Feature Selection for SMS...")
fitness_func = create_fitness_function(X_train, y_train)
optimizer = BHHO(fitness_func, num_agents=5, max_iter=10, dim=X_train.shape[1])
best_features_mask, best_fitness = optimizer.optimize()

selected_indices = np.where(best_features_mask == 1)[0]
X_train_sel = X_train[:, selected_indices]
X_test_sel = X_test[:, selected_indices]

print("Training Final Random Forest...")
sms_rf = RandomForestClassifier(n_estimators=100, random_state=42)
sms_rf.fit(X_train_sel, y_train)

preds = sms_rf.predict(X_test_sel)
print("SMS Model Accuracy:", accuracy_score(y_test, preds))
joblib.dump(best_features_mask, 'models/sms_feature_mask.pkl')
joblib.dump(sms_rf, 'models/sms_model.pkl')


# 3. URL
print("Loading URL Datasets...")
url_df1 = pd.read_csv('Phishing URL Detection Kaggle.csv').dropna(subset=['url', 'label'])
url_df2 = pd.read_csv('Phishing URLs.csv').dropna(subset=['url', 'Type'])
url_df2['label'] = url_df2['Type'].apply(lambda x: 1 if str(x).strip().lower() == 'phishing' else 0)

url_df = pd.concat([url_df1[['url', 'label']], url_df2[['url', 'label']]], ignore_index=True)

sample_size = min(5000, len(url_df))
url_df = url_df.sample(n=sample_size, random_state=42)

X_url_text = url_df['url']
y_url = url_df['label']

X_train_text, X_test_text, y_train, y_test = train_test_split(X_url_text, y_url, test_size=0.2, random_state=42)

print("Extracting features using TF-IDF...")
url_vectorizer = TfidfVectorizer(max_features=500, analyzer='char', ngram_range=(2, 3))
X_train = url_vectorizer.fit_transform(X_train_text).toarray()
X_test = url_vectorizer.transform(X_test_text).toarray()

joblib.dump(url_vectorizer, 'models/url_vectorizer.pkl')

print("Starting BHHO Feature Selection for URL...")
fitness_func = create_fitness_function(X_train, y_train)
optimizer = BHHO(fitness_func, num_agents=5, max_iter=10, dim=X_train.shape[1])
best_features_mask, best_fitness = optimizer.optimize()

selected_indices = np.where(best_features_mask == 1)[0]
X_train_sel = X_train[:, selected_indices]
X_test_sel = X_test[:, selected_indices]

print("Training Final Random Forest...")
url_rf = RandomForestClassifier(n_estimators=100, random_state=42)
url_rf.fit(X_train_sel, y_train)

preds = url_rf.predict(X_test_sel)
print("URL Model Accuracy:", accuracy_score(y_test, preds))
joblib.dump(best_features_mask, 'models/url_feature_mask.pkl')
joblib.dump(url_rf, 'models/url_model.pkl')
print("All combined models successfully trained and saved!")
