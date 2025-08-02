import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib

df_fake = pd.read_csv("fake.csv")
df_true = pd.read_csv("true.csv")

df_fake["label"] = 0
df_true["label"] = 1

df = pd.concat([df_fake,df_true])

x = df["text"]
y = df["label"]

x_train, x_test, y_train,  y_test= train_test_split(x,y,test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

model=PassiveAggressiveClassifier()
model.fit(x_train_vec,y_train)

#y_pred=model.predict(x_test_vec)
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

print("Accuracy:", accuracy_score(y_test,model.predict(x_test_vec)))
#print("Confusion matrix:", confusion_matrix(y_test,y_pred))
