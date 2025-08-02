from flask import Flask,render_template,request,url_for
import joblib 


app=Flask(__name__)

model =joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    input_text = request.form['news']
    transformed_text = vectorizer.transform([input_text])
    prediction = model.predict(transformed_text)[0]
    result = "Real✅" if prediction == 1 else "Fake ❌"
    return render_template("index.html", prediction=result)
if __name__ == "__main__":
    #app.run(debug=True)
    app.run(debug=False, host='0.0.0.0', port=10000)

