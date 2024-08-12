from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form values to integers
        features = [int(request.form['id']),
                    int(request.form['pregnancies']),
                    int(request.form['age'])]
        
        # Convert the features into a numpy array and reshape it for the model
        final_features = np.array(features).reshape(1, -1)
        
        # Make a prediction using the loaded model
        prediction = model.predict(final_features)
        
        # Interpret the prediction result
        if prediction[0] == 0:
            output = "Not Diabetes"
        else:
            output = "Diabetes"
    except Exception as e:
        output = f"Error: {str(e)}"

    # Render the result back to the form
    return render_template('index.html', prediction=output)

if __name__ == "__main__":
    app.run(debug=True)
