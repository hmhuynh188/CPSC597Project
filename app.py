import matplotlib
matplotlib.use('Agg')  # use non-interactive backend before importing pyplot
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import io
import base64

app = Flask(__name__) # starts Flask application 
CORS(app)  # handles requests 

def process_data(file):
    data = pd.read_csv(file)

    # finding the fraud column
    fraud_column = None
    # class, is_fraud, Fraud, fraud, Label are a few examples of the names of the variables for fraud/non-fraud 
    # checks to find the variable
    for col in ['Class', 'is_fraud', 'Fraud', 'fraud', 'Label']:
        if col in data.columns:
            fraud_column = col
            break

    if fraud_column is None:
        raise KeyError("The dataset must contain a fraud indicator column, such as 'Class' or 'is_fraud'.")

    # make sure the fraud column is an integer 
    data[fraud_column] = data[fraud_column].astype(int)

    # balance the dataset by undersampling
    non_fraud = data[data[fraud_column] == 0]
    fraud = data[data[fraud_column] == 1]
    non_fraud_undersampled = non_fraud.sample(n=len(fraud), random_state=42)
    balanced_data = pd.concat([fraud, non_fraud_undersampled], ignore_index=True)

    print("Original Class Distribution:")
    print(data[fraud_column].value_counts())

    print("Balanced Class Distribution:")
    print(balanced_data[fraud_column].value_counts())

    # prepare features and labels
    Y = balanced_data[fraud_column]
    X = balanced_data.drop([fraud_column], axis=1)

    # make sure all features are numbers 
    X = pd.get_dummies(X, drop_first=True)
    X = X.reindex(columns=sorted(X.columns), fill_value=0)  # make sure there are consistent columns 

    # split into train and test sets
    xTrain, xTest, yTrain, yTest = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    # train the model 
    rfc = RandomForestClassifier(random_state=42)  # Set random_state
    rfc.fit(xTrain, yTrain)
    yPred = rfc.predict(xTest)

    print("Predicted Class Distribution:")
    print(pd.Series(yPred).value_counts())

    # calculating the scores - https://www.geeksforgeeks.org/ml-credit-card-fraud-detection/ 
    results = {
        "accuracy": accuracy_score(yTest, yPred),
        "precision": precision_score(yTest, yPred),
        "recall": recall_score(yTest, yPred),
        "f1_score": f1_score(yTest, yPred),
    }

    # generate plots
    plots = {}

    # histograms for the columns
    for col in balanced_data.columns:
        if col == fraud_column:  # only calculate the fraud columns 
            continue
        
        plt.figure(figsize=(12, 8))
        if balanced_data[col].dtype in ['object', 'category'] or balanced_data[col].nunique() < 20:  # categorical 
            top_values = balanced_data[col].value_counts().head(10)
            sns.barplot(x=top_values.index, y=top_values.values)
            plt.title(f"Top 10 Values in {col}")
            plt.xticks(rotation=45, fontsize=16)  # Increase font size for x-axis
            plt.yticks(fontsize=16)
        else:  # numerical 
            sns.histplot(balanced_data[col].dropna(), bins=10, kde=False)
            plt.title(f"Histogram of {col}", fontsize = 10)
            plt.xticks(fontsize=16)  # Increase font size for x-axis
            plt.yticks(fontsize=16)
        plt.tight_layout()
        plots[f'{col}_histogram'] = get_base64_plot()

    # confusion matrix - https://www.geeksforgeeks.org/ml-credit-card-fraud-detection/ 
    plt.figure(figsize=(12, 12))
    conf_matrix = confusion_matrix(yTest, yPred)
    sns.heatmap(conf_matrix, annot=True, fmt="d",
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title("Confusion Matrix", fontsize = 16)
    plt.xticks(fontsize=16, rotation=45)  
    plt.yticks(fontsize=16)
    plots['confusion_matrix'] = get_base64_plot()

    return results, plots
  
# to embed in html 
def get_base64_plot():
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return plot_base64
  
# defines route for the homepage 
@app.route('/')
def home():
    return render_template('index.html')
  
# handles POST requests 
@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('file')
    if file:
        results, plots = process_data(file)
        return jsonify({"metrics": results, "plots": plots})
    else:
        return jsonify({"status": "error", "message": "No file uploaded."})
      
# runs flask server 
if __name__ == '__main__':
    app.run(debug=True)
