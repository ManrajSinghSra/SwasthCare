from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)
CORS(app)

# Load dataset
data = pd.read_csv("Medicine_Details.csv")
data.dropna(inplace=True)
data.rename(columns={'Medicine Name': 'Medicine_Name', 'Image URL': 'Image_URL'}, inplace=True)

# Ensure 'Image_URL' exists
if 'Image_URL' not in data.columns:
    data['Image_URL'] = None  # Add empty column if missing

# Text processing
data['Uses'] = data['Uses'].astype(str).apply(lambda x: x.split())
data['Composition'] = data['Composition'].astype(str).apply(lambda x: x.split())
data['Side_effects'] = data['Side_effects'].astype(str).apply(lambda x: x.split())
data['tags'] = data['Uses'] + data['Composition'] + data['Side_effects']
data['tags'] = data['tags'].apply(lambda x: " ".join(x).lower())

# Load SentenceTransformer model and encode medicine tags
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(data['tags'].tolist(), show_progress_bar=True)

# Normalize text for search
def normalize_text(text):
    return str(text).strip().lower()

# Function to get medicine information and alternatives
def get_medicine_info(search_term, search_type="both"):
    normalized_input = normalize_text(search_term)
    
    matches = pd.DataFrame()
    
    if search_type in ["name", "both"]:
        name_matches = data[data['Medicine_Name'].apply(lambda x: normalize_text(x)).str.contains(normalized_input, na=False)]
        matches = pd.concat([matches, name_matches])
    
    if search_type in ["composition", "both"] and len(matches) == 0:
        comp_matches = data[data['Composition'].apply(lambda x: normalize_text(" ".join(x))).str.contains(normalized_input, na=False)]
        matches = pd.concat([matches, comp_matches])
    
    matches = matches.drop_duplicates(subset=['Medicine_Name'])
    
    if not matches.empty:
        row = matches.iloc[0]
        index = data.index[data['Medicine_Name'] == row['Medicine_Name']][0]
        
        # Get top 5 similar medicines using cosine similarity of embeddings
        similarities = cosine_similarity([embeddings[index]], embeddings)[0]
        similar_indices = similarities.argsort()[::-1][1:6]
        alternatives = data.iloc[similar_indices]['Medicine_Name'].tolist()
        
        return {
            "Medicine_Name": row['Medicine_Name'],
            "Composition": " ".join(row['Composition']),
            "Uses": " ".join(row['Uses']),
            "Side_Effects": " ".join(row['Side_effects']),
            "Alternatives": alternatives,
            "Image_URL": row['Image_URL'] if pd.notna(row['Image_URL']) else None
        }

    return {"error": "Medicine not found"}

# Evaluate accuracy using precision, recall, F1 score
def evaluate_model(test_set):
    ground_truth = []
    predictions = []
    
    for query, expected_recommendations in test_set:
        # Get predictions from model
        response = get_medicine_info(query)
        predicted_recommendations = response.get('Alternatives', [])
        
        # Collect ground truth and predicted recommendations
        ground_truth.append([1 if med in expected_recommendations else 0 for med in expected_recommendations])
        predictions.append([1 if med in predicted_recommendations else 0 for med in expected_recommendations])
    
    # Flatten the lists and calculate precision, recall, and F1 score
    y_true = [item for sublist in ground_truth for item in sublist]
    y_pred = [item for sublist in predictions for item in sublist]
    
    # Calculate Precision, Recall, and F1 Score
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    
    return precision, recall, f1

# Test set (query, expected recommendations)
test_set = [
    ("Maxvoid 8 Tablet", ["Prostagard 8 Capsule", "Silofast 8 Capsule", "Silodal 8 Capsule", "Silotime 8 Capsule", "Silotrif 8 Capsule"]),
    ("Brufen 400 Tablet", ["Brufen 200 Tablet", "Brufen 600 Tablet", "Imol Suspension", "Ibucon Plus Suspension", "Ibukind Plus 100 mg/162.5 mg Suspension"]),
    # Add more queries and expected recommendations as needed
]

# Evaluate the model on the test set
precision, recall, f1 = evaluate_model(test_set)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

if __name__ == "__main__":
    app.run(debug=True)
