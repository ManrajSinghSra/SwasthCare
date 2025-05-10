from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# NLP Processing
ps = PorterStemmer()
data['tags'] = data['tags'].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))
cv = CountVectorizer(stop_words="english", max_features=5000)
vectors = cv.fit_transform(data['tags']).toarray()
similarity = cosine_similarity(vectors)

# Normalize text for search
def normalize_text(text):
    return str(text).strip().lower()

# Function to get medicine information and alternatives
def get_medicine_info(search_term, search_type="both"):
    normalized_input = normalize_text(search_term)
    
    # Initialize empty DataFrame for matches
    matches = pd.DataFrame()
    
    if search_type in ["name", "both"]:
        # Search by medicine name
        name_matches = data[data['Medicine_Name'].apply(lambda x: normalize_text(x)).str.contains(normalized_input, na=False)]
        matches = pd.concat([matches, name_matches])
    
    if search_type in ["composition", "both"] and len(matches) == 0:
        # Search by composition if no name matches or specifically requested
        comp_matches = data[data['Composition'].apply(lambda x: normalize_text(" ".join(x))).str.contains(normalized_input, na=False)]
        matches = pd.concat([matches, comp_matches])
    
    # Remove duplicates if any
    matches = matches.drop_duplicates(subset=['Medicine_Name'])
    
    if not matches.empty:
        row = matches.iloc[0]
        index = data.index[data['Medicine_Name'] == row['Medicine_Name']][0]
        
        # Get top 5 alternative medicines based on similarity
        similar_meds = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)[1:6]
        alternatives = [data.iloc[i[0]]['Medicine_Name'] for i in similar_meds]
        
        return {
            "Medicine_Name": row['Medicine_Name'],
            "Composition": " ".join(row['Composition']),
            "Uses": " ".join(row['Uses']),
            "Side_Effects": " ".join(row['Side_effects']),
            "Alternatives": alternatives,
            "Image_URL": row['Image_URL'] if pd.notna(row['Image_URL']) else None
        }
    
    return {"error": "Medicine not found"}

# API Route with enhanced search capability
@app.route("/search", methods=["POST"])
def search():
    request_data = request.get_json()
    search_term = request_data.get("medicine", "")
    search_type = request_data.get("search_type", "both")  # Default to searching both
    
    response = get_medicine_info(search_term, search_type)
    return jsonify(response)

# New route to get top 5 medicines
@app.route("/top-medicines", methods=["GET"])
def get_top_medicines():
    top_medicines = []
    for i in range(min(5, len(data))):
        med = data.iloc[i]
        top_medicines.append({
            "Medicine_Name": med['Medicine_Name'],
            "Composition": " ".join(med['Composition']),
            "Uses": " ".join(med['Uses']),
            "Side_Effects": " ".join(med['Side_effects']),
            "Image_URL": med['Image_URL'] if pd.notna(med['Image_URL']) else None
        })
    return jsonify({"top_medicines": top_medicines})

# New route to suggest medicines by partial search term
@app.route("/suggest", methods=["GET"])
def suggest_medicines():
    search_term = request.args.get("term", "")
    normalized_input = normalize_text(search_term)
    
    # Get matches from both name and composition
    name_matches = data[data['Medicine_Name'].apply(lambda x: normalize_text(x)).str.contains(normalized_input, na=False)]
    comp_matches = data[data['Composition'].apply(lambda x: normalize_text(" ".join(x))).str.contains(normalized_input, na=False)]
    
    # Combine results and get unique medicine names
    all_matches = pd.concat([name_matches, comp_matches])
    suggestions = all_matches['Medicine_Name'].drop_duplicates().tolist()[:10]  # Limit to 10 suggestions
    
    return jsonify({"suggestions": suggestions})

if __name__ == "__main__":
    app.run(debug=True)