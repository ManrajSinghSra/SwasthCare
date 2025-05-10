from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
 
data = pd.read_csv("Medicine_Details.csv")
data.dropna(inplace=True)
data.rename(columns={'Medicine Name': 'Medicine_Name', 'Image URL': 'Image_URL'}, inplace=True)
 
if 'Image_URL' not in data.columns:
    data['Image_URL'] = None  # Add empty column if missing

 
data['Uses'] = data['Uses'].astype(str).apply(lambda x: x.split())
data['Composition'] = data['Composition'].astype(str).apply(lambda x: x.split())
data['Side_effects'] = data['Side_effects'].astype(str).apply(lambda x: x.split())
data['tags'] = data['Uses'] + data['Composition'] + data['Side_effects']
data['tags'] = data['tags'].apply(lambda x: " ".join(x).lower())


model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(data['tags'].tolist(), show_progress_bar=True)
 
def normalize_text(text):
    return str(text).strip().lower()
 
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
 
@app.route("/search", methods=["POST"])
def search():
    request_data = request.get_json()
    search_term = request_data.get("medicine", "")
    search_type = request_data.get("search_type", "both")
    
    response = get_medicine_info(search_term, search_type)
    return jsonify(response)
 
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
 
@app.route("/suggest", methods=["GET"])
def suggest_medicines():
    search_term = request.args.get("term", "")
    normalized_input = normalize_text(search_term)
    
    name_matches = data[data['Medicine_Name'].apply(lambda x: normalize_text(x)).str.contains(normalized_input, na=False)]
    comp_matches = data[data['Composition'].apply(lambda x: normalize_text(" ".join(x))).str.contains(normalized_input, na=False)]
    
    all_matches = pd.concat([name_matches, comp_matches])
    suggestions = all_matches['Medicine_Name'].drop_duplicates().tolist()[:10]
    
    return jsonify({"suggestions": suggestions})

if __name__ == "__main__":
    app.run(debug=True)



# this is finnal  ~