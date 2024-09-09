from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from together import Together

app = Flask(__name__)

# Allow CORS from all origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the datasets
laws_data = pd.read_csv('top_sections.csv')
cases_data = pd.read_csv('case_details.csv')

# Load a pre-trained sentence transformer model for semantic search
model = SentenceTransformer('msmarco-distilbert-base-v4')

# Initialize the Together client with the API key
api_key = 'e745a473362a583a6834266b8b49ad13ffc5a3ddd0b6d383993d23e2fb1bfa3f'
client = Together(api_key=api_key)

# Encode all the law titles into embeddings for semantic search
law_titles = laws_data['title'].tolist()
law_embeddings = model.encode(law_titles, convert_to_tensor=True)

def get_ipc_laws(case_description):
    response = client.chat.completions.create(
        model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        messages=[{"role": "user", "content": f"I have a case description: {case_description}. What IPC laws apply to this case?"}]
    )
    
    ipc_laws = response.choices[0].message.content
    return ipc_laws

@app.route('/get_laws_and_cases', methods=['POST'])
def get_laws_and_cases():
    data = request.get_json()
    case_description = data.get('case_description', '')

    # Encode the case description into an embedding
    case_embedding = model.encode(case_description, convert_to_tensor=True)

    # Compute cosine similarity between the case description and law titles
    cosine_scores = util.pytorch_cos_sim(case_embedding, law_embeddings)[0]

    # Get the top 5 most similar laws
    top_results = torch.topk(cosine_scores, k=5)

    relevant_laws = []

    for score, idx in zip(top_results[0], top_results[1]):
        law_title = laws_data.iloc[idx.item()]['title']
        law_url = laws_data.iloc[idx.item()]['url']
        cited_by = int(laws_data.iloc[idx.item()]['citedby'])  # Ensure cited_by is an int for JSON serialization

        # Retrieve cases and filter out incomplete ones, preferring those with all fields
        complete_cases = cases_data.dropna(subset=['case_no', 'court_name', 'judgment_date', 'judgment_link'])
        past_case_details = complete_cases[['case_no', 'court_name', 'judgment_date', 'judgment_link']].to_dict('records')

        # Limit the number of past cases to 10
        past_case_details = past_case_details[:10]

        relevant_laws.append({
            'law_title': law_title,
            'url': law_url,
            'cited_by': cited_by,
            'past_case_details': past_case_details,
            'similarity_score': float(score)
        })

    # Get IPC laws using the Mistral model
    ipc_laws = get_ipc_laws(case_description)

    return jsonify({
        'relevant_laws': relevant_laws,
        'ipc_laws': ipc_laws
    })

# No need for app.run() here, since Gunicorn will handle it
