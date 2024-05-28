from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to encode a sentence into an embedding
def encode_sentence(sentence):
    return model.encode(sentence)

# Route for the home page
@app.route('/')
def home():
    return "Welcome to the Sentence Transformer API!"

# Route for encoding sentences
@app.route('/encode', methods=['POST'])
def encode():
    data = request.json
    if 'sentence' not in data:
        return jsonify({'error': 'No sentence provided'}), 400
    
    sentence = data['sentence']
    embedding = encode_sentence(sentence)
    
    return jsonify({'embedding': embedding.tolist()})

# Route for calculating similarity between sentences
@app.route('/similarity', methods=['POST'])
def similarity():
    try:
        data = request.json
        if 'sentence1' not in data or 'sentence2' not in data:
            return jsonify({'error': 'Two sentences required'}), 400
        
        sentence1 = data['sentence1']
        sentence2 = data['sentence2']
        
        embedding1 = encode_sentence(sentence1)
        embedding2 = encode_sentence(sentence2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        return jsonify({'similarity': float(similarity)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
