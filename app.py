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
    return "Welcome to the Custom Sentence Transformer API!"

# Route for encoding sentences
@app.route('/cust_encode', methods=['POST'])
def cust_encode():
    data = request.json
    if 'sentence' not in data:
        return jsonify({'error': 'No sentence provided'}), 400
    
    sentence = data['sentence']
    embedding = encode_sentence(sentence)
    
    return jsonify({'embedding': embedding.tolist()})

# Route for calculating similarity between sentences
@app.route('/cust_similarity', methods=['POST'])
def cust_similarity():
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

# Route for document embeddings
@app.route('/cust_doc_embedding', methods=['POST'])
def cust_doc_embedding():
    try:
        data = request.json
        if 'document' not in data:
            return jsonify({'error': 'No document provided'}), 400
        
        document = data['document']
        sentences = document.split('.')
        embeddings = [encode_sentence(sentence.strip()) for sentence in sentences if sentence.strip()]
        
        return jsonify({'embeddings': [embedding.tolist() for embedding in embeddings]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route for calculating similarity between documents
@app.route('/cust_doc_similarity', methods=['POST'])
def cust_doc_similarity():
    try:
        data = request.json
        if 'document1' not in data or 'document2' not in data:
            return jsonify({'error': 'Two documents required'}), 400
        
        document1 = data['document1']
        document2 = data['document2']
        
        sentences1 = [sentence.strip() for sentence in document1.split('.') if sentence.strip()]
        sentences2 = [sentence.strip() for sentence in document2.split('.') if sentence.strip()]
        
        embeddings1 = [encode_sentence(sentence) for sentence in sentences1]
        embeddings2 = [encode_sentence(sentence) for sentence in sentences2]
        
        # Calculate similarity for each sentence in document1 with all sentences in document2
        similarities = []
        for emb1 in embeddings1:
            max_similarity = max(
                np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)) for emb2 in embeddings2
            )
            similarities.append(max_similarity)
        
        average_similarity = float(np.mean(similarities))
        
        return jsonify({'average_similarity': average_similarity})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
