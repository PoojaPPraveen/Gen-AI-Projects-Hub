# Custom Sentence Transformer API

Welcome to the Custom Sentence Transformer API! This repository hosts a Flask-based web application that leverages the SentenceTransformer model to provide various functionalities for encoding sentences and documents, as well as calculating similarities between them.

## Features

- **Encode Sentence**: Converts a sentence into an embedding using the SentenceTransformer model.
- **Calculate Sentence Similarity**: Computes the cosine similarity between two sentence embeddings.
- **Generate Document Embeddings**: Encodes each sentence in a document into an embedding.
- **Calculate Document Similarity**: Computes the similarity between two documents based on their sentence embeddings.

## Endpoints

### Home

- **URL**: `/`
- **Method**: GET
- **Description**: Returns a welcome message.

### Encode Sentence

- **URL**: `/cust_encode`
- **Method**: POST
- **Description**: Encodes a given sentence into an embedding.
- **Request Body**:
  ```json
  {
      "sentence": "Your sentence here"
  }
Calculate Sentence Similarity
URL: /cust_similarity
Method: POST
Description: Calculates the cosine similarity between two sentences.
Request Body:
json
Copy code
{
    "sentence1": "First sentence",
    "sentence2": "Second sentence"
}
Response:
json
Copy code
{
    "similarity": similarity_value
}
Generate Document Embeddings
URL: /cust_doc_embedding
Method: POST
Description: Encodes each sentence in a document into an embedding.
Request Body:
json
Copy code
{
    "document": "Your document text here."
}
Response:
json
Copy code
{
    "embeddings": [[embedding1_values], [embedding2_values], ...]
}
Calculate Document Similarity
URL: /cust_doc_similarity
Method: POST
Description: Calculates the similarity between two documents.
Request Body:
json
Copy code
{
    "document1": "First document text here.",
    "document2": "Second document text here."
}
Response:
json
Copy code
{
    "average_similarity": average_similarity_value
}
Installation
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/custom-sentence-transformer-api.git
cd custom-sentence-transformer-api
Create and Activate a Virtual Environment:

bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Application:

bash
Copy code
flask run
Usage
To use the API, send HTTP POST requests to the provided endpoints with the appropriate JSON payloads. The responses will contain the requested embeddings or similarity values.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.
