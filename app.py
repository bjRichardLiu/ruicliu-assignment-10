from flask import Flask, request, render_template, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import torch.nn.functional as F
from open_clip import create_model_and_transforms, tokenizer
import os
from search import find_top_5_images, image_to_embedding, text_to_embedding, hybrid_to_embedding

app = Flask(__name__)

# Load pre-trained model and embeddings
model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
model.eval()
df = pd.read_pickle('image_embeddings.pickle')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    mode = request.form.get('mode')
    text_query = request.form.get('text_query')
    image_query = request.files.get('image_query')
    lam = request.form.get('weight')
    k = request.form.get('k')
    
    query_embedding = None
    
    if mode == 'text':
        query_embedding = text_to_embedding(text_query, model)
    elif mode == 'image':
        query_embedding = image_to_embedding(image_query, model, preprocess)
    elif mode == 'hybrid':
        query_embedding = hybrid_to_embedding(text_query, image_query, model, float(lam), preprocess)
    
    if query_embedding is None:
        return jsonify({'error': 'Invalid mode'})
    search_results = find_top_5_images(query_embedding, df)
    # Add folder path to file names
    for result in search_results:
        result['file_name'] = os.path.join("static", "coco_images_resized", result['file_name'])

    
    image_paths = [result['file_name'] for result in search_results]
    similarities = [result['similarity'] for result in search_results]
    
    print(image_paths)
    print(similarities)
    
    return jsonify({'image_paths': image_paths, 'similarities': similarities})

if __name__ == '__main__':
    app.run(debug=True)
