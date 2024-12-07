import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer, get_tokenizer
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import pickle

def find_top_5_images(query_embedding, df):
    # Calculate cosine similarity for each embedding in the DataFrame
    df['similarity'] = df['embedding'].apply(lambda x: F.cosine_similarity(query_embedding, torch.tensor(x).unsqueeze(0)).item())
    
    # Sort the DataFrame by similarity in descending order
    top_5_df = df.sort_values(by='similarity', ascending=False).head(5)
    
    # Extract the file names and similarities of the top 5 images
    top_5_images = top_5_df[['file_name', 'similarity']].to_dict(orient='records')
    
    return top_5_images

def image_to_embedding(image, model, preprocess):
    # Preprocess image and compute embedding
    image = preprocess(Image.open(image)).unsqueeze(0)
    query_embedding = F.normalize(model.encode_image(image))
    return query_embedding

def text_to_embedding(text, model):
    # Tokenize text and compute embedding
    tokenizer = get_tokenizer('ViT-B-32')
    model.eval()
    text = tokenizer([text])
    query_embedding = F.normalize(model.encode_text(text))
    return query_embedding

def hybrid_to_embedding(text, image, model, lam, preprocess):
    # Compute embeddings for text and image
    text_embedding = text_to_embedding(text, model)
    image_embedding = image_to_embedding(image, model, preprocess)
    # Combine embeddings
    query_embedding = F.normalize(lam * text_embedding + (1 - lam) * image_embedding)
    return query_embedding


def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.convert('L')  # Convert to grayscale ('L' mode)
    img = img.resize(target_size)  # Resize to target size
    img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
    img_array.flatten()
    return img_array
    
def load_images(image_dir, max_images=None, target_size=(224, 224)):
    images = []
    image_names = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_dir, filename))
            img = img.convert('L')  # Convert to grayscale ('L' mode)
            img = img.resize(target_size)  # Resize to target size
            img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array.flatten())  # Flatten to 1D
            image_names.append(filename)
        if max_images and i + 1 >= max_images:
            break
    return images, image_names

def nearest_neighbors(query_embedding, embeddings, top_k=5):
    # query_embedding: The embedding of the query item (e.g., the query image) in the same dimensional space as the other embeddings.
    # embeddings: The dataset of embeddings that you want to search through for the nearest neighbors.
    # top_k: The number of most similar items (nearest neighbors) to return from the dataset.
    # Hint: flatten the "distances" array for convenience because its size would be (1,N)
    # Use euclidean distance
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    nearest_indices = np.argsort(distances)[:top_k] #TODO get the indices of ntop k results
    return nearest_indices, distances[nearest_indices]

def pca(test, image, image_dir, model, k, preprocess):
    # Check if pickle file exists
    if os.path.exists('reduced_embeddings.pickle'):
        with open('reduced_embeddings.pickle', 'rb') as f:
            reduced_embeddings = pickle.load(f)
    else:
        # Step 1: Load and preprocess training images
        train_images, train_image_names = load_images(image_dir, max_images=2000, target_size=(224, 224))
        train_images = np.array(train_images)
        
        # Step 2: Apply PCA to reduce dimensions
        pca = PCA(n_components=k)
        pca.fit(train_images)
        print("Finished PCA fitting")
        
        # Step 3: Load and preprocess all images for comparison
        transform_images, transform_image_names = load_images(image_dir, max_images=10000, target_size=(224, 224))
        reduced_embeddings = pca.transform(transform_images)
        # Use pickle to save the reduced_embeddings
        with open('reduced_embeddings.pickle', 'wb') as f:
            pickle.dump(reduced_embeddings, f)
        print("Finished PCA transformation")
    
    # Step 4: Load and preprocess the user input image
    user_image = preprocess_image(image, target_size=(224, 224))
    # Apply PCA to reduce dimensions
    reduced_user_embedding = pca.transform(user_image.reshape(1, -1))
    print("Finished PCA transformation for user input")
    
    # Step 5: Compute similarity (e.g., Euclidean distance)
    nearest_indices, distances = nearest_neighbors(reduced_user_embedding, reduced_embeddings)
    print("Finished nearest neighbors search")
    
    # Make top 5 results as a dictionary
    results = []
    for i, idx in enumerate(nearest_indices):
        results.append({'file_name': transform_image_names[idx], 'similarity': distances[i].item()})
    
    return results


