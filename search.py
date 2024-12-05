import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer, get_tokenizer
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

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