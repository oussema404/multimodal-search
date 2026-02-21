from sentence_transformers import SentenceTransformer
from PIL import Image
import torch

class CLIPEmbedder:
    def __init__(self, model_name="sentence-transformers/clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
    
    def encode_text(self, text_list):
        return self.model.encode(text_list, convert_to_tensor=True, normalize_embeddings=True)
    
    def encode_images(self, image_paths):
        images = [Image.open(p).convert("RGB") for p in image_paths]
        return self.model.encode(images, batch_size=8, convert_to_tensor=True, normalize_embeddings=True)
