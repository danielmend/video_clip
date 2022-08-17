import torch.optim as optim
import torch
from model import CLIPFormer
from data import CLIPEmbeddingsDataset, VideoEmbeddingDataset, CLIPBatchedEmbeddingsDataset
from torch.utils.data import DataLoader
from utils import contrastive_loss, cyclic_contrastive_loss, get_vid_embedding_from_model_output, get_uncertainty_for_batch, contrastive_loss_with_uncertainty
import numpy as np 
from torch.optim.lr_scheduler import StepLR
import clip
from eval import eval_model, compute_video_embeddings

device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings_train_vid_path = 'embeddings/train/video/'
embeddings_train_txt_path = 'embeddings/train/text/train_word_embeds.pth'

train_set = CLIPBatchedEmbeddingsDataset(embeddings_train_vid_path, embeddings_train_txt_path, device = device)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

print(train_set.__getitem__(7)[0].shape)
print(train_set.__getitem__(7)[2])

print('done')