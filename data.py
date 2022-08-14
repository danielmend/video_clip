import torch
from torch.utils.data import Dataset
import os
import numpy as np
from utils import pad_video_frames

class CLIPEmbeddingsDataset(Dataset):
    def __init__(self, vid_embeddings_dir, txt_embeddings_dir, seq_len=1024, device='cuda'):
        self.vid_embeddings_dir = vid_embeddings_dir
        self.txt_embeddings_dir = txt_embeddings_dir
        self.filenames = sorted(os.listdir(txt_embeddings_dir))
        self.seq_len = seq_len
        self.device = device

        self.len = len(os.listdir(self.txt_embeddings_dir))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        txt_embedding = torch.from_numpy(np.load(f'{self.txt_embeddings_dir}{fname}')).float().to(self.device)

        vid_name = fname
        vid_embeddings = torch.from_numpy(np.load(f'{self.vid_embeddings_dir}{vid_name}'))
        vid_embeddings, num_frames = pad_video_frames(vid_embeddings, self.seq_len)

        return (vid_embeddings.float().to(self.device), txt_embedding, num_frames, vid_name)

class VideoEmbeddingDataset(Dataset):
    def __init__(self, vid_embeddings_dir, seq_len=1024, device='cuda'):
        self.vid_embeddings_dir = vid_embeddings_dir
        self.filenames = sorted(os.listdir(vid_embeddings_dir))
        self.seq_len = seq_len
        self.device = device

        self.len = len(os.listdir(self.vid_embeddings_dir))
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        vid_embeddings = torch.from_numpy(np.load(f'{self.vid_embeddings_dir}{fname}'))
        vid_embeddings, num_frames = pad_video_frames(vid_embeddings, self.seq_len)

        return (vid_embeddings.to(self.device), num_frames, fname)

