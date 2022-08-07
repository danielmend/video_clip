import torch
from torch.utils.data import Dataset
import os
import numpy as np
from utils import pad_video_frames

class CLIPEmbeddingsDataset(Dataset):
    def __init__(self, vid_embeddings_dir, txt_embeddings_dir, seq_len=1024):
        self.vid_embeddings_dir = vid_embeddings_dir
        self.txt_embeddings_dir = txt_embeddings_dir
        self.filenames = os.listdir(vid_embeddings_dir)
        self.seq_len = seq_len

    def __len__(self):
        return len(os.listdir(self.vid_embeddings_dir))

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        vid_embeddings = torch.from_numpy(np.load(f'{self.vid_embeddings_dir}{fname}'))
        vid_embeddings, num_frames = pad_video_frames(vid_embeddings, self.seq_len)
        txt_embedding = torch.from_numpy(np.load(f'{self.txt_embeddings_dir}{fname}'))
        return vid_embeddings, txt_embedding, num_frames


    