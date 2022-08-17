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

class ResidualCLIPFormerDataset(Dataset):
    '''
    Creates offseted copies of each video in memory. Samples windows of size window_size to feed into residual transformer.
    '''
    def __init__(self, vid_embeddings_dir, txt_embed_path, seq_len=1024, num_txt_per_video=20, device='cuda'):
        self.vid_embeddings_dir = vid_embeddings_dir
        self.txt_embeds = torch.load(txt_embed_path)
        
        self.num_txt_per_video = num_txt_per_video
        self.device = device
        self.seq_len = seq_len

        self.video_ids = np.repeat(sorted(os.listdir(vid_embeddings_dir)), num_txt_per_video)

        self.len = len(self.video_ids)
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frames = torch.from_numpy(np.load(f'{self.vid_embeddings_dir}{video_id}')).to(self.device)
        
        offset = idx % self.num_txt_per_video
        frames_idx = idx//self.num_txt_per_video
        num_frames = len(frames)
        txt_embedding = self.txt_embeds[frames_idx, offset].to(self.device)
        idxs = [offset+i for i in range(0, len(frames)-offset, self.num_txt_per_video)]
        
        out_frames = torch.stack([
            torch.mean(frames[idxs[i]:idxs[i+1]], axis=-2) for i in range(len(idxs)-1)
        ])

        window_frames, _ = pad_video_frames(out_frames, self.seq_len)
        padded_frames, _ = pad_video_frames(frames, self.seq_len)
        return padded_frames.to(self.device), window_frames.to(self.device), txt_embedding, num_frames, video_id

class EvalResidualCLIPFormerDataset(ResidualCLIPFormerDataset):
    def __init__(self, *args, **kwargs):
        super(EvalResidualCLIPFormerDataset, self).__init__(*args, **kwargs)
        self.video_ids = sorted(os.listdir(self.vid_embeddings_dir))
        self.len = len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frames = torch.from_numpy(np.load(f'{self.vid_embeddings_dir}{video_id}')).to(self.device)
        num_frames = len(frames)
        idxs = [i for i in range(0, len(frames), self.num_txt_per_video)]

        out_frames = torch.stack([
            torch.mean(frames[idxs[i]:idxs[i+1]], axis=-2) for i in range(len(idxs)-1)
        ])

        window_frames, _ = pad_video_frames(out_frames, self.seq_len)
        padded_frames, _ = pad_video_frames(frames, self.seq_len)

        return padded_frames.to(self.device), window_frames.to(self.device), num_frames, video_id

class ConcatenatedCaptionsDataset(CLIPEmbeddingsDataset):
    def __init__(self, *args, **kwargs):
        self.window_size = kwargs.pop('window_size')

        super(ConcatenatedCaptionsDataset, self).__init__(*args, **kwargs)
        self.video_ids = sorted(os.listdir(self.vid_embeddings_dir))
        self.len = len(self.video_ids)

    def __getitem__(self, idx):
        fname = self.video_ids[idx]
        
        txt_embedding = torch.from_numpy(np.load(f'{self.txt_embeddings_dir}{fname}')).to(self.device)
        frames = torch.from_numpy(np.load(f'{self.vid_embeddings_dir}{fname}'))
        frames_padded, num_frames = pad_video_frames(frames, self.seq_len)

        idxs = [i for i in range(0, len(frames), self.window_size)]

        window_frames = torch.stack(
            [torch.mean(frames[idxs[i]:idxs[i+1]], axis=-2) for i in range(len(idxs)-1)]
        )

        window_frames, _ = pad_video_frames(window_frames, self.seq_len)

        return (frames_padded.half().to(self.device), window_frames.half().to(self.device), txt_embedding, num_frames, fname)

class TestConcatenatedCaptionsDataset(ConcatenatedCaptionsDataset):
    def __init__(self, *args, **kwargs):
        super(TestConcatenatedCaptionsDataset, self).__init__(*args, **kwargs)
        self.filenames = sorted(os.listdir(self.txt_embeddings_dir))
        self.len = len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        txt_embedding = torch.from_numpy(np.load(f'{self.txt_embeddings_dir}{fname}')).to(self.device)
        vid_name = fname.split('_')[0] + '.npy'
        
        frames = torch.from_numpy(np.load(f'{self.vid_embeddings_dir}{vid_name}'))
        frames_padded, num_frames = pad_video_frames(frames, self.seq_len)

        idxs = [i for i in range(0, len(frames), self.window_size)]

        window_frames = torch.stack(
            [torch.mean(frames[idxs[i]:idxs[i+1]], axis=-2) for i in range(len(idxs)-1)]
        )

        window_frames, _ = pad_video_frames(window_frames, self.seq_len)

        return (frames_padded.half().to(self.device), window_frames.half().to(self.device), txt_embedding, num_frames, fname)

class EvalConcatenatedCaptionsDataset(ConcatenatedCaptionsDataset):
    def __init__(self, *args, **kwargs):
        super(EvalConcatenatedCaptionsDataset, self).__init__(*args, **kwargs)
        self.video_ids = sorted(os.listdir(self.vid_embeddings_dir))
        self.len = len(self.video_ids)

    def __getitem__(self, idx):
        fname = self.video_ids[idx]
        
        frames = torch.from_numpy(np.load(f'{self.vid_embeddings_dir}{fname}'))
        frames_padded, num_frames = pad_video_frames(frames, self.seq_len)

        idxs = [i for i in range(0, len(frames), self.window_size)]

        window_frames = torch.stack(
            [torch.mean(frames[idxs[i]:idxs[i+1]], axis=-2) for i in range(len(idxs)-1)]
        )

        window_frames, _ = pad_video_frames(window_frames, self.seq_len)

        return (frames_padded.half().to(self.device), window_frames.half().to(self.device), num_frames, fname)

class CLIPBatchedEmbeddingsDataset(Dataset):
    def __init__(self, vid_embeddings_dir, txt_embeddings_dir, seq_len=1024, batch_size = 5, device='cuda'):
        self.vid_embeddings_dir = vid_embeddings_dir
        self.txt_embeddings_dir = txt_embeddings_dir
        self.seq_len = seq_len
        self.device = device
        self.batch_size = batch_size

        self.len = len(os.listdir(self.vid_embeddings_dir)) * 20

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        video_idx = idx // 20
        txt_idx = idx % 20

        txt_embedding = torch.load(f'{self.txt_embeddings_dir}', map_location=torch.device(self.device)).float().to(self.device)[video_idx][txt_idx]

        vid_name = f'video{video_idx}.npy'
        vid_embeddings = torch.from_numpy(np.load(f'{self.vid_embeddings_dir}video{video_idx}.npy'))
        vid_len = len(vid_embeddings)

        vid_idxs = torch.arange(txt_idx, vid_len - (vid_len - txt_idx) % self.batch_size).reshape((vid_len - txt_idx) // self.batch_size, self.batch_size)
        vid_embeddings = vid_embeddings[vid_idxs]
        vid_embeddings = torch.mean(vid_embeddings, 1)

        vid_embeddings, num_frames = pad_video_frames(vid_embeddings, self.seq_len)

        return (vid_embeddings.float().to(self.device), txt_embedding, num_frames, vid_name)

vid_embeddings_dir = 'video_clip/embeddings/test/video/'
txt_embeddings_dir = 'video_clip/embeddings/test/text/'
window_size = 24
x = ConcatenatedCaptionsDataset(vid_embeddings_dir, txt_embeddings_dir, window_size=24)
y = EvalConcatenatedCaptionsDataset(vid_embeddings_dir, txt_embeddings_dir, window_size=24)
z = TestConcatenatedCaptionsDataset(vid_embeddings_dir, txt_embeddings_dir, window_size=24)
