import torch
from torch import nn
import torch.nn.functional as F

import glob
from clip_video_encode import clip_video_encode

import clip

import numpy as np

def contrastive_loss(video_embeddings, text_embeddings, temperature = 0.9):
    logits = text_embeddings @ video_embeddings.T # shape: (batch_size x batch_size)
    
    text_similarity = text_embeddings @ text_embeddings.T # shape: (batch_size x batch_size)
    video_similarity = video_embeddings @ video_embeddings.T # shape: (batch_size x batch_size)
    
    targets = F.softmax(
        (text_similarity + video_similarity) / 2 * temperature, dim=-1 # shape: (batch_size x batch_size)
    )
    
    text_loss = F.cross_entropy(logits, targets, reduction = 'none') # shape: batch_size
    video_loss = F.cross_entropy(logits.T, targets.T, reduction = 'none') # shape: batch_size
    
    total_loss = (
        text_loss + video_loss
    ) / 2 # shape: batch_size
    
    return total_loss.mean() # scalar

def clip_encode_dir(data_dir, out_dir, take_every=5):
    VIDS = glob.glob(f'{data_dir}/*.mp4')
    clip_video_encode(VIDS, out_dir, take_every)

def clip_encode_captions(model, captions, out_dir, device = 'cpu'):
    text = clip.tokenize(
        list(captions.values())
    ).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)

    for idx, key in enumerate(
        list(
            captions.keys()
        )
    ):
        np.save(f'{out_dir + key}.npy', text_features[idx].numpy())
    
def pad_video_frames(frames, seq_len):

    num_frames, frames_dim = frames.shape
    out_frames = torch.zeros(seq_len, frames_dim)
    out_frames[0:num_frames] = frames
    
    return out_frames, num_frames

def get_last_frame_from_model_output(model_output, last_frame_idxs):
    out = [
        model_output[i, last_frame_idxs[i], :]
        for i in range(len(last_frame_idxs))
    ]
    return torch.stack(out)

if __name__ == '__main__':

    data_dir = 'data/'
    video_dir = 'embeddings/video'
    clip_encode_dir(data_dir, video_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    captions = {
        'backyard': 'kid playing with weapons in his backyard',
        'beatbox': 'a man beatboxing',
        'dance': 'a group of men dancing with a casket',
        'man_looking_into_cam': 'a man staring into the camera',
        'squid_games': 'hundreds of people competing on a game show',
        'iphone_recording': 'iphone recording of a man in his bed'
    }
    out_dir = 'embeddings/text/'
    clip_encode_captions(model, captions, out_dir, device)
