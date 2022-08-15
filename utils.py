import torch
from torch import nn
import torch.nn.functional as F

import glob
from clip_video_encode import clip_video_encode

import os
import clip
import numpy as np
import pandas as pd

import cv2
from PIL import Image

def cyclic_contrastive_loss(video_embeddings, text_embeddings, temperature=0.9, lambda1=0.01, lambda2=0.01):
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
    batch_size = len(logits)
    inmodal_cyclic_loss = (video_similarity - text_similarity).square().mean()
    #print(inmodal_cyclic_loss)
    logits_text_per_image = video_embeddings @ text_embeddings.t()
    logits_image_per_text = logits_text_per_image.t()

    crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean()
    #print(crossmodal_cyclic_loss)
    return total_loss.mean() + lambda1 * inmodal_cyclic_loss + lambda2 * crossmodal_cyclic_loss# scalar

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

def get_non_padded_frames(batch, idxs):
    return [
        batch[i, 0:idxs[i].item(), :]
        for i in range(len(idxs))
    ]

def get_uncertainty_for_batch(frames_emb, txt_emb, last_frame_idx, temperature=5):
    frames_np = get_non_padded_frames(frames_emb, last_frame_idx)
    u = []
    for vid in frames_np:
        unc = -torch.var(txt_emb @ vid.t())
        u.append(unc)
    u = torch.stack(u)
    uncertainty = F.softmax(u/temperature)
    
    return uncertainty

def contrastive_loss_with_uncertainty(video_embeddings, text_embeddings, uncertainty_weights, temperature = 0.9):
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
    
    uncertainty_loss = torch.mul(total_loss, uncertainty_weights).sum()/uncertainty_weights.sum()

    return uncertainty_loss # scalar
    
def concatenate_captions(model, captions, out_dir, device='cuda'):
    captions_df = pd.DataFrame({'video_id_inx': list(captions.keys()), 'caption': list(captions.values())})
    captions_df['video_id'] = captions_df['video_id_inx'].map(lambda x: ''.join(x.split('_')[:-1]))

    captions_grouped = captions_df.groupby('video_id')

    for video in captions_grouped:
        fname, df = video
        caption = ' '.join(list(df['caption']))

        text = clip.tokenize([caption]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text).cpu()
            fname = video['video_id']
            np.save(f'{out_dir + fname}.npy', text_features.numpy())


def average_captions(model, captions, out_dir, clip_model, device='cuda'):
    captions_df = pd.DataFrame({'video_id_inx': list(captions.keys()), 'caption': list(captions.values())})
    captions_df['video_id'] = captions_df['video_id_inx'].map(lambda x: ''.join(x.split('_')[:-1]))

    captions_grouped = captions_df.groupby('video_id')
    
    for video in captions_grouped:
        fname, df = video
        captions = list(df['caption'])
        
        text = clip.tokenize(captions).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            avg = torch.mean(text_features, axis=-2).cpu()
            np.save(f'{out_dir + fname}.npy', avg.numpy())

def clip_encode_captions(model, captions, out_dir, device = 'cuda', handle_dupes = 'average'):
    if handle_dupes == 'average':
        average_captions(model, captions, out_dir, device)
    elif handle_dupes == 'concatenate':
        concatenate_captions(model, captions, out_dir, device)
    else:
        text = clip.tokenize(
            list(captions.values())
        ).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text).cpu()

        for idx, key in enumerate(
            list(
                captions.keys()
            )
        ):
            np.save(f'{out_dir + key}.npy', text_features[idx].numpy())

def load_frames_from_fpath(fpath, out_fps=None):
    cap = cv2.VideoCapture(fpath)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
        
    if out_fps is not None:
        return buf[::max(int(fps//out_fps), 1)]
    else:
        return buf

    cap.release()

def clip_encode_frames(model, preprocess, frames, device='cuda'): # pads to max_frames
    fr = []
    num_frames = len(frames)
    embed_dim = 512
    preprocessed = torch.stack([
        preprocess(
            Image.fromarray(
                np.uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            )
        ).to(device)
        for frame in frames
    ])
    with torch.no_grad():
        image_features = model.encode_image(preprocessed)
    
    image_features.reshape(num_frames, embed_dim)
    return image_features

def clip_encode_video_file(model, preprocess, fpath, out_dir, out_fps=None, device='cuda'):
    frames = load_frames_from_fpath(fpath, out_fps=out_fps)
    file_name = fpath.split('/')[-1].split('.mp4')[0]
    clip_encodings = clip_encode_frames(model, preprocess, frames, device)

    save_path = f'{out_dir}{file_name}.npy'
    np.save(save_path, clip_encodings.cpu().numpy())

def clip_encode_video_dir(model, preprocess, data_dir, out_dir, fps=1, device='cuda'):
    vids = os.listdir(data_dir)
    for idx, video in enumerate(vids):
        if idx%100 == 0:
            print(idx)
        clip_encode_video_file(model, preprocess, data_dir + video, out_dir, out_fps=fps, device=device)

def has_no_frames(filepath):
    cap = cv2.VideoCapture(filepath)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length == 0

def get_no_frame_videos(dir):
    video_names = [
        video
        for video in os.listdir(dir)
        if has_no_frames(dir + video)
    ]
    return video_names

def pad_video_frames(frames, seq_len):

    num_frames, frames_dim = frames.shape
    out_frames = torch.zeros(seq_len, frames_dim)
    out_frames[0:num_frames] = frames
    
    return out_frames, num_frames

def get_vid_embedding_from_model_output(model_output, last_frame_idxs):
    out = [
        torch.mean(model_output[i, 0:last_frame_idxs[i], :], axis=-2)
        for i in range(len(last_frame_idxs))
    ]
    return torch.stack(out)

if __name__ == '__main__':
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    clip_encode_video_file(model, preprocess, 'data/TrainValVideo/video0.mp4', 'data/', out_fps=1, device=device)

def add_residuals(vid, residuals, window_size):
    out_vid = vid
    for idx, i in enumerate(range(0, len(vid), window_size)):
        out_vid[i:min(i+window_size, len(vid))] += residuals[idx]
    return out_vid
    

def process_residuals(batch, residuals, window_size):
    processed_batch = batch
    for idx, vid in enumerate(batch):
        processed_batch[idx] = add_residuals(vid, residuals[idx], window_size)

    return processed_batch