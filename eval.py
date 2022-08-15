from model import CLIPFormer, ResidualCLIPFormer
import torch
import os
from data import CLIPEmbeddingsDataset, VideoEmbeddingDataset, ResidualCLIPFormerDataset
from utils import pad_video_frames, get_vid_embedding_from_model_output, get_non_padded_frames, process_residuals
import clip
import numpy as np
from torch.utils.data import DataLoader
from preprocess import get_captions_df_from_json
import pandas as pd
import json
import copy

def compute_video_embeddings_residual(test_loader, model, window_size=20, device='cuda'):
    model.eval()
    predicted_embeddings = []
    fnames = []
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            
            frames_padded, window_frames, txt_embedding, num_frames, video_id = data
            frames = get_non_padded_frames(frames_padded, num_frames)

            residuals = model(window_frames)
       
            resid = process_residuals(frames, residuals, window_size)

            video_embeddings = []
            for vid in resid:
                video_embeddings.append(torch.mean(vid, axis=0))
            vid_embedding = torch.stack(video_embeddings).half()

            predicted_embeddings.extend(vid_embedding)
            
    predicted_embeddings = torch.stack(predicted_embeddings)[::window_size]
    return predicted_embeddings, fnames

def compute_video_embeddings(test_loader, model, device='cuda'):
    model.eval()
    predicted_embeddings = []
    fnames = []

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            frames_emb, last_frame_idx, fname = data
            
            fnames.extend(fname)

            self_attended = model(frames_emb)
            last_frame = get_vid_embedding_from_model_output(self_attended, last_frame_idx)
            predicted_embeddings.extend(last_frame)
            
    predicted_embeddings = torch.stack(predicted_embeddings)
    return predicted_embeddings, fnames

def eval_model(test_loader, video_clip_model, embedding_fn, device='cuda'):
    embeddings, fnames = embedding_fn(test_loader, video_clip_model, device='cuda')
    text_features = torch.load('video_clip/test_word_embeds.pth')

    sim_tensor = text_features @ embeddings.t()

    return get_metrics(sim_tensor)

def get_metrics(sim_tensor, top_k = [1,5,10], return_ranks = False):
    stacked_sim_matrices = sim_tensor.permute(1,0,2)
    first_argsort = torch.argsort(stacked_sim_matrices, dim = -1, descending= True)
    second_argsort = torch.argsort(first_argsort, dim = -1, descending= False)

    ranks = torch.flatten(torch.diagonal(second_argsort, dim1 = 1, dim2 = 2))

    permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(permuted_original_data), torch.isnan(permuted_original_data))
    valid_ranks = ranks[mask]
    
    if return_ranks:
        return list_recall(valid_ranks, top_k), valid_ranks
    else:
        return list_recall(valid_ranks, top_k)
    
def list_recall(lst, top_k):
    if not torch.is_tensor(lst):
        lst = torch.tensor(lst)

    lst = lst.cpu()
    results = {f"R@{k}" : float(torch.sum(lst < k) * 100 / len(lst)) for k in top_k}
    results["Median_Rank"] = float(torch.median(lst + 1))
    results["Mean_Rank"] = float(np.mean(lst.numpy() + 1))
    results["Std_Rank"] = float(np.std(lst.numpy() + 1))
    return results

def compute_topk_acc_old(captions_df_path, test_loader, model, clip_model, device='cuda', k=5, embeddings=None, fnames=None):
    captions_df = pd.read_csv(captions_df_path)[['caption', 'video_id']]
    chunk_size = 10000
    if embeddings == None or fnames == None:
        embeddings, fnames = compute_video_embeddings(test_loader, model, device)
    num_correct = 0
    for i in range(0, len(captions_df), chunk_size):
        df = captions_df.iloc[i:min(i+chunk_size,len(captions_df))]
    
        captions = list(df['caption'])
        video_ids = list(df['video_id'])

        text = clip.tokenize(captions).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text) # might OOM

        similarities = text_features.float() @ embeddings.t().float()
        top_k_indices = torch.topk(similarities, k=k, dim=1).indices
        
        for idx, row in enumerate(top_k_indices):
            
            vids = np.asarray(fnames)[row.cpu().numpy()]
            correct_video_id = video_ids[idx]
            if correct_video_id + '.npy' in vids:
                num_correct += 1

    return num_correct/len(captions_df)

def test_30th_frame_method(test_loader):
    predicted_embeddings = []
    fnames = []
    for i, data in enumerate(test_loader, 0):
        frames_emb, last_frame_idx, fname = data
        predicted_embeddings.extend(frames_emb[:, 6])
        fnames.extend(fname)

    predicted_embeddings = torch.stack(predicted_embeddings)
    return predicted_embeddings, fnames

def average_embed(test_loader):
    predicted_embeddings = []
    fnames = []
    for i, data in enumerate(test_loader, 0):
        frames_emb, last_frame_idx, fname = data
        predicted_embeddings.extend(torch.mean(frames_emb, axis=-2))
        fnames.extend(fname)

    predicted_embeddings = torch.stack(predicted_embeddings)
    return predicted_embeddings, fnames

def get_top_k_video_ids_for_caption(caption, test_loader, model, clip_model, k=5, device='cuda', embeddings=None):
    fnames = []
    if embeddings == None:
        predicted_embeddings, fnames = compute_video_embeddings(test_loader, model, device)
    else:
        predicted_embeddings = embeddings
    
    text = clip.tokenize([caption]).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text)

    similarities = text_features.float() @ predicted_embeddings.T.float()
    top_idxs = torch.topk(similarities, k).indices.cpu().numpy()
    ids = np.asarray(fnames)[top_idxs][0]
    return [x.split('.npy')[0] + '.mp4' for x in ids]

if __name__ == '__main__':
    torch.manual_seed(0)
    ckpt = torch.load('../dev/ckpts_8_mha_64_dim/16_heads_512_dim_8_layers_epoch_100_no_mask_model_ckpt.pth')
    n_layers = 8
    n_heads = 16
    #attn_dim = 512
    device = 'cuda'

    video_clip_model = ResidualCLIPFormer(n_layers=n_layers, n_heads=n_heads).to(device)

    #video_clip_model.load_state_dict(ckpt)

    embeddings_test_vid_path = 'video_clip/embeddings/test/video/'
    txt_embed_path = 'video_clip/test_word_embeds.pth'
    test_set = ResidualCLIPFormerDataset(embeddings_test_vid_path, txt_embed_path)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    
    metrics = eval_model(test_loader, video_clip_model, compute_video_embeddings_residual, device='cuda')