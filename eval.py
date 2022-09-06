from model import CLIPFormer, ResidualCLIPFormer
import torch
import os
from data import CLIPEmbeddingsDataset, VideoEmbeddingDataset, ResidualCLIPFormerDataset, EvalResidualCLIPFormerDataset
from utils import pad_video_frames, get_vid_embedding_from_model_output, get_non_padded_frames, process_residuals
import clip
import numpy as np
from torch.utils.data import DataLoader
from preprocess import get_captions_df_from_json
import pandas as pd
import json
import copy

def compute_video_embeddings_residual(eval_test_loader, model, window_size=20, device='cuda'):
    model.eval()
    predicted_embeddings = []
    fnames = []
    with torch.no_grad():
        for i, data in enumerate(eval_test_loader, 0):
            frames_padded, window_frames, num_frames, video_id = data
            frames = get_non_padded_frames(frames_padded, num_frames)
            batch_size = frames_padded.shape[0]

            mask = torch.tensor([
                [0  if j < num_frames[i] else 1 for j in range(model.seq_len)]
                for i in range(len(num_frames))
            ]).float().to('cuda')

            mean_clip_embeddings = torch.stack([
                torch.mean(vid, axis=-2) for vid in frames
            ])

            for idx in range(len(window_frames)):
                window_frames[idx] = torch.roll(window_frames[idx], 1, 0)
                window_frames[idx][0] = mean_clip_embeddings[idx]
                
            residuals = torch.mean(model(window_frames, mask = mask), axis=-2)
            vid_embedding = (mean_clip_embeddings + residuals).half()

            fnames.extend(video_id)
            predicted_embeddings.extend(vid_embedding)
            
    predicted_embeddings = torch.stack(predicted_embeddings)
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

def get_top_k_video_ids_for_caption(caption, test_loader, model, clip_model, embedding_fn, k=5, device='cuda', embeddings=None, fnames=None):
    if embeddings == None:
        predicted_embeddings, fnames = embedding_fn(test_loader, model, device)
    else:
        predicted_embeddings = embeddings
        fnames = fnames
    
    text = clip.tokenize([caption]).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text)

    similarities = text_features.float() @ predicted_embeddings.T.float()
    top_idxs = torch.topk(similarities, k).indices.cpu().numpy()
    
    ids = np.asarray(fnames)[top_idxs][0]
    return [x.split('.npy')[0] + '.mp4' for x in ids]

if __name__ == '__main__':
    torch.manual_seed(0)
    ckpt = torch.load('../dev/updated_residual_ckpts/epoch_1.pth')
    n_layers = 4
    n_heads = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    video_clip_model = ResidualCLIPFormer(n_layers=n_layers, n_heads=n_heads).to(device)

    video_clip_model.load_state_dict(ckpt)

    embeddings_test_vid_path = 'video_clip/embeddings/test/video/'
    test_txt_embed_path = 'video_clip/test_word_embeds.pth'

    embeddings_train_vid_path = 'video_clip/embeddings/train/video/'
    train_txt_embed_path = 'video_clip/train_word_embeds.pth'

    eval_train_set = EvalResidualCLIPFormerDataset(embeddings_train_vid_path, train_txt_embed_path)
    eval_train_loader = DataLoader(eval_train_set, batch_size=16, shuffle=False)
    
    eval_test_set = EvalResidualCLIPFormerDataset(embeddings_test_vid_path, test_txt_embed_path)
    eval_test_loader = DataLoader(eval_test_set, batch_size=16, shuffle=False)
    
    train_embeddings, train_fnames = compute_video_embeddings_residual(eval_train_loader, video_clip_model, device='cuda')
    print(len(train_embeddings))
    np.save('video_clip/embeddings/predicted/embeds_train.npy', train_embeddings.cpu().numpy())
    np.save('video_clip/embeddings/predicted/fnames_train.npy', train_fnames)

    test_embeddings, test_fnames = compute_video_embeddings_residual(eval_test_loader, video_clip_model, device='cuda')
    print(len(test_embeddings))
    np.save('video_clip/embeddings/predicted/embeds_test.npy', test_embeddings.cpu().numpy())
    np.save('video_clip/embeddings/predicted/fnames_test.npy', test_fnames)

    #metrics = eval_model(test_loader, video_clip_model, compute_video_embeddings_residual, device='cuda')
    #print(metrics)
    #caption = 'eating noodles'
    #vids = get_top_k_video_ids_for_caption(caption, None, None, clip_model, None, embeddings=embeddings, fnames=fnames)
   # print(vids)