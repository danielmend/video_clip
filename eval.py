from model import CLIPFormer
import torch
import os
from data import CLIPEmbeddingsDataset, VideoEmbeddingDataset
from utils import pad_video_frames, get_vid_embedding_from_model_output
import clip
import numpy as np
from torch.utils.data import DataLoader
from preprocess import get_captions_df_from_json
import pandas as pd
import json
import copy

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

def compute_topk_acc(captions_df_path, test_loader, model, clip_model, device='cuda', k=5, embeddings=None, fnames=None):
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
    attn_dim = 512
    device = 'cuda'

    video_clip_model = CLIPFormer(n_layers=n_layers, attn_dim=attn_dim, n_heads=n_heads, mask=False, using_tel=True).to(device)

    video_clip_model.load_state_dict(ckpt)

    embeddings_test_txt_path = 'video_clip/embeddings/test/text/'
    embeddings_test_vid_path = 'video_clip/embeddings/test/video/'
    
    test_set = VideoEmbeddingDataset(embeddings_test_vid_path)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
    
    captions_dict = get_captions_df_from_json('data/test_videodatainfo.json').to_dict()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    #caption = 'hockey footage'
    #vids = get_top_k_video_ids_for_caption(caption, test_loader, video_clip_model, clip_model)
    #print(vids)
    #cat_fpath = 'data/category.txt'
    #test_data_fpath = 'data/test_videodatainfo.json'
    #train_set = VideoEmbeddingDataset('video_clip/embeddings/train/video/')
    #train_loader = DataLoader(train_set, batch_size=16, shuffle=False)
    #pred_embeddings1, fnames1 = compute_video_embeddings(train_loader, video_clip_model)
    pred_embeddings, fnames = compute_video_embeddings(test_loader, video_clip_model)

    overall_embeddings = pred_embeddings.cpu().numpy()#torch.concat([pred_embeddings1, pred_embeddings2], axis=0).cpu().numpy()
    print(overall_embeddings.shape)
    
    overall_fnames = np.asarray(fnames)#np.concatenate([fnames1, fnames2], axis=0)
    print(overall_fnames.shape)
    
    np.save('/home/daniel/dev/video_clip/embeddings/predicted/embed.npy', overall_embeddings)
    np.save('/home/daniel/dev/video_clip/embeddings/predicted/fnames.npy', overall_fnames)


    #captions_df_path = 'data/test_videodatainfo.csv'   
    #new_embeds, fnames = average_embed(test_loader)
    #print(new_embeds.shape)
    #acc = compute_topk_acc(captions_df_path, test_loader, video_clip_model, clip_model=clip_model, device='cuda', k=5, embeddings=None, fnames=None)
    #print(acc)
     
    #acc = compute_topk_acc(captions_df_path, test_loader, video_clip_model, clip_model=clip_model, device='cuda', k=5)
    #print(acc)