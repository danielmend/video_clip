import json
from utils import clip_encode_video_dir, clip_encode_captions
import pandas as pd
import clip
import torch
import numpy as np
np.random.seed(42)

def get_captions_df_from_json(captions_json, num_sample=2, concat=False):
    with open(captions_json, 'r') as f:
        data = json.load(f)

    captions_df = pd.DataFrame(data['sentences'])
    
    captions_df['sub_id'] = captions_df.groupby('video_id').cumcount()
    captions_df['video_id_inx'] = captions_df['video_id'] + captions_df['sub_id'].map(lambda x: '_'+str(x))
    captions_df_sampled = captions_df.groupby('video_id').apply(lambda x: x.sample(min(num_sample, len(x)))).reset_index(drop=True)
    fname = captions_json.split('.json')[0]
    if concat:
        concat_captions = captions_df_sampled.groupby(['video_id'])['caption'].apply(lambda x: ' '.join(x)).reset_index()
        captions_df_out = concat_captions.set_index('video_id')['caption']
    else:
        concat_captions = captions_df_sampled
        captions_df_out = concat_captions.set_index('video_id_inx')['caption']
   
    concat_captions.to_csv(f'{fname}.csv')

    return captions_df_out


def load_captions_json_into_dir(model, captions_json, output_dir, device = 'cuda', num_sample=2, concat=False):
    captions_df = get_captions_df_from_json(captions_json, num_sample=num_sample, concat=concat)
    print(captions_df.head())
    chunk_size = 10000
    for i in range(1, len(captions_df), chunk_size):
    
        df = captions_df.iloc[i:min(i+chunk_size,len(captions_df))]
        clip_encode_captions(model, df.to_dict(), output_dir, device, handle_dupes = 'none')

if __name__ == '__main__':
    train_json = 'data/train_val_videodatainfo.json'
    train_videos = 'data/TrainValVideo/'

    train_caption_dir = 'video_clip/embeddings/train/text/'
    train_videos_dir = 'video_clip/embeddings/train/video/'

    test_json = 'data/test_videodatainfo.json'
    test_videos = 'data/TestVideo/'

    test_caption_dir = 'video_clip/embeddings/test/text/'
    test_videos_dir = 'video_clip/embeddings/test/video/'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    print('encoding train data')
    load_captions_json_into_dir(model, train_json, train_caption_dir, device, num_sample=2, concat=True)
    print('captions encoded')
    #clip_encode_video_dir(model, preprocess, train_videos, train_videos_dir, fps=None, device=device)
    print('videos encoded')

    print('encoded train')
    print('encoding test data')

    load_captions_json_into_dir(model, test_json, test_caption_dir, device, num_sample=100, concat=False)
    print('captions encoded')
    #clip_encode_video_dir(model, preprocess, test_videos, test_videos_dir, fps=None, device=device)
    print('videos encoded')

    print('encoded test')