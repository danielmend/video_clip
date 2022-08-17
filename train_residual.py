import torch.optim as optim
import torch
from model import CLIPFormer, ResidualCLIPFormer
from data import CLIPEmbeddingsDataset, VideoEmbeddingDataset, ResidualCLIPFormerDataset, EvalResidualCLIPFormerDataset, ConcatenatedCaptionsDataset, TestConcatenatedCaptionsDataset, EvalConcatenatedCaptionsDataset
from torch.utils.data import DataLoader
from utils import contrastive_loss, cyclic_contrastive_loss, get_vid_embedding_from_model_output, get_uncertainty_for_batch, contrastive_loss_with_uncertainty, process_residuals, get_non_padded_frames
import numpy as np 
from torch.optim.lr_scheduler import StepLR
import wandb
import clip
from eval import eval_model, compute_video_embeddings_residual

wandb.init(project="video-clip")

n_epochs = 300
lr = 1e-4
train_batch_size = 16
test_batch_size = 16

n_layers = 4
n_heads = 16

dropout_p = 0
device = 'cuda'
uncertainty = False

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

wandb.config = {
  "learning_rate": lr,
  "epochs": n_epochs,
  "train_batch_size": train_batch_size,
  "test_batch_size": test_batch_size
}

embeddings_train_vid_path = 'video_clip/embeddings/train/video/'
embeddings_train_txt_path = 'video_clip/train_word_embeds.pth'

embeddings_test_vid_path = 'video_clip/embeddings/test/video/'
embeddings_test_txt_path = 'video_clip/test_word_embeds.pth'

train_set = ResidualCLIPFormerDataset(embeddings_train_vid_path, embeddings_train_txt_path)
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)

test_set = ResidualCLIPFormerDataset(embeddings_test_vid_path, embeddings_test_txt_path)
test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True)

eval_test_set = EvalResidualCLIPFormerDataset(embeddings_test_vid_path, embeddings_test_txt_path)
eval_test_loader = DataLoader(eval_test_set, batch_size=16, shuffle=False)

net = ResidualCLIPFormer(n_layers=n_layers, n_heads=n_heads).to(device)
print(f'num params: {net._num_params}')

wandb.watch(net, log='all', log_freq=100)
criterion = cyclic_contrastive_loss
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = StepLR(optimizer, 1, gamma = 0.95)

def train(epoch, train_loader, net, optimizer, criterion, window_size=20):
    net.train()
    running_loss = 0.0
    print_loss = 0
    for i, data in enumerate(train_loader, 0):
        
        optimizer.zero_grad()

        frames_padded, window_frames, txt_embedding, num_frames, video_id = data
        frames = get_non_padded_frames(frames_padded, num_frames)
        batch_size = frames_padded.shape[0]

        mean_clip_embeddings = torch.stack([
            torch.mean(vid, axis=-2) for vid in frames
        ])

        for idx in range(len(window_frames)):
            window_frames[idx] = torch.roll(window_frames[idx], 1, 0)
            window_frames[idx][0] = mean_clip_embeddings[idx]

        residuals = torch.mean(net(window_frames), axis=-2)
        vid_embedding = (mean_clip_embeddings + residuals).half()
        '''
        resid = process_residuals(frames, residuals, window_size)

        video_embeddings = []
        for vid in resid:
            video_embeddings.append(torch.mean(vid, axis=0))
        vid_embedding = torch.stack(video_embeddings).half()
        '''
        loss = criterion(vid_embedding, txt_embedding)
        wandb.log({"train_loss": loss})

        loss.backward()
        optimizer.step()

        print_loss += loss.item()
        running_loss += loss.item()
        print_every = 25
        if i % print_every == print_every-1:
            print(f'[train epoch {epoch + 0}, batch {i+1:5d}] loss: {print_loss / print_every:.3f}')
            print_loss = 0.0

    print(f'Avg training loss: {running_loss/len(train_loader)}')
    return running_loss / len(train_loader)

def test(epoch, test_loader, net, criterion, window_size = 20):
    net.eval()
    running_loss = 0
    
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            frames_padded, window_frames, txt_embedding, num_frames, video_id = data
            frames = get_non_padded_frames(frames_padded, num_frames)
            batch_size = frames_padded.shape[0]

            mean_clip_embeddings = torch.stack([
                torch.mean(vid, axis=-2) for vid in frames
            ])

            for idx in range(len(window_frames)):
                window_frames[idx] = torch.roll(window_frames[idx], 1, 0)
                window_frames[idx][0] = mean_clip_embeddings[idx]
                
            residuals = torch.mean(net(window_frames), axis=-2)
            vid_embedding = (mean_clip_embeddings + residuals).half()

            #resid = process_residuals(frames, residuals, window_size)

            #video_embeddings = []
            #for vid in resid:
            #    video_embeddings.append(torch.mean(vid, axis=0))
            #vid_embedding = torch.stack(video_embeddings).half()
            
            loss = criterion(vid_embedding, txt_embedding)
            wandb.log({"eval_loss": loss})
            
            running_loss += loss.item()
        
        test_loss = running_loss/len(test_loader)
    
    print(f'Test Loss: {test_loss}')
    return test_loss


embeddings_fn = compute_video_embeddings_residual
train_losses = []
test_losses = []

k = 5
for epoch in range(n_epochs):

    print('learning rate:', scheduler.get_last_lr())

    

    train_loss = train(epoch, train_loader, net, optimizer, criterion)
    wandb.log({'train_epoch_loss': train_loss, 'epoch': epoch})

    test_loss = test(epoch, test_loader, net, criterion)

    wandb.log({'eval_epoch_loss': test_loss, 'epoch': epoch})

    test_losses.append(test_loss)
    train_losses.append(train_loss)

    print('saving checkpoint...')
    torch.save(net.state_dict(), f'updated_residual_ckpts/epoch_{epoch}.pth')

    metrics = eval_model(eval_test_loader, net, embeddings_fn, device='cuda')        
    print(f'Top {k} acc: {metrics}')
    metrics['epoch'] = epoch
    wandb.log(metrics)

    scheduler.step()
    
print('Finished Training')
print(f'Min eval loss: epoch {np.argmin(test_losses)}')