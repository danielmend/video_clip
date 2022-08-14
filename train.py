import torch.optim as optim
import torch
from model import CLIPFormer
from data import CLIPEmbeddingsDataset, VideoEmbeddingDataset
from torch.utils.data import DataLoader
from utils import contrastive_loss, cyclic_contrastive_loss, get_vid_embedding_from_model_output, get_uncertainty_for_batch, contrastive_loss_with_uncertainty
import numpy as np 
from torch.optim.lr_scheduler import StepLR
import wandb
import clip
from eval import compute_topk_acc

wandb.init(project="video-clip")

n_epochs = 300
lr = 1e-4
train_batch_size = 16
test_batch_size = 16

n_layers = 8
n_heads = 16
attn_dim = 512
dropout_p = 0
device = 'cuda'
uncertainty = True

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

wandb.config = {
  "learning_rate": lr,
  "epochs": n_epochs,
  "train_batch_size": train_batch_size,
  "test_batch_size": test_batch_size
}

embeddings_train_vid_path = 'video_clip/embeddings/train/video/'
embeddings_train_txt_path = 'video_clip/embeddings/train/text/'

embeddings_test_vid_path = 'video_clip/embeddings/test/video/'
embeddings_test_txt_path = 'video_clip/embeddings/test/text/'

train_set = CLIPEmbeddingsDataset(embeddings_train_vid_path, embeddings_train_txt_path)
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)

test_set = CLIPEmbeddingsDataset(embeddings_test_vid_path, embeddings_test_txt_path)
test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

test_vid_set = VideoEmbeddingDataset(embeddings_test_vid_path)
test_vid_loader = DataLoader(test_vid_set, batch_size=test_batch_size, shuffle=False)

net = CLIPFormer(n_layers=n_layers, attn_dim=attn_dim, n_heads=n_heads, using_tel=True, mask=False).to(device)
print(f'num params: {net._num_params}')

wandb.watch(net, log='all', log_freq=100)
criterion = contrastive_loss_with_uncertainty
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
scheduler = StepLR(optimizer, 20, gamma = 0.5)

def train(epoch, train_loader, net, optimizer, criterion):
    net.train()
    running_loss = 0.0
    print_loss = 0
    for i, data in enumerate(train_loader, 0):
        
        frames_emb, txt_emb, last_frame_idx, _ = data
        if uncertainty:
            uncertainty_weights = get_uncertainty_for_batch(frames_emb, txt_emb, last_frame_idx)

        optimizer.zero_grad()

        self_attended = net(frames_emb)
        last_frame = get_vid_embedding_from_model_output(self_attended, last_frame_idx)
        

        if uncertainty:
            loss = criterion(txt_emb, last_frame, uncertainty_weights)
        else:
            loss = criterion(txt_emb, last_frame)
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

def test(epoch, test_loader, net, criterion):
    net.eval()
    running_loss = 0
    
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            frames_emb, txt_emb, last_frame_idx, _ = data

            if uncertainty:
                uncertainty_weights = get_uncertainty_for_batch(frames_emb, txt_emb, last_frame_idx)


            self_attended = net(frames_emb)
            last_frame = get_vid_embedding_from_model_output(self_attended, last_frame_idx)
            
            if uncertainty:
                loss = criterion(txt_emb, last_frame, uncertainty_weights)
            else:
                loss = criterion(txt_emb, last_frame)
            wandb.log({"eval_loss": loss})
            
            running_loss += loss.item()
        
        test_loss = running_loss/len(test_loader)
    
    print(f'Test Loss: {test_loss}')
    return test_loss

train_losses = []
test_losses = []
captions_df_path = 'data/test_videodatainfo.csv'
k = 5
for epoch in range(n_epochs):

    print('learning rate:', scheduler.get_last_lr())
    wandb.log({'last_lr': scheduler.get_last_lr()[0]})

    train_loss = train(epoch, train_loader, net, optimizer, criterion)
    wandb.log({'train_epoch_loss': train_loss, 'epoch': epoch})

    test_loss = test(epoch, test_loader, net, criterion)

    wandb.log({'eval_epoch_loss': test_loss, 'epoch': epoch})

    test_losses.append(test_loss)
    train_losses.append(train_loss)
    if epoch % 5 == 0:
        print('saving checkpoint...')
        torch.save(net.state_dict(), f'ckpts_8_mha_64_dim/{n_heads}_heads_{attn_dim}_dim_{n_layers}_layers_epoch_{epoch}_no_mask_model_ckpt.pth')
        topk_acc = compute_topk_acc(captions_df_path, test_vid_loader, net, clip_model=clip_model, device='cuda', k=5)
        print(f'Top {k} acc: {topk_acc}')
        wandb.log({'topk_acc': topk_acc, 'epoch': epoch})
    scheduler.step()

np.save('test_losses.npy', np.asarray(test_losses))
torch.save(net.state_dict(), f'ckpts_8_mha_64_dim/{n_heads}_heads_{attn_dim}_dim_{n_layers}_layers_no_mask_final_model_ckpt.pth')

print('Finished Training')
print(f'Min eval loss: epoch {np.argmin(test_losses)}')