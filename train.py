import torch.optim as optim
import torch
from model import CLIPFormer
from data import CLIPEmbeddingsDataset
from torch.utils.data import DataLoader
from utils import contrastive_loss, get_last_frame_from_model_output

embeddings_vid_path = 'embeddings/video/'
embeddings_txt_path = 'embeddings/text/'

train_set = CLIPEmbeddingsDataset(embeddings_vid_path, embeddings_txt_path)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

net = CLIPFormer(n_layers=2)
criterion = contrastive_loss
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        frames_emb, txt_emb, last_frame_idx = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        self_attended = net(frames_emb)
        
        last_frame = get_last_frame_from_model_output(self_attended, last_frame_idx)
        
        loss = criterion(txt_emb, last_frame)
        print(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')