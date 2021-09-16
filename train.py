import torch
import numpy as np
from dataset import FRDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from model import SiameseNet
from loss import TripletLoss
import warnings


warnings.filterwarnings("ignore")

PATH=config.PATH
epochs=config.EPOCHS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=SiameseNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
criterion=TripletLoss()

print(model)

dataset = FRDataset(
    'csv/train_imgs_triplet.csv',
    transform=config.transform
)

l = len(dataset)
tr_size = int(l*0.8)
val_size = (l-tr_size)


train_set, val_set = torch.utils.data.random_split(dataset, [tr_size, val_size])


train_loader = DataLoader(dataset=train_set,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True, 
                          drop_last=True,
                          num_workers=0)


val_loader = DataLoader(dataset=val_set,
                        batch_size=config.BATCH_SIZE,
                        shuffle=True,
                        drop_last=True,
                        num_workers=0)

e_loss = []
e_val_loss = []
e_val_score = []
n_iter = 0
n_epochs_stop = 60
epochs_no_improve = 0
early_stop = True
min_val_loss = np.Inf
e = 0


for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    running_loss = 0.0
    running_score = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
        for (anchor, positive, negative) in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")

            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_out, pos_out, neg_out = model.forward_triple(anchor, positive, negative)

            loss = criterion(anchor_out, pos_out, neg_out)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    print(f"Train - Loss: {epoch_loss:.4f}")
    e_loss.append(epoch_loss)
    e += 1

    with torch.no_grad():
        val_loss = 0
        for batch_idx, (anchor, positive, negative) in enumerate(val_loader):
            
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_out, pos_out, neg_out = model.forward_triple(anchor, positive, negative)

            loss = criterion(anchor_out, pos_out, neg_out)
            val_loss += loss.item()

        val_epoch_loss = val_loss / len(val_loader)
        print(f"Val Loss: {val_epoch_loss:.4f}")
        e_val_loss.append(val_epoch_loss)

        torch.save(model.state_dict(), PATH)
        print('Model Saved at:', str(PATH))
    
        if val_epoch_loss < min_val_loss:
            print('val_loss<min_val_loss', min_val_loss)
            epochs_no_improve = 0
            min_val_loss = val_epoch_loss

        else:
            epochs_no_improve += 1
        n_iter += 1
        
        if epoch > 5 and epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            early_stop = True
            break
        else:
            continue
        
