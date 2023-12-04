import cProfile
import gc

import pytorch_lightning as pl
import torch
import tqdm
import os
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models


def collate_fn(batch):
    batch_size = sum([1 for b in batch if b is not None])
    max_options = max(len(b[2]) for b in batch if b is not None)
    anchors = torch.empty((batch_size,20,90,8,8))
    positives = torch.empty((batch_size,12,8,8))
    negatives = torch.zeros((batch_size,max_options,12,8,8))
    negative_lens = torch.zeros((batch_size)).long()
    encodings = torch.empty((batch_size,50,8,8))
    padding = torch.zeros((batch_size,20), dtype = torch.bool)
    num_options = torch.zeros((batch_size)).long()
    i = 0
    for batch_index in range(len(batch)):
        if batch[batch_index] is None or batch[batch_index][3] == 0:
            raise Exception('Batch is None')
        padded_anchor = torch.zeros((20,90,8,8))
        len_unpadded_anchor = batch[batch_index][0].size(0)
        padded_anchor[:len_unpadded_anchor,:,:,:] = batch[batch_index][0]
        pos = batch[batch_index][1]
        negative_list = batch[batch_index][2]
        negative_len = batch[batch_index][3]
        assert negative_len != 0
        player_encoding = batch[batch_index][4]

        for j,neg in enumerate(negative_list):
            negatives[i,j,:,:,:] = neg
        anchors[i,:,:,:,:] = padded_anchor
        positives[i,:,:,:] = pos
        negative_lens[i] = torch.LongTensor([negative_len])
        encodings[i,:,:,:] = player_encoding
        padding[i,len_unpadded_anchor:] = 1
        num_options[i] = torch.LongTensor([len(negative_list)])
        i += 1
    return anchors,positives,negatives,negative_lens,encodings,padding.bool(),num_options

def pick_collate_fn(batch):
    batch_size = sum([1 for b in batch if b is not None])
    anchors = torch.empty((batch_size,20,90,8,8))
    positives = torch.empty((batch_size,12,8,8))
    negatives = torch.zeros((batch_size,num_choices,12,8,8))
    negative_lens = torch.zeros((batch_size)).long()
    encodings = torch.empty((batch_size,50,8,8))
    sequence_lengths = torch.zeros((batch_size)).long()
    num_options = torch.zeros((batch_size)).long()
    i = 0
    for batch_index in range(len(batch)):
        if batch[batch_index] is None or batch[batch_index][3] == 0:
            raise Exception('Batch is None')
        padded_anchor = torch.zeros((20,90,8,8))
        len_unpadded_anchor = batch[batch_index][0].size(0)
        padded_anchor[:len_unpadded_anchor,:,:,:] = batch[batch_index][0]
        padded_anchor = padded_anchor.view(20*90,8,8)
        pos = batch[batch_index][1]
        negative_list = batch[batch_index][2]
        negative_len = batch[batch_index][3]
        player_encoding = batch[batch_index][4]

        choices[i,0,:,:,:] = pos
        anchors[i,:,:,:] = padded_anchor
        choice_lengths[i] = torch.LongTensor([negative_len+1])
        encodings[i,:,:,:] = player_encoding
        j = 1
        for neg in negative_list:
            assert not torch.equal(pos,neg)
            choices[i,j,:,:,:] = neg
            j += 1
        i += 1
    return anchors,choices,choice_lengths, encodings

def validation_loop(network,val_loader,epoch):
    with torch.no_grad():
        network.eval()
        bar = tqdm.tqdm(enumerate(val_loader),\
                total = len(val_loader), mininterval = 1, desc = 'Validation')
        for i,batch in bar:
            if epoch == 0 and i >= len(val_loader)//10:
                break

            anchor,positive,negative_list,negative_len,player,padding,num_options = batch
            neg = negative_list[:,0,:,:,:]

            loss = network.validation_step((anchor,positive,neg,negative_len,player,padding),i)
            loss = loss.item()
            network.eval_losses.append(loss)
            if i % 10 == 0:
                network.writer.add_scalar('eval_loss',loss,epoch*len(val_loader)+i)
                bar.set_postfix({'Loss': loss})
        return network.validation_epoch_end(epoch)

def get_hardest_triplet(batch):
    anchor,positive,negative_list,negative_len,player,padding,num_options = batch
    batch_size = num_options.size(0)
    distances = network.get_distances_of_options((anchor,player,padding),negative_list)
    for j in range(batch_size):
        distances[j,num_options[j]:] = float('inf')
    closest_negative = torch.argmin(distances, dim = 1)
    negative_list = negative_list.to(network.device)
    neg = negative_list[torch.arange(closest_negative.size(0)),closest_negative,:,:,:]
    return anchor,positive,neg,negative_len,player,padding
    
def training_loop(network):
    #hyperparameters
    patience = 3
    num_choices = 3

    network = network.to(network.device)
    val_data = models.Siamese_RBC_dataset(val_path,num_choices = 1,max_samples = val_max_samples)
    val_loader = DataLoader(val_data,batch_size,shuffle = False, persistent_workers=True, pin_memory = True,collate_fn= collate_fn, num_workers=num_workers)
    train_data = models.Siamese_RBC_dataset(train_path,num_choices = num_choices,max_samples = max_samples)
    train_loader = DataLoader(train_data,batch_size,shuffle = True, persistent_workers=True, pin_memory = True, collate_fn= collate_fn, num_workers=num_workers)

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_losses.append(validation_loop(network,val_loader,0))
    epoch = 0
    no_improvement = 0
    while True:
        network.train()

        bar = tqdm.tqdm(enumerate(train_loader),\
                total = len(train_loader), mininterval = 1, desc = 'Training')
        for i,batch in bar:
            if epoch == 0:
                batch = get_hardest_triplet(batch)
            else:
                batch = get_hardest_triplet(batch)
            loss = network.training_step(batch,i)
            loss = loss.item()
            network.train_losses.append(loss)
            if i % 10 == 0:
                network.writer.add_scalar('train_loss',loss,epoch*len(train_loader)+i)
                bar.set_postfix({'Loss': loss})


        new_loss = network.training_epoch_end(epoch)
        epoch_train_losses.append(new_loss)
        print(f'Training loss for epoch {epoch}: {new_loss}')

        
        new_loss = validation_loop(network,val_loader, epoch +1)
        #Check for early stopping
        if len(epoch_val_losses) != 0 and new_loss >= min(epoch_val_losses):
            print('No improvement for this epoch')
            no_improvement += 1
            num_choices += 2
            print(f'Now using {num_choices} choices')
            train_data = models.Siamese_RBC_dataset(train_path,num_choices = num_choices,max_samples = max_samples)
            train_loader = DataLoader(train_data,batch_size,shuffle = True, persistent_workers=True, pin_memory = True, collate_fn= collate_fn, num_workers=num_workers)
        else:
            torch.save(network.state_dict(),'Siamese_Network_New.pt')
        epoch_val_losses.append(new_loss)
        print(f'Validation loss for epoch {epoch}: {new_loss}')

        if no_improvement == patience:
            break
        epoch += 1
        gc.collect()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    batch_size = 1024
    num_workers = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    #create network
    network = models.Siamese_Network(embedding_dimensions = 512, create_writer = True, device = device)
    network.optimizer = torch.optim.AdamW(network.parameters(), lr = 1e-4)

    #create data
    max_samples = None
    val_max_samples = int(max_samples*0.1) if max_samples is not None else None
    train_path = 'data/siamese/train/'
    val_path = 'data/siamese/val/'

    training_loop(network)
