import csv
import lzma
import math
import os
import pickle
import random
from utils import *
import time
from collections import defaultdict
from random import shuffle

import chess
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

piece_to_index= {
    'P':0, 'R':1,'N':2,'B':3,'Q':4,'K':5,
    'p':6, 'r':7,'n':8, 'b': 9,'q':10,'k':11
}

class Siamese_RBC_dataset(Dataset):
    def __init__(self, path,num_choices,max_samples = None, shuffle_on_init = True, color = None, mode = ('move','sense')):
        self.path = path
        self.files = []
        for m in mode:
            m_iterator = sorted(os.scandir(f'{path}{m}/'), key = lambda k: random.random()) if shuffle_on_init else os.listdir(f'{path}{m}/')
            j = 0
            for i,file in enumerate(m_iterator):
                name = file.name if type(file) == os.DirEntry else file
                if color is None or color in name:
                    self.files.append(f'{path}{m}/{name}')
                    j += 1
                if max_samples and j == (max_samples//len(mode))-1:
                    break
        if shuffle_on_init:
            shuffle(self.files)
        self.len = len(self.files)
        self.num_choices = num_choices
        self.max_samples = max_samples
        self.num_players, self.player_encoding, self.empty_encoding = Siamese_RBC_dataset.create_player_encoding()
    
    def __len__(self):
        return self.len

    def __getitem__(self,id):
        try:
            with lzma.open(self.files[id], 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(id, e)
            return self.__getitem__(id+1)

        else:
            anchor = data[0]
            positive = board_from_fen(data[1])
            if data[2]:
                if self.num_choices:
                    shuffle(data[2])
                negatives = []
                for b in data[2]:
                    b_tensor = board_from_fen(b)
                    if not torch.equal(positive,b_tensor):
                        negatives.append(b_tensor)
                    if self.num_choices and len(negatives) >= self.num_choices:
                        break

                if len(negatives) > 0:
                    if data[3] in self.player_encoding:
                        return anchor,positive,negatives,len(negatives),self.player_encoding[data[3]],self.empty_encoding
                    else:
                        return anchor,positive,negatives,len(negatives),self.empty_encoding, self.empty_encoding 
                else:
                    return self.__getitem__(id+1)
            else:
                return self.__getitem__(id+1)


    #this creates the encoding for the top 50 most played players
    @staticmethod
    def create_player_encoding(path = 'game_numbers.csv'):
        num_players = 50
        player_dict = {}
        player_encoding = {}
        try:
            with open(path, 'r') as f:
                reader = csv.reader(f, delimiter = ',')
                for line in reader:
                    player_dict[line[0]] = int(line[1])
        except:
            with open('game_numbers.csv', 'r') as f:
                reader = csv.reader(f, delimiter = ',')
                for line in reader:
                    player_dict[line[0]] = int(line[1])
        
        player_tuples = sorted(player_dict.items(), key = lambda x: x[1],reverse = True)
        valid_players = [p[0] for p in player_tuples[:num_players]]
        empty_encoding = torch.zeros((num_players,8,8))
        for i,p in enumerate(valid_players):
            encoding = torch.zeros_like(empty_encoding)
            encoding[i,:,:] = 1
            player_encoding[p] = encoding
        return num_players,player_encoding,empty_encoding

class Embedding_Network_Convolution(nn.Module):
    def __init__(self,input_layers,output_size, num_layers, layer_size):
        super(Embedding_Network_Convolution,self).__init__()

    
        self.input = nn.Conv2d(input_layers,layer_size,kernel_size=3, padding = 'same')
        self.hidden = nn.ModuleList([nn.Conv2d(layer_size,layer_size,kernel_size=3, padding = 'same') for i in range(num_layers)])
        self.output = nn.Conv2d(layer_size,output_size,kernel_size=3, padding = 'same')


    def forward(self,x):
        x = self.input(x)
        x = F.elu(x)
        for m in self.hidden:
            x = m(x)
            x = F.elu(x)

        x = self.output(x)
        x = torch.tanh(x)
        return x

class Siamese_Block_Convolution(nn.Module):
    def __init__(self,in_size,out_size):
        super(Siamese_Block_Convolution,self).__init__()
        self.conv = nn.Conv2d(in_size,out_size,kernel_size=3, padding = 'same')
        self.activation = nn.ELU()
        self.norm = nn.InstanceNorm2d(out_size)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class Siamese_Network(nn.Module):
    def __init__(self,embedding_dimensions, path = None, pre_embed_dim = 128, create_writer = False, device = 'cuda'):
        super(Siamese_Network,self).__init__()
        num_players,self.player_encoding,self.empty_encoding = Siamese_RBC_dataset.create_player_encoding('game_numbers.csv')
        self.reverse_encoding = {torch.argmax(v,dim=0)[0,0].item():k for k,v in self.player_encoding.items()}
        self.embedding_dimensions = embedding_dimensions
        self.train_pick_choices = []
        self.visualize = False
        self.main_block_size = 128

        self.loss_fn = nn.TripletMarginLoss(reduction = 'none')
        self.train_pick_accuracy = []
        self.pick_accuracy = []
        self.pick_choices = []
        self.train_losses = []
        self.eval_losses = []
        self.test_losses = []
        self.pick_distance = []
        self.train_pick_distance = []
        self.choice_num_to_accuracy = defaultdict(list)
        self.top_k_accuracy = defaultdict(list)
        self.player_to_accuracy = defaultdict(lambda: defaultdict(list))
        self.device = device
        self.writer = SummaryWriter()


        self.input_observations = Embedding_Network_Convolution(input_layers = 1850,output_size = pre_embed_dim, num_layers = 5, layer_size = 64)
        self.input_board = Embedding_Network_Convolution(input_layers = 12,output_size = pre_embed_dim, num_layers = 5, layer_size = 64)
        self.first_block = Siamese_Block_Convolution(pre_embed_dim,self.main_block_size)
        self.main_network = nn.ModuleList()
        for _ in range(10):
            self.main_network.append(Siamese_Block_Convolution(self.main_block_size,self.main_block_size))
        self.last_block = nn.Sequential(
            nn.Conv2d(self.main_block_size,64, kernel_size=1, padding = 'same'),
            nn.InstanceNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64,1, kernel_size=1, padding = 'same'),
            nn.InstanceNorm2d(1),
            nn.ELU()
        )
        self.output = nn.Linear(8*8*1,embedding_dimensions)
        if path is not None:
            self.load_state_dict(torch.load(path))
            self.eval()

    def get_distances_of_options(self,full_anchor,options):
        self.eval()
        anchor,player,padding = full_anchor
        B,num_options,_,_,_ = options.shape
        with torch.no_grad():
            anchor = self.anchor_forward(anchor.to(self.device),player.to(self.device),padding.to(self.device))
            anchor = anchor.repeat_interleave(num_options, dim = 0)
            options = self.choice_forward(options.view(B*num_options,12,8,8).to(self.device))
            distances = get_distance(anchor,options)
            distances = distances.view(B,num_options)
        self.train()
        return distances
    
    def forward(self,anchor,positive,negative,player,padding):
        anchor_out = self.anchor_forward(anchor,player,padding)
        positive_out = self.choice_forward(positive)
        negative_out = self.choice_forward(negative)
        return anchor_out,positive_out,negative_out

    def main_network_pass(self,input):
        input = self.first_block(input)
        for block in self.main_network:
            input = input + block(input)
        output = self.last_block(input)
        output = output.view(-1,1*8*8)
        output = self.output(output)
        output = torch.tanh(output)
        return output

    def anchor_forward(self,anchor,player,padding):
        a = anchor.view(-1,1800,8,8)
        p = player.view(-1,50,8,8)
        anchor = self.main_network_pass(self.input_observations(torch.cat((a,p), dim = 1)))
        return anchor

    def choice_forward(self,choice):
        choice = self.main_network_pass(self.input_board(choice))
        return choice
    
    def training_step(self,batch,batch_idx):
        anchor,positive,negative,negative_len,player,padding = batch
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        negative_len = negative_len.to(self.device)
        player = player.to(self.device)
        padding = padding.to(self.device)

        self.optimizer.zero_grad()
        anchor_out,positive_out,negative_out = self(anchor,positive,negative,player,padding)
        loss = self.loss_fn(anchor_out,positive_out,negative_out)
        loss = torch.mean(loss)
        loss.backward()
        self.optimizer.step()
        return loss

    def validation_step(self,batch,batch_idx):
        anchor,positive,negative,lens,player,padding = batch
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        lens = lens.to(self.device)
        player = player.to(self.device)
        padding = padding.to(self.device)
        anchor_out,positive_out,negative_out = self(anchor,positive,negative,player,padding)
        loss = self.loss_fn(anchor_out,positive_out,negative_out)
        loss = torch.mean(loss)
        return loss

    def test_step(self,batch,batch_idx, top_k = (1,), top_k_percentage = None):
        if self.test_picks:
            anchor,positive,choice_list,num_negatives,players,padding,choice_lengths = batch
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            choice_list = choice_list.to(self.device)
            choice_lengths = choice_lengths.to(self.device)
            players = players.to(self.device)
            padding = padding.to(self.device)
            maximum_length = torch.max(choice_lengths).item()+1
            choice_list = torch.cat((positive.unsqueeze(dim=0),choice_list),dim = 1)

            random_indizes = list(range(int(maximum_length)))
            #shuffle(random_indizes)
            try:
                correct_choice = np.argmin(random_indizes)
            except Exception as e:
                raise e
            choice_list = choice_list.view(maximum_length,12,8,8)[random_indizes,:,:,:]
            batch_size = anchor.size(0)
        
            anchors_embedded = self.anchor_forward(anchor,players,padding)
            choices = self.choice_forward(choice_list)
            choices = choices.view(batch_size,maximum_length,self.embedding_dimensions)
            obs_distances = torch.stack([get_distance(anchors_embedded,choices[:,j,:]).cpu() for j in range(choice_lengths)])
            obs_distances = obs_distances.transpose(0,1)
            distances = obs_distances

            for i in range(batch_size):
                if choice_lengths[i] < maximum_length:
                    distances[i,choice_lengths[i]:] = float('inf')
            picks = torch.argmin(distances,dim=1)
            pick_distances = torch.argsort(distances,descending = False,dim= 1)
            index_of_correct_choice = (pick_distances == correct_choice).nonzero(as_tuple=True)[1]
            self.pick_distance.extend(list(index_of_correct_choice))
            self.pick_accuracy.extend([1 if pick == correct_choice else 0 for pick in picks])
            for i in range(picks.shape[0]):
                for k in top_k:
                    if index_of_correct_choice[i] < k:
                        self.top_k_accuracy[k].append(1)
                    else:
                        self.top_k_accuracy[k].append(0)
                
                for k in top_k_percentage:
                    if index_of_correct_choice[i] <= math.ceil(choice_lengths[i].item()*k):
                        self.top_k_accuracy[k].append(1)
                    else:
                        self.top_k_accuracy[k].append(0)
                if picks[i].item() == correct_choice:
                    self.choice_num_to_accuracy[choice_lengths[i].item()].append(1)
                    if torch.sum(players).item() == 64:
                        p = self.player_to_accuracy[torch.argmax(players,dim = 1)[0,0,0].item()]
                        p[choice_lengths[i].item()].append(1)
                    else:
                        p = self.player_to_accuracy[50]
                        p[choice_lengths[i].item()].append(1)

                else:
                    self.choice_num_to_accuracy[choice_lengths[i].item()].append(0)
                    if torch.sum(players).item() == 64:
                        p = self.player_to_accuracy[torch.argmax(players,dim = 1)[0,0,0].item()]
                        p[choice_lengths[i].item()].append(0)
                    else:
                        p = self.player_to_accuracy[50]
                        p[choice_lengths[i].item()].append(0)
            self.train_pick_choices.extend([choice_lengths[i].item() for i in range(choice_lengths.size(0))])
            return distances
        else:
            anchor,positive,negative,lens = batch
            anchor_out,positive_out,negative_out = self(anchor,positive,negative)
            loss = self.loss_fn(anchor_out,positive_out,negative_out)
            self.test_losses.extend([l.item() for l in loss])
            return None
            

    def training_epoch_end(self, epoch):
        train_loss = np.mean(self.train_losses)
        self.writer.add_scalar('epoch_train_loss',train_loss,epoch)
        self.train_losses.clear()
        return np.mean(train_loss)

    def test_epoch_end(self, outs = None):
        if self.test_picks:
            accuracy = np.mean(self.pick_accuracy)
            self.pick_accuracy.clear()

            top_k_accuracy = {k:np.mean(v) for k,v in self.top_k_accuracy.items()}
            self.top_k_accuracy.clear()

            distance = np.mean(self.pick_distance)
            with open('siamese_pick_distance.txt', 'w') as f:
                for d in self.pick_distance:
                    f.write(f'{d}\n')
            self.pick_distance.clear()

            test_choices = np.mean(self.train_pick_choices)
            self.train_pick_choices.clear()

            player_to_num_accuracy = self.player_to_accuracy.items()
            sorted_dict = {}
            for player,acc_dict in player_to_num_accuracy:
                num_tuples = acc_dict.items()
                num_tuples = sorted(num_tuples, key = lambda x: np.mean(x[1]))
                all_results = []
                for tup in num_tuples:
                    all_results.extend(tup[1])
                total_acc = np.mean(all_results)
                num_tuples = [(tup[0],np.mean(tup[1]),len(tup[1])) for tup in num_tuples]
                num_tuples.append(f'Total accuracy = {total_acc}')
                if player == 50:
                    sorted_dict['No player'] = num_tuples
                else:
                    sorted_dict[self.reverse_encoding[player]] = num_tuples
                
            self.player_to_accuracy = defaultdict(lambda: defaultdict(list))
            self.epoch_accuracy = accuracy


            print(f'Pick accuracy: {self.epoch_accuracy}')
            print(f'Top k accuracy: {top_k_accuracy}')
            print(f'Pick distance: {distance}')
            print(f'Pick choices: {test_choices}')
            print(f'Accuracy by player and choice number:')
            for k,v in sorted_dict.items():
                print(f'{k}:{v}')
        else:
            print(self.test_losses[:100])
            print(np.mean(self.test_losses))
            self.log('test_loss',np.mean(self.test_losses), sync_dist= True)
            self.test_losses.clear()


    def validation_epoch_end(self, epoch):
        epoch_loss =np.mean(self.eval_losses)
        self.writer.add_scalar('epoch_eval_loss',epoch_loss,epoch)
        self.eval_losses.clear()
        return np.mean(epoch_loss)
class Probability_Network(nn.Module):
    def __init__(self, pre_embed_dim = 128, path = None, create_writer = False, device = 'cuda'):
        super(Probability_Network,self).__init__()
        num_players,self.player_encoding,self.empty_encoding = Siamese_RBC_dataset.create_player_encoding('game_numbers.csv')
        self.reverse_encoding = {torch.argmax(v,dim=0)[0,0].item():k for k,v in self.player_encoding.items()}
        self.train_pick_choices = []
        self.visualize = False

        self.main_block_size = 128

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_pick_accuracy = []
        self.pick_accuracy = []
        self.pick_choices = []
        self.train_losses = []
        self.eval_losses = []
        self.test_losses = []
        self.pick_distance = []
        self.train_pick_distance = []
        self.choice_num_to_accuracy = defaultdict(list)
        self.top_k_accuracy = defaultdict(list)
        self.player_to_accuracy = defaultdict(lambda: defaultdict(list))
        self.device = device
        self.writer = SummaryWriter()


        self.input = Embedding_Network_Convolution(input_layers = 1850+12,output_size = pre_embed_dim, num_layers = 5, layer_size = 64)
        self.first_block = Siamese_Block_Convolution(pre_embed_dim,self.main_block_size)
        self.main_network = nn.ModuleList()
        for _ in range(10):
            self.main_network.append(Siamese_Block_Convolution(self.main_block_size,self.main_block_size))
        self.last_block = nn.Sequential(
            nn.Conv2d(self.main_block_size,64, kernel_size=1, padding = 'same'),
            nn.InstanceNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64,1, kernel_size=1, padding = 'same'),
            nn.InstanceNorm2d(1),
            nn.ELU()
        )
        self.output = nn.Linear(8*8*1,1)
        if path is not None:
            self.load_state_dict(torch.load(path, map_location = self.device))
            self.eval()
        self = self.to(self.device)

    def get_distances_of_options(self,full_anchor,options):
        self.eval()
        anchor,player,padding = full_anchor
        B,num_options,_,_,_ = options.shape
        with torch.no_grad():
            anchor = torch.cat((anchor.to(self.device),player.to(self.device)), dim = 0)
            anchor = anchor.repeat_interleave(num_options, dim = 0)
            inputs = torch.cat((anchor,options), dim = 0)
            out = self.forward(inputs)
        self.train()
        return distances
    
    def forward(self,anchor,board,player):
        input = torch.cat((anchor.to(self.device),player.to(self.device), board.to(self.device)), dim = 1)
        input = self.input(input)
        input = self.first_block(input)
        for block in self.main_network:
            input = input + block(input)
        output = self.last_block(input)
        output = output.view(-1,8*8)
        output = self.output(output)
        return output
    
    def training_step(self,batch,batch_idx):
        anchor,positive,negative,player = batch
        anchor = anchor.view(-1,1800,8,8)
        targets = torch.cat((torch.ones(anchor.size(0)),torch.zeros(anchor.size(0))),dim = 0).to(self.device)
        anchor = anchor.repeat(2,1,1,1)
        choices = torch.cat((positive,negative),dim = 0)
        player = player.repeat(2,1,1,1)

        self.optimizer.zero_grad()
        out = self(anchor,choices,player).squeeze()
        loss = self.loss_fn(out, targets)
        loss = torch.mean(loss)
        loss.backward()
        self.optimizer.step()
        return loss

    def validation_step(self,batch,batch_idx):
        anchor,positive,negative,player = batch
        anchor = anchor.view(-1,1800,8,8)
        targets = torch.cat((torch.ones(anchor.size(0)),torch.zeros(anchor.size(0))),dim = 0).to(self.device)
        anchor = anchor.repeat(2,1,1,1)
        choices = torch.cat((positive,negative),dim = 0)
        player = player.repeat(2,1,1,1)

        with torch.no_grad():
            out = self(anchor,choices,player).squeeze()
            loss = self.loss_fn(out, targets)
            loss = torch.mean(loss)
        return loss

    def test_step(self,batch,batch_idx, top_k = (1,), top_k_percentage = None):
        anchor,positive,negative,player,lens = batch
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        player = player.to(self.device)
        maximum_length = torch.max(lens).item()+1

        choice_list = torch.cat((positive.unsqueeze(dim=0),negative),dim = 1)

        random_indizes = list(range(int(maximum_length)))
        shuffle(random_indizes)
        try:
            correct_choice = np.argmin(random_indizes)
        except Exception as e:
            raise e
        choice_list = choice_list.view(maximum_length,12,8,8)[random_indizes,:,:,:]
        batch_size = anchor.size(0)

    
        anchor = anchor.repeat_interleave(maximum_length, dim = 0)
        player = player.repeat_interleave(maximum_length, dim = 0)
        anchor = anchor.view(maximum_length,1800,8,8)
        distances = torch.sigmoid(self.forward(anchor, choice_list, player))

        for i in range(batch_size):
            if lens[i] < maximum_length:
                distances[i,lens[i]:] = float('-inf')
        distances = distances.view(batch_size,maximum_length)
        picks = torch.argmax(distances,dim=1)
        pick_distances = torch.argsort(distances,descending = True,dim= 1)
        index_of_correct_choice = (pick_distances == correct_choice).nonzero(as_tuple=True)[1]
        self.pick_distance.extend(list(index_of_correct_choice.cpu().numpy()))
        self.pick_accuracy.extend([1 if pick == correct_choice else 0 for pick in picks])
        player = player[0,:,:,:].unsqueeze(dim=0)
        for i in range(picks.shape[0]):
            for k in top_k:
                if index_of_correct_choice[i] < k:
                    self.top_k_accuracy[k].append(1)
                else:
                    self.top_k_accuracy[k].append(0)
            
            for k in top_k_percentage:
                if index_of_correct_choice[i] <= math.ceil(lens[i].item()*k):
                    self.top_k_accuracy[k].append(1)
                else:
                    self.top_k_accuracy[k].append(0)
            if picks[i].item() == correct_choice:
                self.choice_num_to_accuracy[lens[i].item()].append(1)
                if torch.sum(player).item() == 64:
                    p = self.player_to_accuracy[torch.argmax(player,dim = 1)[0,0,0].item()]

                    p[lens[i].item()].append(1)
                else:
                    p = self.player_to_accuracy[50]
                    p[lens[i].item()].append(1)

            else:
                self.choice_num_to_accuracy[lens[i].item()].append(0)
                if torch.sum(player).item() == 64:
                    p = self.player_to_accuracy[torch.argmax(player,dim = 1)[0,0,0].item()]
                    p[lens[i].item()].append(0)
                else:
                    p = self.player_to_accuracy[50]
                    p[lens[i].item()].append(0)
        self.train_pick_choices.extend([lens[i].item() for i in range(lens.size(0))])
        return distances
            

    def training_epoch_end(self, epoch):
        train_loss = np.mean(self.train_losses)
        self.writer.add_scalar('epoch_train_loss',train_loss,epoch)
        self.train_losses.clear()
        return np.mean(train_loss)

    def test_epoch_end(self, outs = None):
        accuracy = np.mean(self.pick_accuracy)
        self.pick_accuracy.clear()

        top_k_accuracy = {k:np.mean(v) for k,v in self.top_k_accuracy.items()}
        self.top_k_accuracy.clear()
        distance = np.mean(self.pick_distance)
        with open('siamese_pick_distance.txt', 'w') as f:
            for d in self.pick_distance:
                f.write(f'{d}\n')
        self.pick_distance.clear()

        test_choices = np.mean(self.train_pick_choices)
        self.train_pick_choices.clear()

        player_to_num_accuracy = self.player_to_accuracy.items()
        sorted_dict = {}
        for player,acc_dict in player_to_num_accuracy:
            num_tuples = acc_dict.items()
            num_tuples = sorted(num_tuples, key = lambda x: np.mean(x[1]))
            all_results = []
            for tup in num_tuples:
                all_results.extend(tup[1])
            total_acc = np.mean(all_results)
            num_tuples = [(tup[0],np.mean(tup[1]),len(tup[1])) for tup in num_tuples]
            num_tuples.append(f'Total accuracy = {total_acc}')
            if player == 50:
                sorted_dict['No player'] = num_tuples
            else:
                sorted_dict[self.reverse_encoding[player]] = num_tuples
            
        self.player_to_accuracy = defaultdict(lambda: defaultdict(list))
        self.epoch_accuracy = accuracy


        print(f'Pick accuracy: {self.epoch_accuracy}')
        print(f'Top k accuracy: {top_k_accuracy}')
        print(f'Pick distance: {distance}')
        print(f'Pick choices: {test_choices}')
        print(f'Accuracy by player and choice number:')
        for k,v in sorted_dict.items():
            print(f'{k}:{v}')


    def validation_epoch_end(self, epoch):
        epoch_loss =np.mean(self.eval_losses)
        self.writer.add_scalar('epoch_eval_loss',epoch_loss,epoch)
        self.eval_losses.clear()
        return np.mean(epoch_loss)

def get_distance(positive,negative):
    return torch.sum(torch.pow(positive-negative,2),dim=1)