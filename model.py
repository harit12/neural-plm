##################################################################################
# WARNING: In order to use loading model, you must use the same DataSet instance #
##################################################################################
import torch
import torch.nn as nn
import nltk
from pathlib import Path
import numpy as np
import spacy
import datetime
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import pickle
import argparse

#TPU Support?
#import torch_xla
#import torch_xla.core.xla_model as xm

#Model architecture
m = 30
n = 5
hidden_dim = 100
dropout = True
relu = True
SGD = True
batch_norm = True
#Hyper parameters
batch_size = 128
lr = .01
weight_decay = 0.0001
momentum = .95
p_inputs = .8
p_hidden = .6

#If using parsing
parser = argparse.ArgumentParser()
#parser.add_argument('m', default = 30,required=False)
#parser.add_argument('n', default =5,required=False)
#parser.add_argument('relu', default = False, action = 'store_true', required=False)
#parser.add_argument('dropout', default = False, action = 'store_true', required=False)
#parser.add_argument('SGD', default = False, action = 'store_true', required=False)
#parser.add_argument('batch_norm', default = False, action = 'store_true', required=False)
#parser.add_argument('lr', default = .01,required=False)
#parser.add_argument('weight_decay', default = .0001,required=False)
#parser.add_argument('momentum', default = .95,required=False)
#parser.add_argument('batch_size', default = 128,required=False)
#parser.add_argument('p_inputs', default = .8,required=False)
#parser.add_argument('p_hidden', default = .6,required=False)

#Loss func
criterion = torch.nn.NLLLoss()
#args = parser.parse_args()

#Model architecture
#m = args.m
#n = args.n
#relu = args.relu
#dropout = args.dropout
#SGD = args.SGD
#batch_norm = args.barch_norm

#Hyper Parameters
#lr = args.lr
#weight_decay = args.weight_decay
#momentum = args.momentum
#p_inputs = args.p_inputs
#p_hidden = args.p_hidden

#TensorBoard logs
experiment_name = Path('act-{}_emb-{}_sample-{}_batch-{}_optim-{}_lr-{}_mom-{}-drop-()-hidden-{}-batch_norm-{}-weight_decay-{}-momentum-{}-p_inputs-{}-p_hidden-{}'.format(
    'ReLU' if relu else 'Tanh',
    m,
    n,
    batch_size,
    'Adam' if not SGD else 'SGD',
    lr,
    momentum,
    dropout,
    hidden_dim,
    batch_norm,
    weight_decay,
    momentum,
    p_inputs,
    p_hidden
))
experiment_path = Path('./logs') / experiment_name
writer = SummaryWriter(experiment_path)

#Random
seed = 814

#Make sure data is avaialble
nltk.download('brown')

#GPU support
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#TPU Support?
#cuda = xm.xla_device()

class DataSet:
    def __init__(self,data, training = .7, val = .15, test = .15):
        self.data = data
        self.split_data(training, val, test)
        self.vocab, self.vocab_eff = self.gen_vocab(data)
    def split_data(self, training, val, test):
        """
        """
        size = len(self.data)
        train_data = self.data[0:int(size*training)]
        val_data = self.data[int(size*training):int((1-test)*size)]
        test_data = self.data[int(size*(1-test)):-1]
        self.data_splitted = {
            'training': [word for sentence in train_data for word in sentence],
            'validation': [word for sentence in val_data for word in sentence],
            'testing': [word for sentence in test_data for word in sentence]
            }
    def gen_vocab(self,data):
        """
        """
        data = [word for sentence in data for word in sentence]
        res = list(set(data))
        res_2 = {word:x for x,word in enumerate(res)}
        return res, res_2
    def w2idx(self, word):
        """
        """
        vocab = self.vocab_eff
        res = vocab[word]
        return res
    def idx2w(self, index):
        """
        """
        vocab = self.vocab
        res = vocab[index]
        return res
    def sample2idx(self, sample):
        """
        """
        res = [self.w2idx(word) for word in sample]
        return res
    def idx2sample(self, idx):
        """
        """
        return [self.idx2w(ind) for ind in idx]

    def gen_sample(self,n,split='training'):
        """
        Dynamic sample generation
        """
        data = self.data_splitted[split]
        random_int = random.randint(0, len(data)-n-1)
        sample = (data[random_int:random_int+n-1], data[random_int+n-1])
        sample = (np.array(self.sample2idx(sample[0])), np.array([self.w2idx(sample[1])]))
        return sample

    def gen_batch(self, n,batch_size, split = 'training'):
        """
        Dynamic batch generation
        """
        data = self.data_splitted[split]
        batch = [self.gen_sample(n) for x in range(batch_size)]
        #print(datetime.datetime.now(), 'create batch')
        xs = [i[0] for i in batch]
        ys = [i[1] for i in batch]
        x_tens = np.stack(xs, axis=0)
        y_tens = np.stack(ys, axis=0)
        batch = (torch.from_numpy(x_tens), torch.from_numpy(y_tens))
        return batch
    
    def pregen_batch(self, n, batch_size, split = 'training'):
        #Preloading batches
        data = self.data_splitted[split]
        data_index = [self.vocab_eff[word] for word in data]
        samples = [(data_index[start_index: start_index+n-1], [data_index[start_index+n-1]]) for start_index in range(len(data)-n)]
        random.shuffle(samples) 
        batches = [samples[start_index:start_index+batch_size] for start_index in range(0,len(samples), batch_size)]
        #Make sure final batch is of correct size
        if len(batches[-1])<batch_size:
            batches[-1] += samples[0: batch_size-len(batches[-1])]
        final_batches = [list(zip(*batch)) for batch in batches]
        final_batches = [(torch.LongTensor(batch[0]), torch.LongTensor(batch[1])) for batch in tqdm(final_batches)]
        return final_batches
class Model(nn.Module):

    def __init__(self, data, m, n, dropout, relu, barch_norm,p_inputs, p_hidden,hidden):
        """
        """
        super(Model, self).__init__()
        self.dropout = dropout
        self.using_relu = relu
        n = n-1
        self.embedding = nn.Embedding(len(data.vocab), m)
        self.batch_norm = nn.BatchNorm1d(n*m)
        self.linear = nn.Linear(n*m, hidden, bias = True)
        self.dropout_input = nn.Dropout(p=p_inputs)
        self.dropout_hidden = nn.Dropout(p=p_hidden)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.log_sm = nn.LogSoftmax(dim= -1)
        self.linearF = nn.Linear(hidden, len(data.vocab), bias= True)

    def forward(self, inputs, batch_size):
        """
        """
        x = inputs
        x = self.embedding(x).view(batch_size,-1)
        x = self.batch_norm(x) if batch_norm else x
        if self.dropout:
            x = self.dropout_input(x)
        x = self.linear(x)
        x = self.relu(x) if self.using_relu else self.tanh(x)
        if self.dropout:
            x = self.dropout_hidden(x)
        x = self.linearF(x)
        x = self.log_sm(x)
        return x


def train(data,n = 5, m = 30, weight_decay = 0, batch_size = 128, lr = .001, epochs = 600, momentum = 0.9, 
          preloaded=False, checkpointed = False, first_time = True, dropout=True, using_relu=False, p_inputs = .8, p_hidden = .5, SGD = True, hidden=128, batch_norm = True):
    """
    """
    net = Model(data, m, n, dropout, using_relu, batch_norm,  p_inputs, p_hidden, hidden).to(cuda)
    optimizer = torch.optim.SGD(net.parameters(), lr= lr, momentum = momentum, weight_decay = weight_decay) if SGD else torch.optim.Adam(net.parameters(), lr= lr, weight_decay = weight_decay)
    epoch_num = 0
    loss = 0
    batches= []
    val_perp = []
    #Loading saved model
    if checkpointed:
        if not first_time:
            checkpointer = torch.load('../input/checkpoint/ngram_training.pt')
            net.load_state_dict(checkpointer['model_state_dict'])
            optimizer.load_state_dict(checkpointer['optimizer_state_dict'])
            epoch_num = checkpointer['epoch']
            loss = checkpointer['loss']
    #for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor])
    if preloaded:
        print('Loading batches...')
        batches = data.pregen_batch(n, batch_size)
    net.train()
    data_count = len(data.data_splitted['training'])
    for epoch in range(epoch_num+1,epochs+epoch_num):
        losses = []
        #Check validation loss to prevent overfitting
        if epoch%10==1:
            print('Loading batches')
            batches = data.pregen_batch(n, batch_size, split='validation')
            net.eval()
            print('VALIDATION:')
            with torch.no_grad():
                for batch in batches:
                    x,y = batch
                    x = x.to(cuda)
                    y = y.squeeze(1).to(cuda)
                    outputs = net(x, batch_size).to(cuda)
                    optimizer.zero_grad()
                    loss = criterion(outputs, y).to(cuda)
                    losses.append(loss.item())
                avg_loss = sum(losses)/len(losses)
                val_perplexity = np.exp(avg_loss)
                print('Epoch: {} Perp:{}'.format(epoch, val_perplexity))
                val_perp.append(val_perplexity)
                writer.add_scalar('Validation perplexity', val_perplexity, epoch)
        losses = []
        net.train()
        #Reshuffle samples in batches every 10 epochs
        if epoch%10==1 and preloaded:
            print('Reshuffling batches')
            batches = data.pregen_batch(n, batch_size)
        random.seed(seed+epoch)
        if preloaded:
            #Shuffle batches
            random.shuffle(batches)
            #Train using backprop
            print('Training epoch...')
            for batch in tqdm(batches):
                #Feature extractions
                x,y = batch
                x = x.to(cuda)
                y = y.squeeze(1).to(cuda)
                #Forward pass
                outputs = net(x, batch_size, p_inputs, p_hidden).to(cuda)
                optimizer.zero_grad()
                #Loss calculation
                loss = criterion(outputs, y).to(cuda)
                losses.append(loss.item())
                #Backprop and weight updates
                loss.backward()
                optimizer.step()
        else:
            #Dynamic batch creation
            print('Training epoch...')
            for i in tqdm(range(int(data_count/batch_size)-n)):
                #Feature extraction
                x, y = data.gen_batch(n, batch_size)
                x = x.to(cuda)
                y = y.squeeze(1).to(cuda)
                #Forward pass
                outputs = net(x, batch_size).to(cuda)
                #Loss calculation and backprop
                optimizer.zero_grad()
                loss = criterion(outputs, y).to(cuda)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
        #Learning rate decay
        if SGD:
            for g in optimizer.param_groups:
                g['lr'] = (.01)/(1+(.0000001*epoch))
            #print(g['lr'])
        #Perplexity Calculations
        avg_loss = sum(losses)/len(losses)
        perplexity = np.exp(avg_loss)
        #TensorBoard stuff
        writer.add_scalar('Training perplexity', perplexity, epoch)
        print('Epoch: {} Perp:{}'.format(epoch, perplexity))
        #Early stopping
        if len(val_perp)>1:
            if val_perp[-2]<val_perp[-1]:
                    break
    
    #Save checkpoint
    if checkpointed:
        checkpoint = {'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}
        torch.save(checkpoint, './ngram_training.pt')
    return net
def test(net, data, n=5, batch_size=128, p_inputs = .5, p_hidden = .8):
    net = net.to(cuda)
    testing_data = data.pregen_batch(n, batch_size, split='testing')
    losses = []
    with torch.no_grad():
        #Ignore batch norm and dropout
        net.eval()
        for batch in testing_data:
            #Feature extraction
            x,y = batch
            x = x.to(cuda)
            y= y.squeeze(1).to(cuda)
            #Forward pass
            yhat = net(x, batch_size, p_inputs, p_hidden).to(cuda)
            loss = criterion(yhat, y)
            losses.append(loss.item())
    #Perplexity calculation
    avg_loss = sum(losses)/len(losses)
    perplexity = np.exp(avg_loss)
    print('Testing perplexity: {}'.format(perplexity))
    return test

def main():
    data = DataSet(nltk.corpus.brown.sents())
    #Pickle error so using torch for saving
    if not os.path.exists('data.obj'):
        torch.save(data, 'data.obj')
    else:
        data = torch.load('data.obj')
    #Train model
    net_trained = train(data, n = n, m = m, weight_decay = weight_decay, batch_size = batch_size, lr = lr, 
                            epochs = 200, momentum = momentum, preloaded=True, checkpointed = True, first_time = True, dropout=dropout, using_relu=relu, p_inputs = p_inputs, p_hidden = p_hidden, SGD = SGD, hidden=hidden_dim)
    #Save trained model
    torch.save(net_trained.state_dict(), './NGRAM.pt')
    #Test model on 1 epoch
    testing_loss = test(net_trained, data)
if __name__ == '__main__':
    main()
