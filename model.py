#AI
import torch
import torch.nn as nn
import numpy as np
#Data
import nltk
import spacy
#Utilities
from pathlib import Path
import datetime
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
#rm -rf logs
#!mkdir logs
#%tensorboard --logdir ./logs
#Fixed(not used for validation)
m = 30
n = 5
hidden_dim = 128

#Hyperparameters
batch_size = 128
lr = .001
weight_decay = 0
momentum = .9

experiment_name = Path('act-{}_emb-{}_sample-{}_batch-{}_optim-{}_lr-{}_mom-{}'.format(
    'ReLU',
    m,
    n,
    batch_size,
    'sgd',
    lr,
    momentum
))
experiment_path = Path('./logs') / experiment_name
writer = SummaryWriter(experiment_path)
seed = 814
nltk.download('brown')
#Set device in case for GPU
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class DataSet:
    def __init__(self,data, training = .7, val = .15, test = .15):
        self.data = data
        self.split_data(training, val, test)
        self.vocab, self.vocab_eff = DataSet.gen_vocab(data)
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
    def gen_vocab(data):
        """
        """
        data = [word for sentence in nltk.corpus.brown.sents() for word in sentence]
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
        #Split input and outputs
        xs = [i[0] for i in batch]
        ys = [i[1] for i in batch]
        #Convert and create batch
        x_tens = np.stack(xs, axis=0)
        y_tens = np.stack(ys, axis=0)
        batch = (torch.from_numpy(x_tens), torch.from_numpy(y_tens))
        return batch
    def pregen_batch(self, n, batch_size, split = 'training'):
        data = self.data_splitted[split]
        data_index = [self.vocab_eff[word] for word in data]
        samples = [(data_index[start_index: start_index+n-1], [data_index[start_index+n-1]]) for start_index in range(len(data)-n)]
        #Shuffle all n-grams
        random.shuffle(samples) 
        batches = [samples[start_index:start_index+batch_size] for start_index in range(0,len(samples), batch_size)]
        #Make sure final batch is of correct size
        if len(batches[-1])<batch_size:
            batches[-1] += samples[0: batch_size-len(batches[-1])]
        final_batches = [list(zip(*batch)) for batch in batches]
        final_batches = [(torch.LongTensor(batch[0]), torch.LongTensor(batch[1])) for batch in tqdm(final_batches)]
        return final_batches
class Model(nn.Module):

    def __init__(self, data, m, n):
        """
        """
        super(Model, self).__init__()
        n = n-1
        self.embedding = nn.Embedding(len(data.vocab), m)
        self.linear = nn.Linear(n*m, 128, bias = True)
        self.relu = nn.ReLU()
        self.log_sm = nn.LogSoftmax(dim= -1)
        self.linearF = nn.Linear(128, len(data.vocab), bias= True)

    def forward(self, inputs, batch_size):
        """
        """
        #print(inputs)
        x = self.embedding(inputs).view(batch_size,-1)
        #print(x.size())
        x = self.linear(x)
        x = self.relu(x)
        x = self.linearF(x)
        x = self.log_sm(x)
        #print(yhat.size())
        return x


def train(data,n = 5, m = 30, weight_decay = 0, batch_size = 128, lr = .001, epochs = 600, momentum = 0.9, preloaded=False, checkpointed = False, first_time = True):
    """
    """
    net = Model(data, m, n).to(cuda)
    optimizer = torch.optim.SGD(net.parameters(), lr= lr, momentum = momentum, weight_decay = weight_decay)
    #optimizer = torch.optim.Adam(net.parameters(), lr= lr, amsgrad = True)
    epoch_num = 0
    loss = 0
    overfitter = 0
    batches= []
    val_perp = []
    if checkpointed:
        if not first_time:
            checkpoint = torch.load('./ngram_training.pt', map_location=cuda)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_num = checkpoint['epoch']
            loss = checkpoint['loss']
    #for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor])
    if preloaded:
        print('Loading batches...')
        batches = data.pregen_batch(n, batch_size)
    net.train()
    criterion = torch.nn.NLLLoss()
    data_count = len(data.data_splitted['training'])
    for epoch in range(epoch_num,epochs+epoch_num):
        losses = []
        #Check testing loss to prevent overfitting
        if epoch%10==1:
            print('Loading batches')
            batches = data.pregen_batch(n, batch_size, split='validation')
            net.eval()
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
                x,y = batch
                x = x.to(cuda)
                y = y.squeeze(1).to(cuda)
                outputs = net(x, batch_size).to(cuda)
                optimizer.zero_grad()
                loss = criterion(outputs, y).to(cuda)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
        else:
            #Dynamic batch creation
            print('Training epoch...')
            for i in tqdm(range(int(data_count/batch_size)-n)):  
                #Get data              
                x, y = data.gen_batch(n, batch_size)
                x = x.to(cuda)
                y = y.squeeze(1).to(cuda)
                #Forward pass and loss
                outputs = net(x, batch_size).to(cuda)
                optimizer.zero_grad()
                loss = criterion(outputs, y).to(cuda)
                losses.append(loss.item())
                #Backprop and weights update
                loss.backward()
                optimizer.step()
        print(datetime.datetime.now(), '1 epoch')
        #Early stopping
        if loss<.001:
            break
        #Calculate perplexity
        avg_loss = sum(losses)/len(losses)
        perplexity = np.exp(avg_loss)
        #TensorBoard stuff
        writer.add_scalar('Training perplexity', perplexity, epoch)
        print('Epoch: {} Perp:{}'.format(epoch, perplexity))
        #Early stopping
        if len(val_perp)>1:
            if val_perp[-1]<val_perplexity:
                    break
    #Saving checkpoint
    if checkpointed:
        checkpoint = {'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}
        torch.save(checkpoint, './ngram_training.pt')
    return net



def main():
    print(datetime.datetime.now(), 'initial time')
    data = DataSet(nltk.corpus.brown.sents())
    net_trained = train(data, n = n, m = m, weight_decay = weight_decay, batch_size = batch_size, lr = lr, 
                            epochs = 5, momentum = momentum, preloaded=True, checkpointed = True, first_time = False)
    torch.save(net_trained.state_dict(), './NGRAM.pt')
    print(datetime.datetime.now(), 'after training')

if __name__ == '__main__':
    main()
