import torch
import torch.nn as nn
import nltk
import numpy as np
import spacy
import datetime
import random
seed = 814
nltk.download('brown')
cuda = torch.device('cuda')
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
        #res = list(map(self.w2idx, sample))
        #print(datetime.datetime.now(), 'transform')
        return res
    def idx2sample(self, idx):
        """
        """
        return [self.idx2w(ind) for ind in idx]

    def gen_sample(self,n,split='training'):
        """
        """
        data = self.data_splitted[split]
        random_int = random.randint(0, len(data)-n-1)
        #print(random_int)
        sample = (data[random_int:random_int+n-1], data[random_int+n-1])
        #print(sample)
        #print(datetime.datetime.now(), 'create sample')
        sample = (np.array(self.sample2idx(sample[0])), np.array([self.w2idx(sample[1])]))
        #print(datetime.datetime.now(), 'create tensor')
        return sample

    def gen_batch(self, n,batch_size, split = 'training'):
        """
        """
        data = self.data_splitted[split]
        batch = [self.gen_sample(n) for x in range(batch_size)]
        #print(datetime.datetime.now(), 'create batch')
        xs = [i[0] for i in batch]
        ys = [i[1] for i in batch]
        #print(datetime.datetime.now(), 'list time')
        x_tens = np.stack(xs, axis=0)
        #print(x_tens)
        y_tens = np.stack(ys, axis=0)
        #print(datetime.datetime.now(), 'stack time')
        batch = (torch.from_numpy(x_tens), torch.from_numpy(y_tens))
        #print(datetime.datetime.now(), 'conv time')
        return batch

class Model(nn.Module):

    def __init__(self, data, m, n):
        """
        """
        super(Model, self).__init__()
        n = n-1
        self.embedding = nn.Embedding(len(data.vocab), m)
        self.linear = nn.Linear(n*m, 128, bias = True)
        self.tanh = nn.Tanh()
        self.log_sm = nn.LogSoftmax(dim= 1)
        self.linearF = nn.Linear(128, len(data.vocab), bias= True)

    def forward(self, inputs, batch_size):
        """
        """
        #print(inputs)
        x = self.embedding(inputs).view(batch_size,-1)
        #print(x.size())
        x = self.linear(x)
        x = self.tanh(x)
        x = self.linearF(x)
        x = self.log_sm(x)
        #print(yhat.size())
        return x

def train(data,n = 5, m = 30, batch_size = 128, lr = .01, epochs = 200, momentum = 0.9, preloaded=False, checkpointed = False, first_time = True):
    """
    """
    net = Model(data, m, n).to(cuda)
    optimizer = torch.optim.SGD(net.parameters(), lr= lr, momentum = momentum)
    epoch_num = 0
    loss = 0
    if checkpointed:
        if not first_time:
            print('loading')
            checkpoint = torch.load('/kaggle/input/trainingcheckpoint1/ngram_training.pth.tar')
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_num = checkpoint['epoch']
            loss = checkpoint['loss']
    net.train()
    criterion = torch.nn.NLLLoss()
    data_count = len(data.data_splitted['training'])
    for epoch in range(epoch_num,epochs+epoch_num):
        random.seed(seed+epoch)
        for i in range(int(data_count/batch_size)-n):
            x, y = data.gen_batch(n, batch_size)
            x = x.to(cuda)
            y = y.squeeze(1).to(cuda)
            outputs = net(x, batch_size).to(cuda)
            optimizer.zero_grad()
            loss = criterion(outputs, y).to(cuda)
            loss.backward()
            optimizer.step()
            #if(i%100==1):
             #print(i, datetime.datetime.now(), '100 iters\nLoss:{}'.format(loss))
        print(datetime.datetime.now(), '1 epoch')
        if loss<.001:
            break
        print('Epoch: {} Loss:{}'.format(epoch, loss))
    if checkpointed:
        checkpoint = {'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}
        torch.save(checkpoint, './ngram_training.pth.tar')
    return net

def validation(data,n = 100, m = 30, batch_size = 128, lr = .01,  momentum = 0.9):
    pass
def main():
    print(datetime.datetime.now(), 'initial time')
    data = DataSet(nltk.corpus.brown.sents())
    net_trained = train(data, checkpointed=True, first_time= False)
    torch.save(net_trained.state_dict(), './NGRAM.pth.tar')
    print(datetime.datetime.now(), 'after training')
if __name__ == '__main__':
    main()
