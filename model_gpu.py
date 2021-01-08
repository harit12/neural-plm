import torch
import torch.nn as nn
import nltk
import numpy as np
import spacy
import datetime
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
        random = np.random.randint(0, len(data)-n)
        sample = (data[random:random+n-1], data[random+n])
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

def train(data,n = 5, m = 30, batch_size = 128, lr = .01, epochs = 100, momentum = 0.7):
    """
    """
    net = Model(data, m, n)
    net = net.to(cuda)
    optimizer = torch.optim.SGD(net.parameters(), lr= lr, momentum = momentum)
    criterion = torch.nn.NLLLoss()
    data_count = len(data.data_splitted['training'])
    print(int(data_count/batch_size))
    for epoch in range(epochs):
        for i in range(int(data_count/batch_size)-n):
            x, y = data.gen_batch(n, batch_size)
            x = x.to(cuda)
            y = y.squeeze(1).to(cuda)
            #print(datetime.datetime.now(), 'datagen time')
            #print(x.size())
            outputs = net(x, batch_size).to(cuda)
            #print(datetime.datetime.now(), 'forward time')
            optimizer.zero_grad()
            loss = criterion(outputs, y).to(cuda)
            loss.backward()
            #print(datetime.datetime.now(), 'backprop time')
            optimizer.step()
            #print(datetime.datetime.now(), 'grad_update time')
            #print(i)
            #if(i%100==1):
             # print(i, datetime.datetime.now(), '100 iters\nLoss:{}'.format(loss))
        print(datetime.datetime.now(), '1 epoch')
        if loss<.01:
            break
        print('Epoch: {}\nLoss:{}'.format(epoch, loss))
    return net
def main():
    print(datetime.datetime.now(), 'initial time')
    data = DataSet(nltk.corpus.brown.sents())
    net_trained = train(data)
    print(datetime.datetime.now(), 'after training')
if __name__ == '__main__':
    main()
