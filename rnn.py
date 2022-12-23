from torch import nn
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
import torch
import time
import random
import math
import string
input_data = []
string = open('info.txt').read()
new_str = re.sub(r'[^\w\s]', ' ', string)
list1 = [new_str]
words = list(map(str.split, list1))
#print(words[0][10])

words1 = set(words[0])
cnt = len(words1)
def tanh(x):
    return 1 / (1 + np.tanh(x))

# create the dict.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(words1)

# number of unique words in dict.
# print("Number of unique words in dictionary=",
#       len(tokenizer.word_index))
# print("Dictionary is = ", tokenizer.word_index)
input_data = []
output_data = []
Dict = {}
smallDict = {}
cnt = 0
for i in range(len(words[0])):
    if words[0][i] in Dict.keys():
        continue
    else:
        Dict[words[0][i]] = cnt
        cnt+=1

def embedding_for_vocab(filepath, word_index,
                        embedding_dim):
    vocab_size = len(word_index) + 1

    # Adding again 1 because of reserved 0 index
    embedding_matrix_vocab = np.zeros((vocab_size,
                                       embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix_vocab[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix_vocab


dict = tokenizer.word_index

input_data = []

# matrix for vocab: word_index
embedding_dim = 50
embedding_matrix_vocab = embedding_for_vocab(
    './glove.6B/glove.6B.50d.txt', tokenizer.word_index,
  embedding_dim)


special_dict = {}
for i in range(len(words[0])):
    
    if words[0][i] in dict.keys():
        continue
    else:
        special_dict[words[0][i]] = np.random.random(50)


input_size = 50
hidden_size = 300
output_size = cnt
num_layers = 2

U = np.random.uniform(0,1,(300,50))
V = np.random.uniform(0,1,(output_size,300))
W = np.random.uniform(0,1,(hidden_size,hidden_size))

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden = nn.Linear(hidden_size,hidden_size)
        self.softmax = nn.LogSoftmax(dim=0)
        self.Tanh = nn.Tanh()
    def forward(self,input_tensor,hidden_tensor):
        # Pass the input tensor through each of our operations
        input_tensor =np.transpose(input_tensor)
        hidden_tensor = np.transpose(hidden_tensor)
        
        Uxi = torch.from_numpy(np.dot(U,input_tensor))
        Wht = torch.from_numpy(np.dot(W,hidden_tensor))
        Uxi = Uxi.float()
        Wht = Wht.float()
        add = torch.add(Uxi,Wht)
        hidden = self.hidden(add)
        hidden = torch.tensor(hidden)
        hidden = torch.tanh(hidden)
        output = np.dot(V,add)
        output = torch.tensor(output)
        output = self.softmax(output)
        return output,hidden

    def init_hidden(self):
        return torch.zeros(self.hidden_size)


for i in range(30000):
    input = []
    #print(i)
    # x = torch.zeros(cnt,dtype = torch.float32)
    # x[Dict[words[0][i+4]]] = 1
    p = torch.zeros(cnt)
    p[Dict[words[0][i+1]]] = 1
    output_data.append(p)
    arr = []
    if words[0][i] in dict.keys():
        for j in range(50):
            arr.append(embedding_matrix_vocab[dict[words[0][i]]][j])
    else:
        for j in range(50):
            arr.append(special_dict[words[0][i]][j])     
    input_data.append(arr)

input_data = np.array(input_data)
#output_data = np.array(output_data)
class wordsDataset(Dataset):
    
    def __init__(self):
        self.x = torch.from_numpy(input_data)
        self.y = output_data
        self.n_samples = 30000
        
    def __getitem__(self,index):
        return self.x[index],self.y[index]    
        
    def __len__(self):
        return self.n_samples 



dataset = wordsDataset()

dataloader = DataLoader(dataset=dataset,batch_size = 1000)

dataiter = iter(dataloader)

data = dataiter.next()
    
        
num_epoch = 5

rnn = RNN(input_size,hidden_size,output_size)
loss = nn.NLLLoss()

optimiser = torch.optim.SGD(rnn.parameters(), lr=0.1)
def train():
    
    hidden = rnn.init_hidden()
    losss = 0
    for i in range(30000):
        output ,hidden = rnn(input_data[i],hidden)
        l = loss(output_data[i].float,output)
        losss += l
        l.backward()
        optimiser.step()
        optimiser.zero_grad()
    return losss    


for epoch in range(5):
    lossing= 0
    lossing += train()
    print(f'epoch {epoch+1}: loss = {lossing}')
    time.sleep(2)

# torch.save(model.state_dict(),"model1.pth")
# with open('info.txt') as f:
#     lines = [line.rstrip() for line in f]
    
    
# punc = '''!()-[]{};:"\,<>./?@#$%^&*_~'''
# loaded_model = Models()
# loaded_model.load_state_dict(torch.load('model.pth'))
# loaded_model.eval()


# for i in range(len(lines)):
#     lines[i] = re.sub(r'[^\w\s]', ' ', lines[i])

# res = list(map(str.split, lines))
# for i in range(len(res)):
#     ans = 0
#     if len(res[i]) == 0:
#         continue
#     for j in range(len(res[i])-4):
#         input = []
#         if res[i][j] in dict.keys():
#             for k in range(50):
#                 input.append(embedding_matrix_vocab[dict[res[i][j]]][k])
#         else:
#             for k in range(50):
#                 input.append(special_dict[res[i][j]][k])
#         if res[i][j+1] in dict.keys():
#             for k in range(50):
#                 input.append(embedding_matrix_vocab[dict[res[i][j+1]]][k])
#         else:
#             for k in range(50):
#                 input.append(special_dict[res[i][j+1]][k])
#         if res[i][j+2] in dict.keys():
#             for k in range(50):
#                 input.append(embedding_matrix_vocab[dict[res[i][j+2]]][k])
#         else:
#             for k in range(50):
#                 input.append(special_dict[res[i][j+2]][k])
                
#         if res[i][j+3] in dict.keys():
#             for k in range(50):
#                 input.append(embedding_matrix_vocab[dict[res[i][j+3]]][k])
#         else:
#             for k in range(50):
#                 input.append(special_dict[res[i][j+3]][k])
#         input = np.array(input)
#         input = torch.from_numpy(input)
#         output = loaded_model(input.float())
#         ans +=  math.log2(math.exp(output[Dict[res[i][j+4]]]))
#     p = len(res[i])
#     print(p)
#     x=ans/p
#     x*= -1
#     print(f'Perplexity Score for the sentence "{res[i]}" = {math.pow(2,x)}')
