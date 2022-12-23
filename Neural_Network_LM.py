from torch import nn
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

#getting input from input.txt and tokenizing
string = open('info.txt').read()
new_str = re.sub(r'[^\w\s]', ' ', string)
list1 = [new_str]
words = list(map(str.split, list1))
words1 = set(words[0])
cnt = len(words1)


# create the dict annd tokenizing it from glove embedding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(words1)
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

#defining our model
class Models(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(200, 300)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(300, 300)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(300, cnt)
        self.LogSoftmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output(x)
        x = self.LogSoftmax(x)
        return x

## extracting data
v_input_data = []
v_output_data = []
for i in range(30000):
    input = []
    print(f' Completed getting data: {i/300}%')
    output_data.append(Dict[words[0][i+4]])
    v_output_data.append(Dict[words[0][i+30004]])
    arr = []
    arr1 = []
    if words[0][i] in dict.keys():
        for j in range(50):
            arr.append(embedding_matrix_vocab[dict[words[0][i]]][j])
           
            
    else:
        for j in range(50):
            arr.append(special_dict[words[0][i]][j])
           
    if words[0][i+1] in dict.keys():
        for j in range(50):
            arr.append(embedding_matrix_vocab[dict[words[0][i+1]]][j])
           
    else:
        for j in range(50):
            arr.append(special_dict[words[0][i+1]][j])
          
    if words[0][i+2] in dict.keys():
        for j in range(50):
            arr.append(embedding_matrix_vocab[dict[words[0][i+2]]][j])
         
    else:
        for j in range(50):
            arr.append(special_dict[words[0][i+2]][j])
          
    if words[0][i+3] in dict.keys():
        for j in range(50):
            arr.append(embedding_matrix_vocab[dict[words[0][i+3]]][j])
            
    else:
        for j in range(50):
            arr.append(special_dict[words[0][i+3]][j])
           
    input_data.append(arr)


input_data = np.array(input_data)




#Batching data
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
features,labels = data


#model 
num_epoch = 5
model = Models()
loss = nn.NLLLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(num_epoch):
    lossing= 0
    for i,(input,labels) in enumerate(dataloader):
        y_pred = model(input.float())
        #print(y_pred)
        #print(labels)
        print(f'Batches done: {i/0.3}%')
        l = loss(y_pred,labels)
        lossing += l
        l.backward()
        optimiser.step()
        optimiser.zero_grad()
    print(f'epoch {epoch+1}: loss = {lossing}')
    
time.sleep(2)

torch.save(model.state_dict(),"model.pth")


#load saved model
loaded_model = Models()
loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()



## Get sentences for getting perplexity scores
with open('info.txt') as f:
    lines = [line.rstrip() for line in f]    
punc = '''!()-[]{};:"\,<>./?@#$%^&*_~'''

for i in range(len(lines)):
    lines[i] = re.sub(r'[^\w\s]', ' ', lines[i])
res = list(map(str.split, lines))


#Calculating perplexity scores
for i in range(len(res)):
    ans = 0
    if len(res[i]) == 0:
        continue
    for j in range(len(res[i])-4):
        input = []
        if res[i][j] in dict.keys():
            for k in range(50):
                input.append(embedding_matrix_vocab[dict[res[i][j]]][k])
        else:
            for k in range(50):
                input.append(special_dict[res[i][j]][k])
        if res[i][j+1] in dict.keys():
            for k in range(50):
                input.append(embedding_matrix_vocab[dict[res[i][j+1]]][k])
        else:
            for k in range(50):
                input.append(special_dict[res[i][j+1]][k])
        if res[i][j+2] in dict.keys():
            for k in range(50):
                input.append(embedding_matrix_vocab[dict[res[i][j+2]]][k])
        else:
            for k in range(50):
                input.append(special_dict[res[i][j+2]][k])
                
        if res[i][j+3] in dict.keys():
            for k in range(50):
                input.append(embedding_matrix_vocab[dict[res[i][j+3]]][k])
        else:
            for k in range(50):
                input.append(special_dict[res[i][j+3]][k])
        input = np.array(input)
        input = torch.from_numpy(input)
        output = loaded_model(input.float())
        ans +=  math.log2(math.exp(output[Dict[res[i][j+4]]]))
    p = len(res[i])
    #print(p)
    x=ans/p
    x*= -1
    print(f'Perplexity Score for the sentence "{res[i]}" = {math.pow(2,x)}')
