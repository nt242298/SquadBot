import numpy as np
import json
import nltk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.stem.porter import PorterStemmer
from model_neural import bag_of_words, NeuralNet

with open('intents.json', 'r') as file:
    intents = json.load(file)


all_words = []
tags = []
xy = []
# loop through each sentence in the intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenizes each sentence (pattern), splits it into words
        w = nltk.word_tokenize(pattern)
        # add to all_words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))


stemmer = PorterStemmer()
ignore_symbols = ['?', '.', '!', ',']
# stem and lower each word
all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_symbols]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: tags
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# hyper-parameters
num_epochs = 500
batch_size = 10
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 10
output_size = len(tags)
print(input_size, output_size)
#print(*X_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # define indexing so that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # support calling len(dataset) to return the size
    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        # forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

file_path = "data.pth"
torch.save(data, file_path)

print(f'Training is complete. File saved to {file_path}')
