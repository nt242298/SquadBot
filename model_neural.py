import torch.nn as nn
import numpy as np
from nltk.stem.porter import PorterStemmer


def bag_of_words(tokenized_sentence, words):  # returns bag of words array
    stemmer = PorterStemmer()
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in tokenized_sentence]
    # create a bag with 0's for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
 
    return bag


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
