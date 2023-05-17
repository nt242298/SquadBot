import random
import json
import torch
import nltk
from model_neural import bag_of_words, NeuralNet

with open('intents.json', 'r') as file:
    intents = json.load(file)

file_path = "data.pth"
data = torch.load(file_path)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size)
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


def get_response(msg):
    sentence = nltk.word_tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    ERR_THRESHOLD = 0.8
    if prob.item() > ERR_THRESHOLD:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "Sorry, I don't understand. Please check your input and try again."


if __name__ == "__main__":
    print("Welcome to our chatbot! Type in your queries or type '0' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "0":
            break

        resp = get_response(sentence)
        print(f"Bot: {resp}")

