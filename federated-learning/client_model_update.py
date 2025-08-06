import requests
import torch.nn as nn
from copy import deepcopy
import numpy as np
import sys

train_loader = None
test_loader = None

# Regression Model Architecture
num_features = 28*28
num_classes = 10
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        output = self.linear(x)
        return output
    

# History Class from Server
def get_dataset_loss(model, dataset, loss_func):
    # Compute loss of a model on given dataset

    total_loss = 0
    for batch in dataset:
        features, labels = batch
        predictions = model(features)
        total_loss += loss_func(predictions, labels)

    avg_loss = total_loss/len(dataset)
    return avg_loss

def get_accuracy(predictions, labels):

     _, predicted = torch.max(predictions, dim=1)
     accuracy = torch.sum(predicted == labels).item()/len(predicted)
     return accuracy

def get_dataset_accuracy(model, dataset):

    total_accuracy = 0
    for batch in dataset:
        features, labels = batch
        predictions = model(features)
        curr_accuracy = get_accuracy(predictions, labels)
        total_accuracy += curr_accuracy
    
    avg_accuracy = (total_accuracy/len(dataset))*100
    return avg_accuracy


class History:
    def __init__(self):
        
        self.loss = []       # stores model loss
        self.accuracy = []   # stores model accuracy
        self.model = []      # stores model
        self.variance = []

    def log_server(self, model, client, loss_func):
        # Logging loss to history
        curr_loss = [(float)(get_dataset_loss(model, dataset, loss_func).detach()) for dataset in client[0]]
        # self.client_loss.append(curr_loss)
        self.loss.append(sum(curr_loss)/len(curr_loss))

        # Logging accuracy to history
        curr_acc = [get_dataset_accuracy(model, dataset) for dataset in client[1]]
        # self.client_acc.append(curr_acc)

        client_acc_avg = sum(curr_acc)/len(curr_acc)
        self.accuracy.append(client_acc_avg)

        # Logging model to history
        self.model.append(model.state_dict())


        # Logging variance
        sumVar = 0
        for i in range(len(client[1])):
            sumVar += ((curr_acc[i] - client_acc_avg) ** 2)

        self.variance.append(sumVar / (len(client[1]) - 1))

    def log_client(self, model, dataset, loss):
        # Logging loss to history
        self.loss.append(loss)

        # Logging accuracy to history
        curr_acc = get_dataset_accuracy(model, dataset)
        self.accuracy.append(curr_acc)

        # Logging model to history
        self.model.append(model.state_dict())


#Model Updates

def loss_func(predictions, labels):
    loss = nn.CrossEntropyLoss(reduction="mean")
    return loss(predictions, labels)

def train(model, batch):
    images, labels = batch
    predictions = model(images)
    loss = loss_func(predictions, labels)
    return loss

def clientUpdate(client, server_model, history, learning_rate, epochs, mu, type, q=0):
    local_model = deepcopy(server_model)
    local_loss = 0
    optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        epoch_loss = 0

        #Training
        for batch in client:
            loss = train(local_model, batch)

            epoch_loss += loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = epoch_loss / len(client)
        local_loss = epoch_loss

    history.log_client(local_model, client, local_loss)
    return history

def load_model(model_state):
    model = LogisticRegression()
    state_dict = {key: torch.tensor(value) for key, value in model_state.items()}
    model.load_state_dict(state_dict)
    
    return model

#################       #################
################# Q FED #################
#################       #################
def normal(delta_ws):
   
    w = (delta_ws[0]).pow(2).sum()
    b = (delta_ws[1]).pow(2).sum()

    ss = w + b
    return ss

def q_clientUpdate(client, server_model, history, learning_rate, epochs, q):
    # Communicate the latest model
    local_model = deepcopy(server_model)

    # Train server model
    local_loss = 0
    optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
    
    for i in range(epochs):
        epoch_loss = 0
        
        # Training
        for batch in client:
            loss = train(local_model, batch)

            epoch_loss += loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        

        epoch_loss = epoch_loss/len(client)
        local_loss = epoch_loss

    history.log_client(local_model, client, local_loss)

     # Compute loss on the whole training data
    comp_loss = 0
    for batch in client:
        comp_loss += train(local_model, batch)
    
    comp_loss = comp_loss/(len(client))


    # Get difference between the weights
    delta_ws = [(x - y)*(1/learning_rate) for x, y in zip(server_model.state_dict().values(), local_model.state_dict().values())]

    # Calc deltas
    q_loss = comp_loss.detach()
    delta = [np.float_power(q_loss +1e-10, q) * delta_w for delta_w in delta_ws]
  
    # Calc h
    h =  ((q * np.float_power(q_loss+1e-10, (q-1)) * normal(delta_ws)) + ((1/learning_rate) * np.float_power(q_loss+1e-10, q)))

    
    return history, delta, h


#################         #################
################# NETWORK #################
#################         #################


from flask import Flask, request, jsonify
import torch
from torch.utils.data import DataLoader, TensorDataset

app = Flask(__name__)

@app.route("/send_data", methods = ["POST"])
def send_data():
    global train_loader, test_loader

    data = request.get_json()
    features = torch.tensor(data['features'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.long)
    is_train_data = data['is_train_data']
    print(f"Feature length: {len(features)}")
    print(f"Labels length: {len(labels)}")

    client_dataset = TensorDataset(features, labels)

    if is_train_data:
        print("")
        train_loader = DataLoader(client_dataset, batch_size=64, shuffle=True)
        print(f"received train dataset: {len(train_loader)}")
        return {"message" : "Data received successfully."}, 200
    else:
        test_loader = DataLoader(client_dataset, batch_size=64, shuffle=True)
        print(f"received test dataset of length {len(test_loader)}")
        return {"message" : "Data received successfully."}, 200


@app.route("/client_update", methods=["POST"])
def client_update():
    model_data = request.get_json()
    server_model = load_model(model_data["model"])
    learning_rate = model_data["learning_rate"]
    epochs = model_data["epochs"]
    mu = model_data["mu"]
    aggregation_type = model_data["type"]
    print(epochs)

    history = History()
    history = clientUpdate(train_loader, server_model, history, learning_rate, epochs, mu, aggregation_type)
    print("CLIENT UPDATE COMPLETE")

    response = serialize_history(history)
    print("RESPONSE")

    return jsonify(response)

@app.route("/q_client_update", methods=['POST'])
def q_client_update():
    model_data = request.get_json()
    server_model = load_model(model_data["model"])
    learning_rate = model_data["learning_rate"]
    epochs = model_data["epochs"]
    q_val = model_data["q_val"]

    history = History()
    history, delta, h = q_clientUpdate(train_loader, server_model, history, learning_rate, epochs, q_val)
    
    response = serialize_history(history)
    response['delta'] = serialize_tensor(delta)
    response['h'] = serialize_tensor(h)

    return response


# def serialize_model(model_list):
#     serialized = []
#     for model_state in model_list:
#         serialized.append({key: value.tolist() for key, value in model_state.items()})

#     return serialized

def serialize_model(model_list):
    serialized = []
    for model_state in model_list:
        serialized_state = {}
        for key, value in model_state.items():
            if isinstance(value, torch.Tensor):
                serialized_state[key] = value.cpu().detach().numpy().tolist()
            else:
                serialized_state[key] = value  # maybe float, etc.
        serialized.append(serialized_state)
    return serialized



def serialize_history(history):
    model = serialize_model(history.model)
    serialized = {
        'loss': serialize_tensor(history.loss), 
        'accuracy': history.accuracy, 
        'model': model,
        'variance': history.variance
    }
    return serialized

def serialize_tensor(val):
    if type(val) == list:
        return [v.cpu().detach().numpy().tolist() for v in val]
    else:
        return val.cpu().detach().numpy().tolist()


if __name__ == "__main__":
    # port = int(sys.argv[1]) if len(sys.argv) > 1 else 6010
    app.run(host="0.0.0.0", port=6001)