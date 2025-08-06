import requests
import sys
import socket

server_url = "http://128.61.43.145:5000/register"
def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

port = get_free_port()

response = requests.post(server_url, json=port)

if response.status_code == 200:
    print("Successfully registered")
else:
    print("Registration failed")

from flask import Flask, request
import torch
from torch.utils.data import DataLoader, TensorDataset

app = Flask(__name__)

# @app.route("/receive_data", methods = ["POST"])
# def receive_data():
#     data = request.get_json()
#     features = torch.tensor(data['features'], dtype=torch.float32)
#     labels = torch.tensor(data['labels'], dtype=torch.long)

#     client_dataset = TensorDataset(features, labels)
#     global client_loader
#     client_loader = DataLoader(client_dataset, batch_size=64, shuffle=True)

#     print(f"received dataset of length {len(client_dataset)}")
#     return {"message" : "Data received successfully."}, 200

if __name__ == "__main__":
    port = int(sys.argv[1])
    server_address = "128.61.43.145:5000"

    requests.post(f"http://{server_address}/register", json={"port": port})
    print(f"Registered client!")

    app.run(host="0.0.0.0", port=port)