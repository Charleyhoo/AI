import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


class VehicleDataset(Dataset):
    def __init__(self, directory):
        self.data_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        data_frame = pd.read_csv(self.data_files[index])

        data_tensor = torch.tensor(data_frame.values, dtype=torch.float32)  # Skip the header row
        input_data = data_tensor[:62, :]  # 62x12
        target_data = data_tensor[62:67, 0:2]  # 5x2
        return input_data, target_data


import torch.nn as nn

class VehiclePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(VehiclePredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -5:, :])  # Taking the last 5 time steps and predicting the next 5
        return out


def train_model(directory):
    dataset = VehicleDataset(directory)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    input_dim = 12  # number of features/labels
    hidden_dim = 50
    num_layers = 2
    output_dim = 2   # predict (x,y) for the next 5 time steps

    model = VehiclePredictor(input_dim, hidden_dim, num_layers, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    torch.save(model.state_dict(), 'vehicle_predictor.pth')

def predict_with_model(model_path, input_tensor):
    model = VehiclePredictor(12, 50, 2, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        predictions = model(input_tensor)
    return predictions


if __name__ == "__main__":
    directory = "training_data_new"

    # Train the model
    train_model(directory)

    # Load the trained model and make a prediction for demonstration
    sample_input = torch.randn(1, 62, 12)  # Random sample input

    # Print the randomly generated input
    print("Randomly generated input:")
    print(sample_input)

    predictions = predict_with_model('vehicle_predictor.pth', sample_input)

    print("Predicted coordinates for the next 5 time steps:")
    print(predictions)

