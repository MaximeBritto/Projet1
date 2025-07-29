.000000000000000............................................................0import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import onnx
import onnxruntime as ort

# Définition du modèle
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

def train():
    # Chargement des données
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True)

    # Initialisation du modèle
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Entraînement
    model.train()
    for epoch in range(5):  # 5 époques pour un entraînement rapide
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Époque: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tPerte: {loss.item():.6f}')
    
    return model

def export_to_onnx(model):
    # Création d'un exemple d'entrée
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Export du modèle en ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "mnist_model.onnx",
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Vérification du modèle exporté
    onnx_model = onnx.load("mnist_model.onnx")
    onnx.checker.check_model(onnx_model)
    print("Modèle exporté avec succès en ONNX!")

if __name__ == "__main__":
    print("Début de l'entraînement...")
    model = train()
    print("Entraînement terminé!")
    
    print("Export du modèle en ONNX...")
    export_to_onnx(model)
    print("Tâche terminée!")
