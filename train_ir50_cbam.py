import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from models.ir50 import Backbone  # Import modified IR-50 with CBAM
import os

# ✅ Hyperparameters
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 3.5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "/content/drive/My Drive/PosterV2_CBAM/RAF-DB/DATASET"  # Change this to your dataset path
NUM_CLASSES = 7  # RAF-DB has 7 facial expression classes

# ✅ Data Transforms
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.Grayscale(num_output_channels=3),  # Ensuring 3-channel input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ✅ Load RAF-DB Dataset
train_dataset = ImageFolder(root=os.path.join(DATASET_PATH, "train"), transform=transform)
test_dataset = ImageFolder(root=os.path.join(DATASET_PATH, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ✅ Load IR-50 with CBAM
model = Backbone(num_layers=50, drop_ratio=0.0, mode="ir").to(DEVICE)

# ✅ Load Pretrained Weights (Optional, if available)
pretrained_path = "/content/PosterV2_CBAM_New/models/pretrain/ir50.pth"
if os.path.exists(pretrained_path):
    print(f"Loading pretrained IR-50 weights from {pretrained_path}...")
    pretrained_weights = torch.load(pretrained_path, map_location=DEVICE)["state_dict"]
    model_weights = model.state_dict()

    # Filter out incompatible layers
    filtered_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights and v.shape == model_weights[k].shape}

    # Load only compatible weights
    model_weights.update(filtered_weights)
    model.load_state_dict(model_weights, strict=False)

# ✅ Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ Training Loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            x1, x2, outputs = model(images)  # Extract final feature map
            outputs = outputs.view(outputs.size(0), -1)  # Flatten output
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")

        # Save Model Checkpoint Every 5 Epochs
        if (epoch + 1) % 5 == 0:
            torch.save({"state_dict": model.state_dict()}, f"ir50_cbam_epoch{epoch+1}.pth")
            print(f"Checkpoint saved: ir50_cbam_epoch{epoch+1}.pth")

# ✅ Testing Function
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, _, outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {100.*correct/total:.2f}%")

# ✅ Train and Save the Model
train()
torch.save({"state_dict": model.state_dict()}, "ir50_cbam.pth")
print("Final model saved: ir50_cbam.pth")

# ✅ Test Model
test()
