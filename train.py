import os
import random
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet50_Weights

def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    print("\n🚀 Initializing training...")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = ""
    test_dir = ""

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("❌ Error: Training or testing directory does not exist.")
        return

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    if len(full_train_data) == 0:
        print("❌ No training images found.")
        return

    val_ratio = 0.15
    val_size = int(len(full_train_data) * val_ratio)
    train_size = len(full_train_data) - val_size
    train_data, val_data = random_split(full_train_data, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    if len(train_data) == 0:
        print("❌ Training set is empty after split.")
        return

    print(f"✅ Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(full_train_data.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    writer = SummaryWriter(log_dir="runs/IBTD")

    best_val_loss = float('inf')
    best_val_acc = 0

    print("✅ Starting training loop...")

    for epoch in range(30):
        model.train()
        running_loss = 0.0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, targets_a, targets_b, lam = mixup_data(imgs, labels)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"🌀 Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_loss /= len(val_loader)

        scheduler.step(val_acc)
        writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"✅ Epoch {epoch+1}/30 | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), "IBTD.pth")

    model.load_state_dict(torch.load("IBTD.pth"))
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    target_names = full_train_data.classes
    print("\n==========  Intelligent Brain Tumor Detector (IBTD) Evaluation ==========\n")
    print(" Classification Report:\n")
    report = classification_report(y_true, y_pred, target_names=target_names, digits=2, output_dict=True)

    for cls in target_names:
        cls_report = report[cls]
        print(f"{cls.upper()} ➔ Precision: {cls_report['precision']*100:.2f}% | Recall: {cls_report['recall']*100:.2f}% | F1-score: {cls_report['f1-score']*100:.2f}% | Support: {cls_report['support']:.0f}")

    overall = report['accuracy']
    print(f"\n📊 Overall Accuracy: {overall*100:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)

    writer.close()

if __name__ == "__main__":
    main()