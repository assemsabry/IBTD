import torch
from torchvision import transforms, models
from PIL import Image
import sys

def run_prediction():
    model_path = "IBTD.pth"
    image_path = input("Enter path to brain MRI image: ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    classes = ["GLIOMA", "MENINGIOMA", "NOTUMOR", "PITUITARY"]
    print(f"Predicted class: {classes[predicted.item()]}")