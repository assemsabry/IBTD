from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, dataloader, device):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            outputs = model(x)
            _, preds = outputs.max(1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred

def generate_report(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=2, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)
    return report, matrix
