import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import argparse
import sys
from PIL import Image
from tqdm import tqdm

# -----------------------------
# Training Function
# -----------------------------
def train_resnet_classifier(data_dir, output_model_path='models/violation_verifier.pth', num_epochs=25):
    """Trains a ResNet50 model to classify images as 'violation' or 'safe'."""
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val']}
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found in {data_dir}. Ensure 'train' and 'val' folders exist.")
        return

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Classes: {class_names}")

    model_ft = models.resnet50(pretrained=True)
    
    # Modify the classifier output for ResNet50
    num_ftrs = model_ft.fc.in_features
    # Determine safe/violation mapping based on folder names
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    since = time.time()
    best_model_wts = copy.deepcopy(model_ft.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model_ft.train()
            else:
                model_ft.eval()

            running_loss = 0.0
            running_corrects = 0

            # Create progress bar
            pbar = tqdm(dataloaders[phase], desc=f"{phase} Phase", unit="batch")
            
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer_ft.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar description
                pbar.set_postfix({'loss': loss.item()})
            
            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_ft.state_dict())

    model_ft.load_state_dict(best_model_wts)
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(model_ft.state_dict(), output_model_path)
    print(f"✅ Training complete. Model saved to {output_model_path}")

def load_verifier(model_path='models/violation_verifier.pth'):
    """Load the trained ResNet50 verifier model."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model architecture (ResNet50)
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    # Binary classification: 0=Safe, 1=Violation
    model.fc = nn.Linear(num_ftrs, 2)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"✅ Loaded verifier model (ResNet50) from {model_path}")
        return model, device
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

def verify_image(model, device, image_path):
    """
    Run verification on a single image.
    Returns: 'violation' or 'safe' and confidence score.
    """
    if model is None:
        return "error", 0.0

    # Transformations must match training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
       # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        safe_prob = probs[0][0].item()
        violation_prob = probs[0][1].item()
        
        idx = torch.argmax(probs).item()
        confidence = torch.max(probs).item()
        
        # 0 = safe, 1 = violation
        label = 'VIOLATION' if idx == 1 else 'SAFE'
        
        print(f"\n🔍 Image: {os.path.basename(image_path)}")
        print(f"📊 Probabilities: Safe={safe_prob:.4f}, Violation={violation_prob:.4f}")
        print(f"✅ Prediction: {label} ({confidence:.4f})\n")
        
        return label, confidence
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "error", 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ResNet50: Train or Verify')
    
    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Train the model using a dataset')
    group.add_argument('--verify', type=str, metavar='IMAGE_PATH', help='Verify a single image path')
    
    # Arguments
    parser.add_argument('--data_dir', type=str, help='Path to dataset directory (required for --train)')
    parser.add_argument('--model', type=str, default='models/violation_verifier.pth', help='Path to model file')
    parser.add_argument('--epochs', type=int, default=25, help='Epochs to train')

    args = parser.parse_args()

    if args.train:
        if not args.data_dir:
            print("❌ Error: --data_dir is required for training.")
        else:
            train_resnet_classifier(args.data_dir, args.model, args.epochs)
            
    elif args.verify:
        if not os.path.exists(args.verify):
            print(f"❌ Error: Image not found at {args.verify}")
        else:
            model, device = load_verifier(args.model)
            if model:
                label, conf = verify_image(model, device, args.verify)
                print(f" Prediction: {label.upper()} ({conf:.4f})")
