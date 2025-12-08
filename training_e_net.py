# ============================================================================
# STAGE 1: EfficientNet-B6 on HAM10000 Dataset (HIGH PERFORMANCE)
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Tuple
import logging
from PIL import Image
import timm  # pip install timm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ============================================================================
# HAM10000 Dataset Class (224x224 for EfficientNet-B4, 380x380 for B6)
# ============================================================================
IMG_SIZE = 224

class HAM10000Dataset(torch.utils.data.Dataset):
    """Custom Dataset for HAM10000 with metadata."""
    
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
        # Map diagnosis to numeric labels
        self.label_map = {
            'akiec': 0,  # Actinic keratoses
            'bcc': 1,    # Basal cell carcinoma
            'bkl': 2,    # Benign keratosis
            'df': 3,     # Dermatofibroma
            'mel': 4,    # Melanoma
            'nv': 5,     # Melanocytic nevi
            'vasc': 6    # Vascular lesions
        }
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image_id"]
        
        # Try different image directories
        img_path = None
        for folder in ["HAM10000_images_part_1", "HAM10000_images_part_2", "HAM10000_images"]:
            path = os.path.join(self.img_dir, folder, f"{img_id}.jpg")
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row["dx"]]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ============================================================================
# Data Preparation with EfficientNet Augmentation
# ============================================================================
def prepare_ham10000_data(
    data_dir: str,
    metadata_path: str,
    val_split: float = 0.2,
    batch_size: int = 8, 
    random_state: int = 42,
    model_variant: str = "b4"  # "b4" or "b6"
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Prepare HAM10000 dataloaders optimized for EfficientNet.
    """
    global IMG_SIZE
    IMG_SIZE = 380 if model_variant == "b6" else 224
    
    # Load metadata
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} rows from metadata")
    
    # Remove duplicates
    df_unique = df.drop_duplicates(subset=["image_id"])
    logger.info(f"Unique images: {len(df_unique)}")
    
    # Class distribution
    class_counts = df_unique["dx"].value_counts()
    logger.info(f"Class distribution:\n{class_counts}")
    
    # Stratified split
    train_df, val_df = train_test_split(
        df_unique,
        test_size=val_split,
        stratify=df_unique["dx"],
        random_state=random_state
    )
    
    # ADVANCED AUGMENTATION for EfficientNet
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = HAM10000Dataset(train_df, data_dir, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, data_dir, transform=val_transform)
    
    # Class weights for sampling
    label_map = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
    labels = train_df["dx"].map(label_map).values
    
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Dataloaders with gradient accumulation support
    num_workers = min(0, os.cpu_count() or 1)  # Conservative for large images
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger val batch
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    
    info = {
        "class_weights": class_weights,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "num_classes": 7,
        "class_names": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        "img_size": IMG_SIZE
    }
    
    return train_loader, val_loader, info

# ============================================================================
# EfficientNet Model Setup (timm library)
# ============================================================================
def create_efficientnet_model(
    model_name: str = "tf_efficientnet_b4",  # or "tf_efficientnet_b4"
    num_classes: int = 7,
    freeze_backbone: bool = False,
    dropout: float = 0.3
) -> nn.Module:
    """
    Create EfficientNet model using timm library.
    """
    model = timm.create_model(
        model_name,
        pretrained=True,  # ImageNet weights
        num_classes=num_classes,
        drop_rate=dropout,
        drop_path_rate=0.2
    )
    
    if freeze_backbone:
        # Freeze feature extractor
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze classifier head
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_name}")
    logger.info(f"Trainable: {trainable_params:,} / {total_params:,} parameters")
    logger.info(f"Input size: {IMG_SIZE}x{IMG_SIZE}")
    
    return model

# ============================================================================
# Enhanced Training with Gradient Accumulation
# ============================================================================
def train_efficientnet_ham10000(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: np.ndarray,
    epochs: int = 50,
    learning_rate: float = 3e-5,  # Lower LR for EfficientNet
    patience: int = 10,
    save_path: str = "./models/ham10000_efficientnet.pth",
    accum_steps: int = 4  # Gradient accumulation steps
) -> dict:
    """
    Train EfficientNet on HAM10000 with gradient accumulation.
    """
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device)
    )
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
        eps=1e-8
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )
    
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": []
    }
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        optimizer.zero_grad()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accum_steps  # Scale loss
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accum_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * inputs.size(0) * accum_steps
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_val_f1 = f1_score(all_labels, all_preds, average="weighted")
        
        # Update history
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_f1"].append(epoch_val_f1)
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train L: {epoch_train_loss:.4f} A: {epoch_train_acc:.4f} | "
            f"Val L: {epoch_val_loss:.4f} A: {epoch_val_acc:.4f} F1: {epoch_val_f1:.4f}"
        )
        
        scheduler.step(epoch_val_loss)
        
        if early_stopping(epoch_val_loss, model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    early_stopping.load_best_model(model)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "best_val_loss": early_stopping.best_score,
        "num_classes": 7,
        "model_name": "tf_efficientnet_b6",
        "img_size": IMG_SIZE,
        "class_names": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    }, save_path)
    logger.info(f"Model saved to {save_path}")
    
    return history

# Include EarlyStopping class from previous code (same as before)
class EarlyStopping:
    """Early stopping with model checkpointing."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            return False
        
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def save_checkpoint(self, model: nn.Module):
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    def load_best_model(self, model: nn.Module):
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model weights")


# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    set_seed(42)
    
    # Paths - UPDATE THESE
    HAM10000_DATA_DIR = "../dataverse_files"        
    HAM10000_METADATA = "../dataverse_files/HAM10000_metadata"
    
    # Choose B4 (lighter, 224x224) or B6 (better, 380x380)
    MODEL_VARIANT = "b4"  # Change to "b4" for lighter model
    
    # Prepare data
    train_loader, val_loader, info = prepare_ham10000_data(
        data_dir=HAM10000_DATA_DIR,
        metadata_path=HAM10000_METADATA,
        val_split=0.2,
        batch_size=8,
        model_variant=MODEL_VARIANT
    )
    
    # Create EfficientNet model
    model_name = f"tf_efficientnet_{MODEL_VARIANT}"
    model = create_efficientnet_model(
        model_name=model_name,
        num_classes=info["num_classes"],
        freeze_backbone=False
    )
    
    # Train
    history = train_efficientnet_ham10000(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=info["class_weights"],
        epochs=50,
        learning_rate=3e-5,
        patience=10,
        save_path=f"./models/ham10000_efficientnet_{MODEL_VARIANT}.pth"
    )
    
    logger.info("Stage 1 COMPLETE! Ready for Stage 2 fine-tuning.")
