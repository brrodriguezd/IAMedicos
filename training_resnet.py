# ============================================================================
# STAGE 1: Train ResNet50 on HAM10000 Dataset
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Tuple
import logging
from PIL import Image

os.environ['OMP_NUM_THREADS'] = '2'  # P1000 + 8GB limits
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Debug OOM
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = False  # Stable for low VRAM

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
# HAM10000 Dataset Class
# ============================================================================
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
        
        # Try different image directories (HAM10000 splits images into folders)
        img_path = None
        for folder in ["HAM10000_images_part_1", "HAM10000_images_part_2", "HAM10000_images"]:
            path = os.path.join(self.img_dir, folder, f"{img_id}.jpg")
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            # Fallback to direct path
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row["dx"]]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ============================================================================
# Data Preparation
# ============================================================================
def prepare_ham10000_data(
    data_dir: str,
    metadata_path: str,
    val_split: float = 0.2,
    batch_size: int = 2,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Prepare HAM10000 dataloaders with proper handling of class imbalance.
    """
    # Load metadata
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} rows from metadata")
    
    # Remove duplicates (HAM10000 has duplicate images)
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
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = HAM10000Dataset(train_df, data_dir, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, data_dir, transform=val_transform)
    
    # Compute class weights for weighted sampling
    label_map = {
        'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3,
        'mel': 4, 'nv': 5, 'vasc': 6
    }
    labels = train_df["dx"].map(label_map).values
    
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    
    # Create weighted sampler to handle class imbalance
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders
    num_workers = min(0, os.cpu_count() or 1)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
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
        "class_names": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    }
    
    return train_loader, val_loader, info

# ============================================================================
# Model Setup
# ============================================================================
def create_model(num_classes: int = 7, freeze_backbone: bool = False) -> nn.Module:
    """
    Create ResNet50 model for HAM10000.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    if freeze_backbone:
        # Freeze all layers except layer4 and fc
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
    
    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable_params:,} / {total_params:,} parameters")
    
    return model

# ============================================================================
# Early Stopping
# ============================================================================
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
# Training Function
# ============================================================================
def train_ham10000(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: np.ndarray,
    epochs: int = 30,
    learning_rate: float = 1e-4,
    patience: int = 7,
    save_path: str = "./models/ham10000_resnet50.pth"
) -> dict:
    """
    Train model on HAM10000 dataset.
    """
    # Loss with class weights
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device)
    )
    
    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
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
        all_preds = []
        all_labels = []
        
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
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} F1: {epoch_val_f1:.4f}"
        )
        
        # Learning rate scheduling
        scheduler.step(epoch_val_loss)
        
        # Early stopping
        if early_stopping(epoch_val_loss, model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    early_stopping.load_best_model(model)
    
    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "best_val_loss": early_stopping.best_score,
        "num_classes": 7,
        "class_names": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    }, save_path)
    logger.info(f"Model saved to {save_path}")
    
    return history

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    set_seed(42)
    
    HAM10000_DATA_DIR = "../dataverse_files"        # Folder containing image subfolders
    HAM10000_METADATA = "../dataverse_files/HAM10000_metadata"
    
    # Prepare data
    train_loader, val_loader, info = prepare_ham10000_data(
        data_dir=HAM10000_DATA_DIR,
        metadata_path=HAM10000_METADATA,
        val_split=0.2,
        batch_size=32,
        random_state=42
    )
    
    # Create model
    model = create_model(num_classes=info["num_classes"], freeze_backbone=False)
    
    # Train
    history = train_ham10000(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=info["class_weights"],
        epochs=30,
        learning_rate=1e-4,
        patience=7,
        save_path="./models/ham10000_resnet50.pth"
    )
    
    logger.info("Stage 1 training complete!")
