# ============================================================================
# STAGE 2: Transfer Learning with EfficientNet + Fitzpatrick Skin Tones
# Full Integration with HAM10000 Pretrained Backbone
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Optional
import logging
import timm

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ============================================================================
# DIAGNOSIS MAPPING (DDI to HAM10000 + Your Dataset)
# ============================================================================

HAM10000_CATEGORIES = {
    "akiec": 0,  # Actinic keratoses
    "bcc": 1,  # Basal cell carcinoma
    "bkl": 2,  # Benign keratosis
    "df": 3,  # Dermatofibroma
    "mel": 4,  # Melanoma
    "nv": 5,  # Melanocytic nevi
    "vasc": 6,  # Vascular lesions
}


def get_fitzpatrick_from_class(class_name: str) -> int:
    """
    Extract Fitzpatrick group from class name like '12+melanocytic-nevi+False'.
    Returns: 0 (I-II), 1 (III-IV), 2 (V-VI)
    """
    fitz_str = class_name.split("+")[0]
    mapping = {"12": 0, "34": 1, "56": 2}
    return mapping.get(fitz_str, 1)


def get_diagnosis_from_class(class_name: str) -> str:
    """Extract diagnosis from class name."""
    return class_name.split("+")[1]


# ============================================================================
# SKIN TONE + IMAGE DATASET
# ============================================================================


class SkinToneLesionDataset(torch.utils.data.Dataset):
    """Dataset with both image and Fitzpatrick skin tone labels."""

    def __init__(self, dataset: datasets.ImageFolder, transform=None):
        self.dataset = dataset
        self.transform = transform

        # Extract Fitzpatrick groups and diagnoses
        self.fitzpatrick_groups = []
        self.diagnoses = []

        for class_name in dataset.classes:
            self.fitzpatrick_groups.append(get_fitzpatrick_from_class(class_name))
            self.diagnoses.append(get_diagnosis_from_class(class_name))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]

        # Get Fitzpatrick group from target class index
        fitzpatrick_group = self.fitzpatrick_groups[target]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(fitzpatrick_group, dtype=torch.long), target


# ============================================================================
# FITZPATRICK + EFFICIENTNET TRANSFER MODEL
# ============================================================================


class FitzpatrickEfficientNetModel(nn.Module):
    """
    EfficientNet backbone + Fitzpatrick skin tone embedding fusion.
    Learns to condition lesion classification on skin tone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 7,
        num_skin_groups: int = 3,
        embedding_dim: int = 8,
    ):
        super().__init__()
        self.backbone = backbone

        # Get backbone feature dimension
        if hasattr(backbone, "classifier"):
            # EfficientNet has .classifier
            backbone_features = (
                backbone.classifier.in_features
                if hasattr(backbone.classifier, "in_features")
                else 1280
            )
        else:
            backbone_features = backbone.fc.in_features

        # Replace classifier with identity
        if hasattr(backbone, "classifier"):
            backbone.classifier = nn.Identity()
        else:
            backbone.fc = nn.Identity()

        # Fitzpatrick embedding (3 groups: I-II, III-IV, V-VI)
        self.skin_embed = nn.Embedding(num_skin_groups, embedding_dim)

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(backbone_features + embedding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        logger.info(f"Model: EfficientNet + Fitzpatrick ({num_skin_groups} groups)")
        logger.info(
            f"Backbone features: {backbone_features}, Embedding: {embedding_dim}"
        )

    def forward(self, images: torch.Tensor, skin_groups: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
            skin_groups: (B,) - Fitzpatrick group indices 0-2
        Returns:
            logits: (B, num_classes)
        """
        img_feats = self.backbone(images)  # (B, backbone_features)
        skin_feats = self.skin_embed(skin_groups)  # (B, embedding_dim)
        combined = torch.cat([img_feats, skin_feats], dim=1)  # (B, combined_dim)
        logits = self.classifier(combined)
        return logits


# ============================================================================
# DATA PREPARATION
# ============================================================================


def prepare_skin_tone_data(
    data_dir: str,
    val_split: float = 0.2,
    batch_size: int = 16,
    random_state: int = 42,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Prepare dataloaders with image + Fitzpatrick skin tone labels.
    """
    # Data augmentation
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)
            ),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    class_names = full_dataset.classes

    logger.info(f"Dataset: {len(full_dataset)} images, {len(class_names)} classes")
    logger.info(f"Classes: {class_names}")

    # Stratified split
    labels = np.array(full_dataset.targets)
    indices = np.arange(len(labels))

    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=random_state, stratify=labels
    )

    # Create subsets
    train_dataset_base = Subset(full_dataset, train_idx)

    # Validation dataset with different transforms
    val_dataset_base = datasets.ImageFolder(data_dir, transform=val_transform)
    val_dataset_base_subset = Subset(val_dataset_base, val_idx)

    # Wrap with SkinToneLesionDataset
    train_dataset = SkinToneLesionDataset(full_dataset)
    val_dataset = SkinToneLesionDataset(val_dataset_base)

    # Apply indices
    train_dataset = Subset(train_dataset, train_idx)
    val_dataset = Subset(val_dataset, val_idx)

    # Class weights for imbalance
    train_labels = labels[train_idx]
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels
    )

    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Dataloaders
    num_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
    )

    info = {
        "class_weights": class_weights,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "num_classes": len(class_names),
        "class_names": class_names,
        "img_size": img_size,
    }

    return train_loader, val_loader, info


# ============================================================================
# EARLY STOPPING
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
        self.best_model_state = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }

    def load_best_model(self, model: nn.Module):
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model weights")


# ============================================================================
# TRAINING FUNCTION
# ============================================================================


def train_fitzpatrick_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: np.ndarray,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    patience: int = 10,
    save_path: str = "./models/fitzpatrick_efficientnet.pth",
) -> dict:
    """
    Train Fitzpatrick-conditioned EfficientNet model.
    """
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
        eps=1e-8,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, skin_groups, labels in train_loader:
            images = images.to(device)
            skin_groups = skin_groups.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, skin_groups)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
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
            for images, skin_groups, labels in val_loader:
                images = images.to(device)
                skin_groups = skin_groups.to(device)
                labels = labels.to(device)

                outputs = model(images, skin_groups)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_val_f1 = f1_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )

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

    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "best_val_loss": early_stopping.best_score,
            "num_classes": model.classifier[-1].out_features,
            "architecture": "EfficientNet-B4 + Fitzpatrick Embedding",
        },
        save_path,
    )
    logger.info(f"Model saved to {save_path}")

    return history


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    class_names: list,
    device: torch.device = device,
):
    """Evaluate model on validation set."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, skin_groups, labels in val_loader:
            images = images.to(device)
            skin_groups = skin_groups.to(device)
            labels = labels.to(device)

            outputs = model(images, skin_groups)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    logger.info("\nClassification Report:")
    print(
        classification_report(
            all_labels, all_preds, target_names=class_names, zero_division=0
        )
    )

    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    return all_labels, all_preds


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def run(
    data_dir: str,
    pretrained_checkpoint: Optional[str] = None,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    batch_size: int = 16,
    patience: int = 10,
    img_size: int = 224,
    save_path: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[nn.Module, dict]:
    """
    Main transfer learning pipeline with Fitzpatrick skin tones.

    Args:
        data_dir: Path to dataset directory
        pretrained_checkpoint: Path to HAM10000 pretrained EfficientNet-B4 model
        epochs: Number of training epochs
        learning_rate: Learning rate for fine-tuning (1e-4 recommended)
        batch_size: Batch size
        patience: Early stopping patience
        img_size: Input image size (224 for B4, 380 for B6)
        save_path: Path to save final model
        random_state: Random seed

    Returns:
        (model, history)
    """
    set_seed(random_state)

    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Dataset path not found: {data_dir}")

    # Prepare data
    logger.info("Loading dataset...")
    train_loader, val_loader, info = prepare_skin_tone_data(
        data_dir=data_dir,
        val_split=0.2,
        batch_size=batch_size,
        random_state=random_state,
        img_size=img_size,
    )

    # Load pretrained backbone
    logger.info("Loading pretrained backbone...")
    if pretrained_checkpoint and Path(pretrained_checkpoint).exists():
        ckpt = torch.load(pretrained_checkpoint, map_location=device)
        backbone = timm.create_model(
            "tf_efficientnet_b4", pretrained=False, num_classes=7
        )
        backbone.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(f"Loaded pretrained backbone from {pretrained_checkpoint}")

        # Freeze backbone for fine-tuning
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        logger.info("Using ImageNet pretrained EfficientNet-B4")
        backbone = timm.create_model("tf_efficientnet_b4", pretrained=True)

    # Create Fitzpatrick transfer model
    model = FitzpatrickEfficientNetModel(
        backbone=backbone,
        num_classes=info["num_classes"],
        num_skin_groups=3,
        embedding_dim=8,
    )
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable_params:,} / {total_params:,} parameters")

    # Train
    logger.info("Starting training...")
    history = train_fitzpatrick_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=info["class_weights"],
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        save_path=save_path or "./models/fitzpatrick_efficientnet.pth",
    )

    # Evaluate
    logger.info("Evaluating model...")
    evaluate(model, val_loader, info["class_names"], device)

    return model, history


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    set_seed(42)

    # Update these paths
    DDI_DATA_DIR = "../image_dataset_with_skin_tone"
    HAM10000_PRETRAINED = "./models/ham10000_efficientnet_b4.pth"  # From Stage 1

    model, history = run(
        data_dir=DDI_DATA_DIR,
        pretrained_checkpoint=HAM10000_PRETRAINED,
        epochs=50,
        learning_rate=1e-4,
        batch_size=16,
        patience=10,
        img_size=224,
        save_path="./models/fitzpatrick_transfer_efficientenet_b4.pth",
    )

    logger.info("Transfer learning pipeline complete!")
