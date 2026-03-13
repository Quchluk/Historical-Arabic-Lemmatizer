#!/usr/bin/env python3
"""
Advanced Training Script with Model Collapse Fixes
==================================================
Features:
1. Zipf-Aware Balanced Sampling (Inverse Square Root)
2. Batch Hard Triplet Loss (Dynamic Hard Mining)
3. Stratified Train/Val Split (Head vs. Tail)
4. Real-time Validation with Zero-Shot Evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim import AdamW
from pathlib import Path
import logging
from tqdm import tqdm
import time
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split

from model import LemmaProjector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_advanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# BATCH HARD TRIPLET LOSS - Fixes the 0.97 similarity collapse
# ============================================================================

def batch_hard_triplet_loss(embeddings, labels, margin=1.0):
    """
    Batch Hard Triplet Loss with automatic hard negative mining.

    For each anchor:
    - Hardest Positive: Furthest sample with SAME label
    - Hardest Negative: Closest sample with DIFFERENT label

    This forces the model to focus on the difficult cases that are
    causing the collapse (e.g., different words with 0.97 similarity).

    Args:
        embeddings: [Batch, Embedding_Dim] - Model outputs
        labels: [Batch] - Lemma IDs
        margin: Triplet loss margin

    Returns:
        Scalar loss value
    """
    # 1. Calculate pairwise Euclidean distance matrix [Batch x Batch]
    # Manual implementation to avoid MPS cdist_backward issue
    # dist(x, y) = sqrt(sum((x - y)^2))
    # dist_matrix[i,j] = ||embeddings[i] - embeddings[j]||

    # Expand dimensions for broadcasting: [N, 1, D] and [1, N, D]
    emb_i = embeddings.unsqueeze(1)  # [N, 1, D]
    emb_j = embeddings.unsqueeze(0)  # [1, N, D]

    # Calculate squared differences and sum over dimension
    diff_squared = (emb_i - emb_j) ** 2  # [N, N, D]
    dist_matrix = torch.sqrt(diff_squared.sum(dim=2) + 1e-12)  # [N, N], add epsilon for numerical stability

    # 2. Create masks for same/different labels
    labels = labels.unsqueeze(1)  # [Batch, 1]
    mask_pos = (labels == labels.T).float()  # 1 where labels match
    mask_neg = (labels != labels.T).float()  # 1 where labels differ

    # 3. Hardest Positive: MAX distance where label is SAME
    # We zero out negatives by multiplying with mask_pos
    # We also zero out diagonal (self-distance) to avoid picking anchor itself
    mask_pos_no_diag = mask_pos.clone()
    mask_pos_no_diag.fill_diagonal_(0)

    # For positions where mask is 0, set distance to -inf so they aren't picked as max
    dist_pos = dist_matrix * mask_pos_no_diag + (1 - mask_pos_no_diag) * -1e9
    hardest_pos_dist, _ = dist_pos.max(dim=1)

    # 4. Hardest Negative: MIN distance where label is DIFFERENT
    # For positions where mask is 0, set distance to +inf so they aren't picked as min
    dist_neg = dist_matrix * mask_neg + (1 - mask_neg) * 1e9
    hardest_neg_dist, _ = dist_neg.min(dim=1)

    # 5. Triplet Loss: max(0, d(a,p) - d(a,n) + margin)
    loss = F.relu(hardest_pos_dist - hardest_neg_dist + margin)

    # Filter out invalid cases (where no positive or negative exists)
    # This can happen if batch contains only one label
    valid_mask = (hardest_pos_dist > -1e8) & (hardest_neg_dist < 1e8)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    return loss[valid_mask].mean()


# ============================================================================
# ZIPF-AWARE BALANCED SAMPLER
# ============================================================================

def get_balanced_sampler(labels):
    """
    Create weighted sampler using Inverse Square Root of frequency.

    This implements "Robin Hood" sampling:
    - Frequent words (like 'wa' with 16K samples) get weight ~ 1/sqrt(16000) = 0.0079
    - Rare words (appearing 4 times) get weight ~ 1/sqrt(4) = 0.5

    This makes rare words ~63x more likely to be sampled than they would be naturally,
    flattening the Zipfian distribution without discarding data.

    Args:
        labels: Tensor of label IDs [N]

    Returns:
        WeightedRandomSampler
    """
    logger.info("Creating Zipf-aware balanced sampler...")

    # Count frequency of each lemma
    label_array = labels.numpy()
    counts = Counter(label_array)

    # Calculate inverse square root weights for each sample
    weights = []
    for label in label_array:
        freq = counts[label]
        weight = 1.0 / np.sqrt(freq)
        weights.append(weight)

    weights = torch.DoubleTensor(weights)

    # Log statistics
    logger.info(f"Weight statistics:")
    logger.info(f"  Min weight: {weights.min():.6f} (most frequent)")
    logger.info(f"  Max weight: {weights.max():.6f} (most rare)")
    logger.info(f"  Weight ratio: {weights.max()/weights.min():.2f}x")

    # Create sampler with replacement (so we can sample more than dataset size)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True
    )

    return sampler


# ============================================================================
# STRATIFIED ZIPF SPLIT - Head vs. Tail
# ============================================================================

def stratified_zipf_split(dataset, val_ratio=0.1, rare_threshold=3):
    """
    Smart split that handles Zipfian distribution:

    - Rare lemmas (count < rare_threshold) → 100% to TRAIN
      (Can't split them, they need all samples to learn)

    - Frequent lemmas (count ≥ rare_threshold) → Split train/val
      (Ensures validation has enough samples per lemma)

    This prevents wasting rare data on validation where it can't form pairs.

    Args:
        dataset: Dataset with .labels attribute
        val_ratio: Fraction of FREQUENT lemmas to use for validation
        rare_threshold: Lemmas with < this many samples go to train

    Returns:
        (train_subset, val_subset)
    """
    logger.info(f"\nPerforming stratified Zipf split (rare_threshold={rare_threshold})...")

    labels = dataset.labels.numpy()
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Group A: Rare lemmas → TRAIN only
    rare_mask = counts < rare_threshold
    rare_lemmas = unique_labels[rare_mask]

    # Group B: Frequent lemmas → Split
    freq_mask = counts >= rare_threshold
    freq_lemmas = unique_labels[freq_mask]

    logger.info(f"Rare lemmas (< {rare_threshold} samples): {len(rare_lemmas):,}")
    logger.info(f"Frequent lemmas (≥ {rare_threshold} samples): {len(freq_lemmas):,}")

    # Split frequent lemmas themselves (zero-shot split)
    if len(freq_lemmas) > 1:
        train_freq, val_freq = train_test_split(
            freq_lemmas,
            test_size=val_ratio,
            random_state=42
        )
    else:
        train_freq = freq_lemmas
        val_freq = np.array([])

    # Combine: Train = Rare + Train_Freq
    final_train_lemmas = set(np.concatenate([rare_lemmas, train_freq]))
    final_val_lemmas = set(val_freq)

    logger.info(f"\nFinal split:")
    logger.info(f"  Train lemmas: {len(final_train_lemmas):,}")
    logger.info(f"  Val lemmas:   {len(final_val_lemmas):,}")

    # Create indices
    train_idx = [i for i, l in enumerate(labels) if l in final_train_lemmas]
    val_idx = [i for i, l in enumerate(labels) if l in final_val_lemmas]

    logger.info(f"  Train samples: {len(train_idx):,}")
    logger.info(f"  Val samples:   {len(val_idx):,}")

    # Create subsets
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    return train_subset, val_subset


# ============================================================================
# ADVANCED BATCH DATASET - Returns (embedding, label) instead of triplets
# ============================================================================

class BatchDataset(Dataset):
    """
    Dataset that returns (embedding, label) pairs.
    Triplets are constructed dynamically in the loss function.
    """

    def __init__(self, data_file: str = 'train_vectors.pt'):
        if not Path(data_file).exists():
            raise FileNotFoundError(f"Training data not found: {data_file}")

        data = torch.load(data_file, map_location='cpu')
        self.vectors = data['vectors']
        self.labels = data['labels']

        logger.info(f"Loaded {len(self.vectors):,} samples")
        logger.info(f"Unique labels: {len(torch.unique(self.labels)):,}")

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]


# ============================================================================
# TRAINER
# ============================================================================

class AdvancedTrainer:
    """Trainer with Model Collapse Prevention."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        logger.info(f"Model moved to device: {device}")
        logger.info(f"Checkpoints: {self.checkpoint_dir}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 20,
        learning_rate: float = 1e-4,
        margin: float = 1.0,
        warmup_epochs: int = 2
    ):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Learning rate scheduler (warmup then decay)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs  # Linear warmup
            else:
                return 0.5 ** ((epoch - warmup_epochs) / 10)  # Exponential decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        logger.info("=" * 80)
        logger.info("STARTING ADVANCED TRAINING (Anti-Collapse)")
        logger.info("=" * 80)
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch size: {train_loader.batch_size}")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Margin: {margin}")
        logger.info(f"Loss: Batch Hard Triplet Loss")
        logger.info("=" * 80)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # --- TRAINING ---
            self.model.train()
            train_loss = 0.0
            train_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

            for embeddings, labels in pbar:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                projected = self.model(embeddings)

                # Batch Hard Loss
                loss = batch_hard_triplet_loss(projected, labels, margin=margin)

                # Backward
                loss.backward()

                # Gradient clipping (prevents instability)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                train_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = train_loss / train_batches

            # --- VALIDATION ---
            self.model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for embeddings, labels in val_loader:
                    embeddings = embeddings.to(self.device)
                    labels = labels.to(self.device)

                    projected = self.model(embeddings)
                    loss = batch_hard_triplet_loss(projected, labels, margin=margin)

                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')

            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            epoch_time = time.time() - epoch_start

            # --- LOGGING ---
            logger.info(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(f"  Val Loss:   {avg_val_loss:.4f}")
            logger.info(f"  LR:         {current_lr:.6f}")
            logger.info(f"  Time:       {epoch_time:.2f}s")

            # --- CHECKPOINTING ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_checkpoint(epoch, avg_val_loss, is_best=True)
                logger.info(f"  ✓ New best model (Val Loss: {avg_val_loss:.4f})")

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, avg_val_loss, is_best=False)

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE!")
        logger.info(f"Best Val Loss: {best_val_loss:.4f}")
        logger.info("=" * 80)

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
        }

        if is_best:
            filename = 'advanced_best.pth'
        else:
            filename = f'advanced_epoch_{epoch+1}.pth'

        torch.save(checkpoint, self.checkpoint_dir / filename)
        torch.save(checkpoint, self.checkpoint_dir / 'advanced_latest.pth')


# ============================================================================
# MAIN
# ============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    # 1. Load dataset
    logger.info("\nLoading dataset...")
    dataset = BatchDataset('train_vectors.pt')

    # 2. Stratified split (rare lemmas stay in train)
    train_dataset, val_dataset = stratified_zipf_split(
        dataset,
        val_ratio=0.15,
        rare_threshold=3
    )

    # 3. Create balanced sampler for training
    # Extract labels from train subset
    train_indices = train_dataset.indices
    train_labels = dataset.labels[train_indices]
    balanced_sampler = get_balanced_sampler(train_labels)

    # 4. Create data loaders
    batch_size = 64

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=balanced_sampler,  # Use balanced sampling
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"\nData loaders created:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches:   {len(val_loader)}")

    # 5. Initialize model
    model = LemmaProjector(
        input_dim=768,
        hidden_dim=512,
        output_dim=128,
        dropout=0.2
    )

    logger.info(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Train
    trainer = AdvancedTrainer(model, device)
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=25,
        learning_rate=1e-4,
        margin=1.0,
        warmup_epochs=2
    )

    logger.info("\n✓ Training complete! Run test_inference.py to verify.")


if __name__ == "__main__":
    main()

