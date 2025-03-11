from __future__ import absolute_import

import shutil
import time
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from loguru import logger

from .models_example import SimpleNN

ROOT = "/Users/davidzheng/projects/202501-crew-ai/data"


def load_data(train_batch_size: int, test_batch_size: int):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = torchvision.datasets.MNIST(
        root=ROOT, train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=ROOT, train=False, transform=transform, download=True
    )
    logger.info(
        f"Num train samples: {len(train_dataset):,}, Num test samples: {len(test_dataset):,}"
    )
    logger.info(
        f"sample 0 input mean: {train_dataset[0][0].mean():.4f} input std: {train_dataset[0][0].std():.4f} target: {train_dataset[0][1]}"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False
    )

    return train_loader, test_loader


def get_device():
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def create_model_criterion_optim(device, lr: float):
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer


def train(
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    num_epochs: int,
    num_samples_seen: int,
    checkpoint_path: Path,
) -> None:
    best_test_loss = float("inf")
    best_test_accuracy = 0.0
    while epoch < num_epochs:
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            num_samples_seen += len(inputs)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        test_loss, test_accuracy = test(
            device,
            test_loader,
            model,
            criterion,
        )
        best_test_loss = min(test_loss, best_test_loss)
        best_test_accuracy = max(test_accuracy, best_test_accuracy)

        logger.info(
            f"Epoch {epoch + 1} finished in {time.time() - epoch_start_time:.2f}s | Num Samples Seen: {num_samples_seen:,} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}% | Best Test Loss: {best_test_loss:.4f} | Best Test Accuracy: {best_test_accuracy:.2f}%"
        )

        epoch += 1
        # Save the best model
        save_checkpoint(
            epoch,
            num_samples_seen,
            model,
            optimizer,
            train_loss,
            test_loss,
            best_test_loss,
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved at epoch {epoch}")


def test(
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    test_loss = running_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    return test_loss, test_accuracy


def save_checkpoint(
    epoch: int,
    num_samples_seen: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loss: float,
    test_loss: float,
    best_test_loss: float,
    checkpoint_path: Path,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "num_samples_seen": num_samples_seen,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "test_loss": test_loss,
        "best_test_loss": best_test_loss,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    num_samples_seen = checkpoint["num_samples_seen"]
    logger.info(
        f"Checkpoint loaded from epoch {epoch} with train loss {checkpoint['train_loss']:.4f} test loss {checkpoint['test_loss']:.4f} best test loss {checkpoint['best_test_loss']:.4f}"
    )
    return epoch, num_samples_seen


@click.command()
@click.option(
    "--outdir",
    type=Path,
    help="Outdir",
    required=True,
)
@click.option(
    "--num-epochs",
    type=int,
    help="Num epochs",
    default=20,
)
@click.option(
    "--seed",
    type=int,
    help="Seed",
    default=42,
)
@click.option(
    "--train-batch-size",
    type=int,
    help="Train batch size",
    default=128,
)
@click.option(
    "--test-batch-size",
    type=int,
    help="Test batch size",
    default=1000,
)
@click.option(
    "--lr",
    type=float,
    help="Learning rate",
    default=0.0001,
)
@click.option(
    "--clean",
    is_flag=True,
    help="Clean outdir",
)
def main(
    outdir: Path,
    num_epochs: int,
    seed: int,
    train_batch_size: int,
    test_batch_size: int,
    lr: float,
    clean: bool,
) -> None:
    outdir = outdir.resolve()
    if clean:
        logger.info(f"** Cleaning outdir {outdir} **")
        shutil.rmtree(outdir)
    outdir.mkdir(exist_ok=True)
    logger.add(outdir / "train.log")

    logger.info("** Training Configuration **")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Learning Rate: {lr}")
    logger.info(f"Train Batch Size: {train_batch_size}")
    logger.info(f"Test Batch Size: {test_batch_size}")
    logger.info(f"Random Seed: {seed}")
    logger.info(f"Output Directory: {outdir}")
    logger.info("Dataset: MNIST")
    logger.info("Architecture: Dense")
    logger.info("Optimizer: Adam")

    device = get_device()
    logger.info(f"Device: {device}")
    torch.manual_seed(seed)
    logger.info("** Creating model, criterion, optimizer **")
    model, criterion, optimizer = create_model_criterion_optim(device, lr)
    epoch = 0
    num_samples_seen = 0
    checkpoint_path = outdir / "model.pth"
    if checkpoint_path.exists():
        logger.info(f"** Loading checkpoint from {checkpoint_path} **")
        epoch, num_samples_seen = load_checkpoint(
            checkpoint_path, device, model, optimizer
        )

    logger.info("** Loading data **")
    train_loader, test_loader = load_data(train_batch_size, test_batch_size)

    logger.info("** Training **")
    train(
        device,
        train_loader,
        test_loader,
        model,
        criterion,
        optimizer,
        epoch,
        num_epochs,
        num_samples_seen,
        checkpoint_path,
    )


if __name__ == "__main__":
    main()
