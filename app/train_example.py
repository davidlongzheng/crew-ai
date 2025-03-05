from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from app.models_example import SimpleNN

ROOT = "/home/davidzheng/projects/202501-crew-ai/data"


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = torchvision.datasets.MNIST(
        root=ROOT, train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=ROOT, train=False, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )

    return train_loader, test_loader


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model_criterion_optim(device):
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    checkpoint_path: Path,
) -> None:
    best_test_loss = float("inf")

    while epoch < num_epochs:
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)
        test_loss = test(
            device,
            test_loader,
            model,
            criterion,
        )
        best_test_loss = min(test_loss, best_test_loss)

        print(
            f"Epoch {epoch + 1} finished | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Best Test Loss: {best_test_loss:4}"
        )

        # Save the best model
        save_checkpoint(
            epoch,
            model,
            optimizer,
            train_loss,
            test_loss,
            best_test_loss,
            checkpoint_path,
        )
        print(f"Checkpoint saved at epoch {epoch + 1}")

        epoch += 1


def test(
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
) -> float:
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.2f}%")
    return avg_test_loss


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loss: float,
    test_loss: float,
    best_test_loss: float,
    checkpoint_path: Path,
) -> None:
    checkpoint = {
        "epoch": epoch,
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
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(
        f"Checkpoint loaded from epoch {epoch + 1} with train loss {checkpoint['train_loss']:.4f} test loss {checkpoint['test_loss']:.4f} best test loss {checkpoint['best_test_loss']:.4f}"
    )
    return epoch


@click.command()
@click.option(
    "--outdir",
    type=click.Path(exists=False),
    help="Outdir",
    required=True,
)
@click.option(
    "--num-epochs",
    type=int,
    help="Num epochs",
    default=10,
)
def main(outdir: Path, num_epochs: int) -> None:
    device = get_device()
    model, criterion, optimizer = create_model_criterion_optim(device)
    epoch = 0
    checkpoint_path = outdir / "model.pth"
    if checkpoint_path.exists():
        epoch = load_checkpoint(checkpoint_path, device, model, optimizer)
    train_loader, test_loader = load_data()

    train(
        device,
        train_loader,
        test_loader,
        model,
        criterion,
        optimizer,
        epoch,
        num_epochs,
        checkpoint_path,
    )


if __name__ == "__main__":
    main()
