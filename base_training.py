import json
import torch
import argparse

from src.data_utils import get_cifar10_dataloaders
from src.data_utils import get_cifar100_dataloaders
from src.data_utils import get_mnist_dataloaders
from src.data_utils import get_svhn_dataloaders
from src.data_utils import get_caltech256_dataloaders
from src.models import ResNet
from src.models import MultilayerPerceptron
from src.base_trainer import BaseTrainer

torch.manual_seed(42)

parser = argparse.ArgumentParser(
    prog='KD2M - Base training',
    description='Trains the teacher and student networks',
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128
)
parser.add_argument(
    "--n_epochs",
    type=int,
    default=15
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-2
)
parser.add_argument(
    "--benchmark",
    type=str,
    default="mnist"
)
args = parser.parse_args()

# Device
device = torch.device('cuda')
batch_size = args.batch_size
n_epochs = args.n_epochs
lr = args.lr
benchmark = args.benchmark

if benchmark == "mnist":
    train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)
    # Create nets
    teacher = MultilayerPerceptron(depth=5)
    student = MultilayerPerceptron(depth=3)
elif benchmark == "cifar10":
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)
    # Create nets
    teacher = ResNet(resnet_size="34")
    student = ResNet(resnet_size="18")
elif benchmark == "cifar100":
    train_loader, test_loader = get_cifar100_dataloaders(batch_size=batch_size)
    # Create nets
    teacher = ResNet(resnet_size="34", n_classes=100)
    student = ResNet(resnet_size="18", n_classes=100)
elif benchmark == "svhn":
    train_loader, test_loader = get_svhn_dataloaders(batch_size=batch_size)
    # Create nets
    teacher = ResNet(resnet_size="34")
    student = ResNet(resnet_size="18")
elif benchmark == "caltech":
    train_loader, test_loader, class_names = get_caltech256_dataloaders(
        batch_size=batch_size)
    print(f"{len(class_names)} classes")
    # Create nets
    teacher = ResNet(resnet_size="34", n_classes=258)
    student = ResNet(resnet_size="18", n_classes=258)
else:
    raise ValueError(f"Unknown benchmark: {benchmark}")

# Saves student weights
torch.save(
    student.state_dict(),
    f"./pretrained/{benchmark}/student_weights_0.pth"
)

print("\n\n")
print("-" * 100)
print("Training the Teacher")
print("-" * 100)
optimizer = torch.optim.SGD(
    lr=lr,
    momentum=0.9,
    params=teacher.parameters()
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_epochs,
    eta_min=1e-4
)
trainer = BaseTrainer(
    criterion=torch.nn.CrossEntropyLoss(),
    optimizer=optimizer,
    scheduler=scheduler,
    device="cuda",
    save_every=1,
    base_path=f"./pretrained/{benchmark}",
    model_name="ResNet34_Teacher"
)
trainer.fit(
    model=teacher,
    epochs=n_epochs,
    train_loader=train_loader,
    test_loader=test_loader
)

with open(f"./results/{benchmark}/teacher_training_history.json", "w") as f:
    f.write(json.dumps(trainer.history, indent=4))

print("\n\n")
print("-" * 100)
print("Training the Student")
print("-" * 100)
optimizer = torch.optim.SGD(
    lr=lr,
    momentum=0.9,
    params=student.parameters()
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_epochs,
    eta_min=1e-4
)
trainer = BaseTrainer(
    criterion=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(
        lr=0.01,
        momentum=0.9,
        params=student.parameters()
    ),
    scheduler=scheduler,
    device="cuda",
    save_every=1,
    base_path=f"./pretrained/{benchmark}",
    model_name="ResNet18_Student"
)
trainer.fit(
    model=student,
    epochs=n_epochs,
    train_loader=train_loader,
    test_loader=test_loader
)

with open(f"./results/{benchmark}/student_training_history.json", "w") as f:
    f.write(json.dumps(trainer.history, indent=4))
