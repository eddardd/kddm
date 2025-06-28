import os
import json
import time
import torch
import argparse

from src.data_utils import get_cifar10_dataloaders
from src.data_utils import get_cifar100_dataloaders
from src.data_utils import get_mnist_dataloaders
from src.data_utils import get_svhn_dataloaders
from src.models import ResNet
from src.models import MultilayerPerceptron
from src.utils import WassersteinDistance
from src.utils import KullbackLeiblerDivergence
from src.utils import MaximumMeanDiscrepancy
from src.utils import GromovWassersteinDistance
from src.utils import JointWassersteinDistance
from src.feature_distillation import MeasureMatchingDistillator


parser = argparse.ArgumentParser(
    prog='KD2M',
    description=(
        'Trains a Knowledge Distillation model using Wasserstein distances'
    )
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
    "--coef",
    type=float,
    default=1e-4
)
parser.add_argument(
    "--measure_modeling",
    type=str,
    default="empirical"
)
parser.add_argument(
    "--distance",
    type=str,
    default="w2"
)
parser.add_argument(
    "--benchmark",
    type=str,
    default="mnist"
)
args = parser.parse_args()

torch.manual_seed(42)
device = torch.device('cuda')
batch_size = args.batch_size
lr = args.lr
n_epochs = args.n_epochs
coef = args.coef
run_params = {"lr": lr, "n_epochs": n_epochs, "coef": coef}
benchmark = args.benchmark
run_ts = time.time()
distance = args.distance
measure_modeling = args.measure_modeling
model_path = f"./pretrained/{benchmark}"
model_name = f"{run_ts}_{distance}_{measure_modeling}_feat_Student_ResNet18"
results_path = f"./results/{benchmark}"
history_name = (
    f"{run_ts}_{distance}_{measure_modeling}"
    "_feat_student_training_history.json"
)

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
else:
    raise ValueError(f"Unknown benchmark: {benchmark}")

# Loads teacher
teacher.load_state_dict(
    torch.load(
        os.path.join(model_path, "ResNet34_Teacher.pth"),
        weights_only=True
    )
)
teacher = teacher.eval()
teacher = teacher.to(device)

# Loads student
student.load_state_dict(
    torch.load(
        os.path.join(model_path, "student_weights_0.pth"),
        weights_only=True
    )
)
student = student.to(device)

# distillation loss
if distance == "w2":
    measure_matching_criterion = WassersteinDistance(
        measure_modeling=measure_modeling)
if distance == "jw2":
    measure_matching_criterion = JointWassersteinDistance()
elif distance == "gw2":
    measure_matching_criterion = GromovWassersteinDistance()
elif distance == "kl":
    measure_matching_criterion = KullbackLeiblerDivergence(conditional=False)
elif distance == "mmd-l":
    measure_matching_criterion = MaximumMeanDiscrepancy(kernel='linear')
elif distance == "mmd-g":
    measure_matching_criterion = MaximumMeanDiscrepancy(kernel='rbf')

# Optimizer
optimizer = torch.optim.SGD(
    lr=lr,
    momentum=0.9,
    params=student.parameters()
)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_epochs,
    eta_min=1e-4
)

# Trains the student
trainer = MeasureMatchingDistillator(
    teacher=teacher,
    criterion=torch.nn.CrossEntropyLoss(),
    measure_matching_criterion=measure_matching_criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device="cuda",
    save_every=1,
    base_path=model_path,
    model_name=model_name,
    coef=coef,
    coef_clf=1.0
)
trainer.fit(student, train_loader, test_loader, epochs=n_epochs)

trainer.history.update({"params": run_params})  # type: ignore
with open(os.path.join(results_path, history_name), "w") as f:
    f.write(json.dumps(trainer.history, indent=4))
