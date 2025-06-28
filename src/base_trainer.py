import os
import torch
from tqdm import tqdm
from typing import Dict
from typing import Literal
from typing import Optional
from torch.utils.data import DataLoader


class BaseTrainer:
    def __init__(
        self,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        save_every: Optional[int] = 1,
        base_path: str = "./",
        model_name: str = "my_model"
    ) -> None:
        self.criterion = criterion
        self.optimizer = optimizer
        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=2)
        else:
            self.scheduler = scheduler

        if torch.cuda.is_available:
            self.device = torch.device(device)
        else:
            self.devicee = torch.device("cpu")
            print("WARNING: CUDA is not available. Using CPU.")
        self.save_every = save_every
        self.base_path = base_path
        self.model_name = model_name

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "distillation_losses": {
                "e_w2": [],
                "e_cond_w2": [],
                "e_mmd_L": [],
                "e_mmd_G": [],
                "g_w2": [],
                "g_cond_w2": [],
                "g_kl": [],
                "g_cond_kl": [],
            }
        }

    def fit(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 1
    ) -> None:
        for epoch in range(epochs):
            print("=" * 10 + " Training epoch {:^3} ".format(epoch) + "=" * 10)
            train_results = self.__train_epoch(
                model, train_loader=train_loader)
            print(f"Finished training {epoch=}. {train_results=}")
            self.history["train_loss"].append(train_results["loss"])
            self.history["train_acc"].append(train_results["acc"])
            if test_loader:
                test_results = self.__eval_epoch(
                    model, test_loader=test_loader)
                print(f"Finished eval {epoch=}. {test_results=}")
                self.history["test_loss"].append(test_results["loss"])
                self.history["test_acc"].append(test_results["acc"])
            if self.save_every is not None and epoch % self.save_every == 0:
                torch.save(
                    model.to("cpu").state_dict(),
                    os.path.join(self.base_path, f"{self.model_name}.pth")
                )

    def __train_epoch(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        model.to(self.device).train()

        running_loss, running_acc = 0.0, 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (x, y) in pbar:
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            y_hat, _ = model(x)
            loss = self.criterion(y_hat, y)
            acc = ((y_hat.argmax(dim=1) == y) / len(y)).sum().item()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_acc += acc

            _running_loss = running_loss / (batch_idx + 1)
            _running_acc = running_acc / (batch_idx + 1)
            pbar.set_description(
                f"{batch_idx=}, {_running_loss=}, {_running_acc=}"
            )

        if self.scheduler:
            self.scheduler.step()  # type: ignore
            # new_lr = self.scheduler.get_last_lr()
            # print(f"Scheduler updated LR. {new_lr=}")

        return {
            "loss": _running_loss,
            "acc": _running_acc
        }

    def __eval_epoch(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        model.to(self.device).eval()

        with torch.no_grad():
            running_loss, running_acc = 0.0, 0.0
            pbar = tqdm(enumerate(test_loader), total=len(test_loader))
            for batch_idx, (x, y) in pbar:
                self.optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                y_hat, _ = model(x)
                loss = self.criterion(y_hat, target=y)
                acc = ((y_hat.argmax(dim=1) == y) / len(y)).sum().item()

                running_loss += loss.item()
                running_acc += acc

                _running_loss = running_loss / (batch_idx + 1)
                _running_acc = running_acc / (batch_idx + 1)
                pbar.set_description(
                    f"{batch_idx=}, {_running_loss=}, {_running_acc=}"
                )

        return {
            "loss": _running_loss,
            "acc": _running_acc
        }
