import os
import torch
import typing as t
from tqdm import tqdm
from src.base_trainer import BaseTrainer
from src.utils import WassersteinDistance


class MeasureMatchingDistillator(BaseTrainer):
    def __init__(
        self,
        teacher,
        criterion: torch.nn.Module,
        measure_matching_criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: t.Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: t.Literal["cuda", "cpu"] = "cuda",
        save_every: int = 1,
        base_path: str = "./",
        model_name: str = "my_model",
        coef: float = 1.0,
        coef_clf: float = 0.0
    ):
        self.teacher = teacher
        self.coef = coef
        self.coef_clf = coef_clf

        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()

        super().__init__(
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            save_every=save_every,
            base_path=base_path,
            model_name=model_name
        )

        if measure_matching_criterion is None:
            self.measure_matching_criterion = WassersteinDistance()
        else:
            self.measure_matching_criterion = measure_matching_criterion

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_task_loss": [],
            "train_distillation_loss": [],
            "test_loss": [],
            "test_acc": [],
            "test_task_loss": [],
            "test_distillation_loss": []
        }

    def __train_epoch(self, model, train_loader):
        model.to(self.device).train()

        running_loss, running_acc = 0.0, 0.0
        running_task_loss, running_distillation_loss = 0.0, 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (x, y) in pbar:
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)

            logits_student, features_student = model(x)
            with torch.no_grad():
                logits_teacher, features_teacher = self.teacher(x)

            # Computes task loss
            task_loss = self.criterion(logits_student, y)
            acc = ((logits_student.argmax(dim=1) == y).sum() / len(y)).item()

            # Computes distillation loss
            distillation_loss = self.measure_matching_criterion(
                features_student, features_teacher, y,
                logits_student, logits_teacher
            )

            # combines the losses
            loss = self.coef_clf * task_loss + self.coef * distillation_loss

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_acc += acc
            running_task_loss += task_loss.item()
            running_distillation_loss += distillation_loss.item()

            _running_loss = running_loss / (batch_idx + 1)
            _running_acc = running_acc / (batch_idx + 1)
            _running_task_loss = running_task_loss / (batch_idx + 1)
            _running_distillation_loss = (
                running_distillation_loss / (batch_idx + 1)
            )
            pbar.set_description(
                f"{batch_idx=}, {_running_loss=}, {_running_acc=},"
                f" {_running_task_loss=}, {_running_distillation_loss=}"
            )

        if self.scheduler:
            self.scheduler.step()  # type: ignore
            new_lr = self.scheduler.get_last_lr()  # type: ignore
            print(f"Scheduler updated LR. {new_lr=}")

        return {
            "loss": _running_loss,
            "task_loss": _running_task_loss,
            "distillation_loss": _running_distillation_loss,
            "acc": _running_acc
        }

    def __eval_epoch(self, model, test_loader):
        model.to(self.device).eval()

        running_loss, running_acc = 0.0, 0.0
        running_task_loss, running_distillation_loss = 0.0, 0.0
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader))
            for batch_idx, (x, y) in pbar:
                x, y = x.to(self.device), y.to(self.device)
                logits_student, features_student = model(x)
                with torch.no_grad():
                    logits_teacher, features_teacher = self.teacher(x)

                # Computes task loss
                task_loss = self.criterion(logits_student, y)
                acc = (
                    (logits_student.argmax(dim=1) == y).sum() / len(y)
                ).item()

                # Computes distillation loss
                distillation_loss = self.measure_matching_criterion(
                    features_student, features_teacher, y,
                    logits_student, logits_teacher
                )

                # combines the losses
                loss = (
                    self.coef_clf * task_loss + self.coef * distillation_loss
                )

                running_loss += loss.item()
                running_acc += acc
                running_task_loss += task_loss.item()
                running_distillation_loss += distillation_loss.item()

                _running_loss = running_loss / (batch_idx + 1)
                _running_acc = running_acc / (batch_idx + 1)
                _running_task_loss = running_task_loss / (batch_idx + 1)
                _running_distillation_loss = (
                    running_distillation_loss / (batch_idx + 1)
                )
                pbar.set_description(
                    f"{batch_idx=}, {_running_loss=}, {_running_acc=},"
                    f" {_running_task_loss=}, {_running_distillation_loss=}"
                )

        return {
            "loss": _running_loss,
            "task_loss": _running_task_loss,
            "distillation_loss": _running_distillation_loss,
            "acc": _running_acc
        }

    def fit(self, student, train_loader, test_loader=None, epochs=1):
        for epoch in range(epochs):
            print("=" * 10 + " Training epoch {:^3} ".format(epoch) + "=" * 10)
            train_results = self.__train_epoch(
                student, train_loader=train_loader)
            print(f"Finished training {epoch=}. {train_results=}")

            self.history["train_loss"].append(train_results["loss"])
            self.history["train_acc"].append(train_results["acc"])
            self.history["train_task_loss"].append(train_results["task_loss"])
            self.history["train_distillation_loss"].append(
                train_results["distillation_loss"]
            )

            if test_loader:
                test_results = self.__eval_epoch(
                    student, test_loader=test_loader)
                print(f"Finished eval {epoch=}. {test_results=}")
                self.history["test_loss"].append(test_results["loss"])
                self.history["test_acc"].append(test_results["acc"])
                self.history["test_task_loss"].append(
                    test_results["task_loss"])
                self.history["test_distillation_loss"].append(
                    test_results["distillation_loss"]
                )

            if self.save_every and epoch % self.save_every == 0:
                torch.save(
                    student.to("cpu").state_dict(),
                    os.path.join(self.base_path, f"{self.model_name}.pth")
                )
