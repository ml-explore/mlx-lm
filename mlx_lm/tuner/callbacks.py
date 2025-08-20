# Copyright © 2024 Apple Inc.

import os

try:
    import wandb
except ImportError:
    wandb = None

try:
    import swanlab
except ImportError:
    swanlab = None


class TrainingCallback:

    def on_train_loss_report(self, train_info: dict):
        """Called to report training loss at specified intervals."""
        pass

    def on_val_loss_report(self, val_info: dict):
        """Called to report validation loss at specified intervals or the beginning."""
        pass


class WandBCallback(TrainingCallback):
    def __init__(
        self,
        project_name: str,
        log_dir: str,
        config: dict,
        wrapped_callback: TrainingCallback = None,
    ):
        if wandb is None:
            raise ImportError(
                "wandb is not installed. please install wandb via: pip install wandb",
            )
        self.wrapped_callback = wrapped_callback
        wandb.init(
            project=project_name,
            name=os.path.basename(log_dir),
            dir=log_dir,
            config=config,
        )

    def _convert_to_serializable(self, data: dict) -> dict:
        return {k: v.tolist() if hasattr(v, "tolist") else v for k, v in data.items()}

    def on_train_loss_report(self, train_info: dict):
        wandb.log(
            self._convert_to_serializable(train_info), step=train_info.get("iteration")
        )
        if self.wrapped_callback:
            self.wrapped_callback.on_train_loss_report(train_info)

    def on_val_loss_report(self, val_info: dict):
        wandb.log(
            self._convert_to_serializable(val_info), step=val_info.get("iteration")
        )
        if self.wrapped_callback:
            self.wrapped_callback.on_val_loss_report(val_info)


class SwanLabCallback(TrainingCallback):
    def __init__(
        self,
        project_name: str,
        log_dir: str,
        config: dict,
        wrapped_callback: TrainingCallback = None,
    ):
        if swanlab is None:
            raise ImportError(
                "swanlab is not installed. please install swanlab via: pip install swanlab",
            )
        self.wrapped_callback = wrapped_callback
        swanlab.init(
            project=project_name,
            experiment_name=os.path.basename(log_dir),
            logdir=os.path.join(log_dir, "swanlog"),
            config=config,
        )

    def _convert_to_serializable(self, data: dict) -> dict:
        return {k: v.tolist() if hasattr(v, "tolist") else v for k, v in data.items()}

    def on_train_loss_report(self, train_info: dict):
        swanlab.log(
            self._convert_to_serializable(train_info), step=train_info.get("iteration")
        )
        if self.wrapped_callback:
            self.wrapped_callback.on_train_loss_report(train_info)

    def on_val_loss_report(self, val_info: dict):
        swanlab.log(
            self._convert_to_serializable(val_info), step=val_info.get("iteration")
        )
        if self.wrapped_callback:
            self.wrapped_callback.on_val_loss_report(val_info)


SUPPORT_CALLBACK = {
    "wandb": WandBCallback,
    "swanlab": SwanLabCallback,
}


def get_reporting_callbacks(
    report_to: str = None,
    project_name: str = None,
    log_dir: str = None,
    config: str = None,
):
    if report_to is None or report_to == "":
        return None
    report_to = [item.strip().lower() for item in report_to.split(",") if item.strip()]
    training_callback = None
    for callback in report_to:
        try:
            training_callback = SUPPORT_CALLBACK[callback](
                project_name=project_name,
                log_dir=log_dir,
                config=config,
                wrapped_callback=training_callback,
            )
        except KeyError as e:
            raise ValueError(
                f"{callback} callback doesn't exist "
                f"choose from {', '.join(SUPPORT_CALLBACK.keys())}"
            ) from e

    return training_callback
