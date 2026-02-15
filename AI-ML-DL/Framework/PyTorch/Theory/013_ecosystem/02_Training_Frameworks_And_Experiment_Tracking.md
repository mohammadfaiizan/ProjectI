# Training Frameworks and Experiment Tracking

## Table of Contents

1. [PyTorch Lightning](#1-pytorch-lightning)
2. [Ray for Distributed Training](#2-ray-for-distributed-training)
3. [Weights and Biases](#3-weights-and-biases)
4. [Hydra Configuration](#4-hydra-configuration)
5. [MLflow](#5-mlflow)

---

## 1. PyTorch Lightning

PyTorch Lightning organizes PyTorch code into reusable components, handling training boilerplate, distributed training, and logging automatically.

### 1.1 LightningModule

The **LightningModule** encapsulates model logic, training step, validation step, and optimizer configuration:

```python
import pytorch_lightning as pl

class SimpleLightningModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}
        }
```

### 1.2 Trainer

The **Trainer** manages the training loop, device placement, and callbacks:

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

checkpoint_callback = ModelCheckpoint(
    dirpath='./checkpoints',
    filename='{epoch}-{val_loss:.2f}',
    monitor='val_loss',
    mode='min',
    save_top_k=3
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = Trainer(
    max_epochs=100,
    callbacks=[checkpoint_callback, early_stop, lr_monitor],
    logger=TensorBoardLogger('tb_logs', name='experiment'),
    accelerator='gpu',
    devices=1
)

trainer.fit(model, datamodule)
trainer.test(model, datamodule)
```

### 1.3 Callbacks

| Callback | Purpose |
|----------|---------|
| ModelCheckpoint | Save best/periodic checkpoints |
| EarlyStopping | Stop when metric plateaus |
| LearningRateMonitor | Log learning rate |
| RichProgressBar | Enhanced progress display |

### 1.4 Logging

```python
self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
self.log_dict({'val_loss': loss, 'val_acc': acc}, on_epoch=True)
```

### 1.5 Data Modules

**LightningDataModule** encapsulates dataset creation and DataLoader setup:

```python
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = ...
        self.val_dataset = ...
        self.test_dataset = ...

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
```

### 1.6 Automatic Optimization

Lightning handles `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()` automatically. For manual control:

```python
def training_step(self, batch, batch_idx):
    opt = self.optimizers()
    opt.zero_grad()
    loss = self.compute_loss(batch)
    self.manual_backward(loss)
    opt.step()
```

---

## 2. Ray for Distributed Training

Ray provides scalable hyperparameter tuning and distributed training for PyTorch.

### 2.1 Ray Train

**TorchTrainer** wraps training for multi-worker execution:

```python
from ray.train import Trainer
from ray.train.torch import TorchTrainer
from ray.air import session

def train_func(config):
    rank = session.get_world_rank()
    world_size = session.get_world_size()

    model = SimpleModel(config["input_size"], config["hidden_size"], config["num_classes"])
    model = session.prepare_model(model)

    dataset = SyntheticDataset(config["dataset_size"], config["input_size"], config["num_classes"])
    indices = list(range(len(dataset)))
    worker_indices = indices[rank::world_size]
    worker_dataset = torch.utils.data.Subset(dataset, worker_indices)

    train_loader = DataLoader(worker_dataset, batch_size=config["batch_size"], shuffle=True)
    train_loader = session.prepare_data_loader(train_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["num_epochs"]):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if rank == 0:
            session.report({"epoch": epoch, "loss": loss.item()})

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={"input_size": 20, "hidden_size": 128, "num_classes": 3, "batch_size": 32, "lr": 0.001, "num_epochs": 10, "dataset_size": 2000},
    scaling_config={"num_workers": 4, "use_gpu": True}
)

result = trainer.fit()
```

### 2.2 Scaling and Fault Tolerance

Ray handles worker failures and provides automatic checkpointing. Use `session.get_checkpoint()` and `session.report(..., checkpoint=...)` for fault-tolerant training.

### 2.3 Ray Tune Hyperparameter Search

```python
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

def train_model(config):
    model = SimpleModel(20, config["hidden_size"], 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    for epoch in range(10):
        ...
        session.report({"val_accuracy": val_acc, "val_loss": val_loss})

config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([16, 32, 64]),
    "hidden_size": tune.choice([32, 64, 128, 256])
}

scheduler = ASHAScheduler(metric="val_accuracy", mode="max", max_t=10, grace_period=3)
reporter = CLIReporter(metric_columns=["val_loss", "val_accuracy"])

result = tune.run(
    train_model,
    config=config,
    num_samples=20,
    scheduler=scheduler,
    progress_reporter=reporter
)
```

---

## 3. Weights and Biases

Weights and Biases (W&B) provides experiment tracking, visualization, and hyperparameter sweeps.

### 3.1 wandb.init

```python
import wandb

wandb.init(
    project="pytorch-experiments",
    entity="your-entity",
    config={"lr": 0.001, "batch_size": 32, "epochs": 100},
    name="experiment-1",
    tags=["resnet", "classification"]
)
```

### 3.2 wandb.log

```python
wandb.log({"train_loss": loss, "train_acc": acc}, step=step)
wandb.log({"val_loss": val_loss, "val_accuracy": val_acc})
wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})
```

### 3.3 wandb.watch

Track model gradients and parameters:

```python
wandb.watch(model, log="all", log_freq=100)
wandb.watch(model, log="gradients", log_freq=100)
```

### 3.4 Hyperparameter Sweeps

```python
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},
        'batch_size': {'values': [16, 32, 64, 128]},
        'hidden_size': {'values': [64, 128, 256, 512]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="my-project")

def train():
    wandb.init()
    config = wandb.config
    model = create_model(hidden_size=config.hidden_size)
    for epoch in range(config.epochs):
        ...
        wandb.log({"val_accuracy": val_acc})

wandb.agent(sweep_id, train, count=20)
```

### 3.5 Artifact Tracking

```python
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pth")
wandb.log_artifact(artifact)
```

---

## 4. Hydra Configuration

Hydra provides hierarchical configuration management with command-line overrides.

### 4.1 Config Groups

```yaml
defaults:
  - model: resnet
  - optimizer: adam
  - _self_

data:
  batch_size: 64
  num_workers: 4

training:
  num_epochs: 100
  early_stopping: true
```

### 4.2 Overrides

```bash
python train.py model=cnn optimizer=sgd
python train.py training.num_epochs=50 data.batch_size=128
```

### 4.3 Multirun

Run multiple configurations in parallel:

```bash
python train.py -m model=resnet,cnn optimizer=adam,sgd
```

### 4.4 Structured Configs

```python
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

@dataclass
class ModelConfig:
    name: str = "resnet"
    num_classes: int = 10
    dropout: float = 0.2

@dataclass
class Config:
    model: ModelConfig
    batch_size: int = 32
    lr: float = 0.001

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    model = create_model(cfg.model)
```

---

## 5. MLflow

MLflow provides experiment tracking, model registry, and model serving for the ML lifecycle.

### 5.1 Experiment Tracking

```python
import mlflow
import mlflow.pytorch

mlflow.set_experiment("pytorch_experiments")

with mlflow.start_run(run_name="run_1"):
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)

    for epoch in range(100):
        ...
        mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
```

### 5.2 Model Registry

```python
mlflow.pytorch.log_model(model, "model", input_example=sample_input)
model_uri = f"runs:/{run.info.run_id}/model"

model_version = mlflow.register_model(model_uri, "my_classifier")
```

### 5.3 Model Serving

```python
loaded_model = mlflow.pytorch.load_model("models:/my_classifier/Production")
predictions = loaded_model(input_tensor)
```

### 5.4 Autolog

```python
mlflow.pytorch.autolog()

with mlflow.start_run():
    model = create_model()
    trainer.fit(model, datamodule)
```

Autolog captures parameters, metrics, and model artifacts automatically.
