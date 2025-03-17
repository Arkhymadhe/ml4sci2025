# -*- coding: utf-8 -*-

import torch
import torch.optim as optim

import os
import json

from src.deeplense.loss import SuperResolutionLoss
from src.deeplense.models import UNet, UNetTransformer

from src.deeplense.dataset import DeepLenseSRDataset, DeepLenseMaskedDataset, DeepLenseDiffusionDataset

##### Ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEED = 42
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

##### Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##### Model architecture
in_channels = 3
out_channels = 64
num_blocks = 4

embedding_dim = 512
num_time_steps = 1000

model_params = dict(
    in_channels=in_channels,
    out_channels=out_channels,
    num_blocks=num_blocks,
)

##### Data preparation
num_classes = 10
batch_size = 8  # TODO: Experiment with batch size (original: 16)
num_workers = 0
pin_memory = True
finetuning_dataset = False

data_params = dict(
    num_classes=num_classes,
    batch_size=batch_size,
)

#### Objective function hyperparameters
fourier_loss_weight = 0.5  # Importance of fourier loss
amplitude_weight = 0.5  # Importance of amplitude vs phase losses
layers = [8, 17, 25]  # VGG layers for content loss
progressive_weights = False  # Ramp up VGG layer importance with depth

##### Training process
epochs = 20  # TODO: Experiment with this (original: 10)

decoder_lr = 1e-4  # TODO: Experiment with this (original: 1e-4)
encoder_lr = 1e-4  # TODO: Experiment with this (original: 1e-4)
bottleneck_lr = 1e-4

weight_decay = 1e-2
grad_accum_steps = 1

training_params = dict(
    epochs=epochs,
    weight_decay=weight_decay,
    encoder_lr=encoder_lr,
    decoder_lr=decoder_lr,
    bottleneck_lr=bottleneck_lr,
    grad_accum_steps=grad_accum_steps,
)

final_params = {}
final_params.update(model_params)
final_params.update(data_params)
final_params.update(training_params)

try:
    os.makedirs("../experiments")
except:
    pass

num = len(os.listdir("../experiments")) + 1

checkpoint_dir = f"../experiments/experiment_{num}"
plots_dir = f"{checkpoint_dir}/plots"

try:
    os.makedirs(plots_dir)
except:
    pass

with open(f"{checkpoint_dir}/params.json", "w") as fp:
    json.dump(final_params, fp, indent=4)

##### Generate dataloaders

train_ds = DeepLenseSRDataset(finetune=finetuning_dataset)
train_dataloader = train_ds.to_dataloader(
    batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
)

IN, MASK = next(iter(train_dataloader))

print(f"IN shape:", IN.shape)
print(f"MASK shape:", MASK.shape)

print(f"IN dtype:", IN.dtype)
print(f"MASK dtype:", MASK.dtype)


##### Model instantiation

model = UNet(in_channels=in_channels, out_channels=out_channels, num_blocks=num_blocks)


model = model.train()

##### Parameter filtering and grouping

param_groups = []


encoder_params = {
    "params": filter(lambda x: x.requires_grad, model.encoder.parameters()),
    "lr": encoder_lr,
}

bottleneck_params = {
    "params": filter(lambda x: x.requires_grad, model.bottleneck.parameters()),
    "lr": bottleneck_lr,
}

decoder_params = {
    "params": filter(lambda x: x.requires_grad, model.decoder.parameters()),
    "lr": decoder_lr,
}

param_groups.append(encoder_params)
param_groups.append(bottleneck_params)
param_groups.append(decoder_params)

##### Optimizer instantiation
optimizer = optim.AdamW(params=param_groups, lr=encoder_lr, weight_decay=weight_decay)

##### Objective function instantiation
criterion = SuperResolutionLoss(
    fourier_loss_weight=fourier_loss_weight,
    amplitude_weight=amplitude_weight,
    layers=layers,
    progressive_weights=progressive_weights,
)

##### Trainer instantiation
trainer = Trainer(
    model=model,  # UNet model with pretrained backbone
    criterion=criterion,  # loss function for model convergence
    optimizer=optimizer,  # optimizer for regularization
    epochs=epochs,  # number of epochs for model training
)

##### Model training
num_image_samples = len(train_dataloader)
total_num_iters = num_image_samples * epochs

for epoch in range(1, epochs + 1):
    for i, (x_low, x_high) in enumerate(train_dataloader, 1):
        iter_ = (epoch - 1) * num_image_samples + i
        X_high_hat = model(x_low)
        loss = criterion(X_high_hat, x_high)
        loss.backward()
        if iter_ % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

checkpoint = {
    "model": trainer.model.eval().state_dict(),
    "optimizer": trainer.optimizer.state_dict(),
    "train_losses": trainer.train_losses_,
    "val_losses": trainer.val_losses_,
    "epochs": trainer.epochs,
}

torch.save(checkpoint, f"{checkpoint_dir}/checkpoint.pth")
# torch.save(trainer.val_losses_, f"{checkpoint_dir}/val_losses.pth")
