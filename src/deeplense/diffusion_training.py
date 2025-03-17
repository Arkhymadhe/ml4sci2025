# -*- coding: utf-8 -*-

import torch
import torch.optim as optim

import gc
import os
import json

from src.deeplense.loss import DiffusionLoss
from src.deeplense.models import UNet, UNetTransformer

from src.deeplense.dataset import DeepLenseDiffusionDataset, DiffusionHelper

from loguru import logger

logger.add("app.log")

##### Ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEED = 42
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

##### Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##### Model architecture
in_channels = 1
out_channels = 64
num_blocks = 4
input_size = 160

embedding_dim = 128
num_time_steps = 500

model_params = dict(
    in_channels=in_channels,
    out_channels=out_channels,
    num_blocks=num_blocks,
    embedding_dim=embedding_dim,
    num_time_steps=num_time_steps,
)

##### Data preparation
num_classes = 10
batch_size = 8  # TODO: Experiment with batch size (original: 16)
num_workers = 0
pin_memory = True
min_beta = 1e-4
max_beta = 2e-2

data_params = dict(
    num_classes=num_classes,
    batch_size=batch_size,
    min_beta=min_beta,
    max_beta=max_beta,
    num_time_steps=num_time_steps
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
grad_accum_steps = 64

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

BASE_DIR = "../../artefacts/deeplense/diffusion/experiments"

try:
    os.makedirs(BASE_DIR)
except:
    pass

num = len(os.listdir(BASE_DIR)) + 1

checkpoint_dir = f"{BASE_DIR}/experiment_{num}"
plots_dir = f"{checkpoint_dir}/plots"

try:
    os.makedirs(plots_dir)
except:
    pass

with open(f"{checkpoint_dir}/params.json", "w") as fp:
    json.dump(final_params, fp, indent=4)

logger.info(f"Events being logged at {BASE_DIR}\n")

##### Generate dataloaders
diffusion_helper = DiffusionHelper(
    max_beta=max_beta,
    min_beta=min_beta,
    num_time_steps=num_time_steps
)

train_ds = DeepLenseDiffusionDataset(
    input_size=input_size,
    diffusion_helper=diffusion_helper
)
train_dl = train_ds.to_dataloader(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory
)

x, x_t, t, noise = next(iter(train_dl))

print(f"Images shape:", x.shape)
print(f"Corrupted images shape:", x_t.shape)
print(f"Time steps shape:", t.shape)
print(f"Noise shape:", noise.shape)


##### Model instantiation
model = UNetTransformer(
    in_channels=in_channels,
    out_channels=out_channels,
    num_blocks=num_blocks,
    num_timesteps=num_time_steps,
    embedding_dim=embedding_dim
).to(device)

#### Compile model with torch.compile
model = torch.compile(model, mode="max-autotune")

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
criterion = DiffusionLoss()

# ##### Trainer instantiation
# trainer = Trainer(
#     model=model,  # UNet model with pretrained backbone
#     criterion=criterion,  # loss function for model convergence
#     optimizer=optimizer,  # optimizer for regularization
#     epochs=epochs,  # number of epochs for model training
# )

##### Model training
num_image_samples = len(train_ds) / batch_size

if int(num_image_samples) != num_image_samples:
    num_image_samples = int(num_image_samples) + 1

total_num_iters = num_image_samples * epochs

train_losses_ = torch.zeros(epochs)
val_losses_ = torch.zeros(epochs)

iter_ = 0
epoch_loss_ = 0.
scaler = torch.amp.GradScaler(device=str(device))

for epoch in range(1, epochs + 1):
    epoch_loss = 0.

    for i, (x, x_t, t, noise) in enumerate(iter(train_dl), 1):
        # Calculate training iteration
        iter_ = (epoch - 1) * num_image_samples + i

        # Move data to device and make predictions
        x_t = x_t.to(device)
        t = t.to(device).squeeze(-1)
        noise = noise.to(device)

        with torch.amp.autocast(enabled=True, device_type=str(device)):
            noise_hat = torch.tanh(model(x_t, t))

            # Calculate loss and back-propagate
            # Use gradient accumulation if required.
            loss = criterion(noise_hat, noise) / grad_accum_steps

        scaler.scale(loss).backward()

        del x, x_t, t, noise
        gc.collect()

        if iter_ % grad_accum_steps == 0 or iter_ == total_num_iters:
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            logger.info(f"Iteration ({int(iter_)}/{int(total_num_iters)}); Loss: {epoch_loss_:.4f}")

            torch.clear_autocast_cache()
            torch.cuda.empty_cache()

            checkpoint = {
                "model": model.eval().state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_losses": train_losses_,
                "val_losses": val_losses_,
                "iterations_trained": iter_,
                "batch_size": batch_size,
                "total_iterations": total_num_iters,
                "epochs": epochs,
            }

            torch.save(checkpoint, f"{checkpoint_dir}/iteration_{int(iter_)}_checkpoint.pt")

        gc.collect()

        # Update loss values
        epoch_loss += loss.item()
        epoch_loss_ = epoch_loss / iter_

    epoch_loss = epoch_loss / num_image_samples
    logger.info(f"Epoch {epoch} complete! Loss: {epoch_loss: .4f}")

    train_losses_[epoch - 1] = epoch_loss

checkpoint = {
    "model": model.eval().state_dict(),
    "optimizer": optimizer.state_dict(),
    "train_losses": train_losses_,
    "val_losses": val_losses_,
    "iterations_trained": iter_,
    "batch_size": batch_size,
    "total_iterations": total_num_iters,
    "epochs": epochs,
}

torch.save(checkpoint, f"{checkpoint_dir}/final_checkpoint.pt")
