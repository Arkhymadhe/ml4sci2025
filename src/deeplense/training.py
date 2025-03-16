# -*- coding: utf-8 -*-

import torch
import torch.optim as optim

import os
import json

from src.deeplense.loss import SuperResolutionLoss
# Import model here: from src.deeplense.models import

from src.deeplense.dataset import DeepLenseSRDataset, DeepLenseMaskedDataset

##### Ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEED = 42
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)


##### Model architecture
encoder_name = "resnet34"
in_channels = 3
freeze_encoder = True # TODO: Experiment with freezing encoder (original: True)
encoder_weights = "imagenet"
checkpoint = 2 # TODO: Experiment with loading previous model checkpoint (original: None)
smp = False

model_params = dict(
    encoder_name = encoder_name,
    in_channels = in_channels,
    freeze_encoder = freeze_encoder,
    encoder_weights = encoder_weights,
    checkpoint = checkpoint,
    smp = smp,
)

##### Data preparation
num_classes = 10
batch_size = 8 # TODO: Experiment with batch size (original: 16)
num_workers = 0
pin_memory = True
finetuning_dataset = False

data_params = dict(
    num_classes = num_classes,
    batch_size = batch_size,
)

#### Objective function hyperparameters
fourier_loss_weight = .5
amplitude_weight = .5
layers = [8, 17, 25]
progressive_weights = False

##### Training process
epochs = 20 # TODO: Experiment with this (original: 10)
reduce = True
decoder_lr = 1e-4 # TODO: Experiment with this (original: 1e-4)
encoder_lr = 1e-4 # TODO: Experiment with this (original: 1e-4)
weight_decay = 1e-2
gamma = 2. # TODO: Experiment with this (original: 0.)
grad_accum_steps = 1

training_params = dict(
    epochs = epochs,
    weight_decay=weight_decay,
    encoder_lr=encoder_lr,
    decoder_lr=decoder_lr,
    gamma=gamma
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
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory
)

IN, MASK = next(iter(train_dataloader))

print(f"IN shape:", IN.shape)
print(f"MASK shape:", MASK.shape)

print(f"IN dtype:", IN.dtype)
print(f"MASK dtype:", MASK.dtype)


##### Model instantiation

model = load_unet_model(
    encoder_name=encoder_name,
    encoder_weights=encoder_weights,
    freeze_encoder=freeze_encoder,
    num_classes=num_classes,
    in_channels=in_channels,
    checkpoint=checkpoint,
    smp=smp
)

model = model.train()

##### Parameter filtering and grouping

param_groups = []

if not freeze_encoder:
    encoder_params = {
        "params": filter(lambda x: x.requires_grad, model.encoder.parameters()),
        "lr": encoder_lr,
    }
    param_groups.append(encoder_params)

decoder_params = {
    "params":filter(lambda x: x.requires_grad, model.decoder.parameters()),
    "lr": decoder_lr,
}

param_groups.append(decoder_params)

##### Optimizer instantiation
optimizer = optim.AdamW(
    params=param_groups,
    lr=encoder_lr,
    weight_decay=weight_decay
)

##### Objective function instantiation
criterion = SuperResolutionLoss(
    fourier_loss_weight=fourier_loss_weight,
    amplitude_weight=amplitude_weight,
    layers=layers,
    progressive_weights=progressive_weights
)

##### Trainer instantiation
trainer = Trainer(
    model=model,                    # UNet model with pretrained backbone
    criterion=criterion,     # loss function for model convergence
    optimizer=optimizer,      # optimizer for regularization
    epochs=epochs                 # number of epochs for model training
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