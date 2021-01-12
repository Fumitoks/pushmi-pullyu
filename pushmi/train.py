from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from .model import TestMnist

def train_model(config, deterministic=False, constant_split_val=False):
    model = TestMnist(config)
    model.constrant_split_val = constant_split_val
    model_save_dir = config.dirs.saved_models
    model_name = model.name
    # saving checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath = model_save_dir + model_name,
        filename = 'model',
        save_top_k = 1,
        verbose = True,
        monitor = 'val_loss',
        mode = 'min'
    )
    # weight&biases logger used by trainer
    wandb_dir = config.dirs.root + 'wandb/'
    wandb_logger = WandbLogger(
        name = model_name,
        project = 'pushmi',
        dir = wandb_dir
    )
    trainer = pl.Trainer(
        gpus = 1,
        progress_bar_refresh_rate = 20, 
        checkpoint_callback = checkpoint_callback, 
        logger = wandb_logger,
        deterministic = deterministic,
        **config.train.__dict__
        ) #, gradient_clip_val = 0.1)
    trainer.fit(model)
    return model

def train_n_models(num_models, config, deterministic=False, constant_split_val = False, seed_number = 666):
    models = []
    #if use_one_seed:
    #    #print('Seeding')
    #    pl.seed_everything(seed_number)
    if num_models > 1:
        constant_split_val = True
    else:
        constant_split_val = False
    for i in range(num_models):
        model = train_model(config, deterministic=deterministic)
        model.eval()
        models.append(model)
    return models