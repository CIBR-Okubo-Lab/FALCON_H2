from datetime import datetime
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import os
import pickle

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import sys
import torch
import wandb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# local modules
from neural_decoder.callbacks import TimerCallback, NanDetectorCallback
from neural_decoder.dataset import MAMLDataModule
from neural_decoder.maml_align import MAMLAligner

torch.set_float32_matmul_precision("medium")

def trainModel(args):

    # set seed
    pl.seed_everything(args["seed"], workers=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args["seed"])
    torch.backends.cudnn.deterministic = True
    
    # output directory
    if 'wandb' in args and args.wandb.enabled:
        args["outputDir"] = os.path.join(args["outputDir"], wandb.run.name)
    else:
        args["outputDir"] = os.path.join(args["outputDir"],f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    # save arguments
    os.makedirs(args["outputDir"], exist_ok=True)
    with open(os.path.join(args["outputDir"], "args"), "wb") as file:
        pickle.dump(args, file)
    
    # load data
    with open(args["datasetPath"], "rb") as handle:
        loadedData = pickle.load(handle)

    # data module
    dm = MAMLDataModule(loadedData, 
                        args["n_task"], 
                        args["k_tr"], 
                        args["k_ts"], 
                        args["k_val_fs"], 
                        args["numWorkers"])

    # wandb logger
    logger = WandbLogger(
        project=args.wandb.setup.project,
        name=f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        save_dir=args["outputDir"],
    )
    
    # model
    model = MAMLAligner(
        pretrain_model=args["pretrain"],
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        dropout=args["dropout"],
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        whiteNoiseSD=args["whiteNoiseSD"], 
        constantOffsetSD=args["constantOffsetSD"],
        bidirectional=args["bidirectional"],
        first_order=args["first_order"],
        inner_lr=args["inner_lr"],
        outer_lr=args["outer_lr"],
        optimizer_type=args["optimizer_type"],
        use_scheduler=args["use_scheduler"],
        l2_decay=args["l2_decay"], 
        momentum=args["momentum"],
        nesterov=args["nesterov"],
        gamma=args["gamma"],
        stepSize=args["stepSize"],
        inner_steps=args["inner_steps"],
        single_layer=args["single_layer"],
        align_hidden=args["align_hidden"]
    )
    
    # checkpoint callback
    checkpointCallback = ModelCheckpoint(filename=args["outputDir"]+"/modelWeights", monitor="val/cer", mode="min", save_top_k=1, every_n_train_steps=None)
    checkpointCallback.FILE_EXTENSION = ""

    # save last model
    lastmodeCallback = ModelCheckpoint(dirpath=args["outputDir"], save_last=True, save_top_k=0)

    # trainer
    trainer = pl.Trainer(
        logger=logger,
        min_steps=1,
        max_steps=args["nSteps"],
        accelerator=args["accelerator"], 
        devices=args["devices"],
        gradient_clip_val=args["gradClipNorm"],
        precision=args["precision"],
        log_every_n_steps=1,
        val_check_interval=50, 
        limit_val_batches=1,
        check_val_every_n_epoch=None,
        callbacks=[checkpointCallback, lastmodeCallback, TimerCallback(), NanDetectorCallback()]
)

    # train
    trainer.fit(model, dm)

@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    # local rank for distributed training
    if 'wandb' in cfg and cfg.wandb.enabled:
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        run = wandb.init(**cfg.wandb.setup,
                            config=wandb_config,
                            name=f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                            sync_tensorboard=False)
    trainModel(cfg)
    if 'wandb' in cfg and cfg.wandb.enabled:
        run.finish()

if __name__ == "__main__":
    main()