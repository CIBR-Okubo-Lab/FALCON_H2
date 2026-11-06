from datetime import datetime
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import os
import pickle

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

import sys
import torch
import wandb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# local modules
from neural_decoder.callbacks import TimerCallback
from neural_decoder.dataset import SpeechDataModule
from neural_decoder.joint_align import JointAligner

torch.set_float32_matmul_precision("medium")

def trainModel(args):

    # set seed
    pl.seed_everything(args["seed"], workers=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args["seed"])
    torch.backends.cudnn.deterministic = True
    
    # output directory
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if 'wandb' in args and args.wandb.enabled and local_rank == 0:
        args["outputDir"] = os.path.join(args["outputDir"], wandb.run.name)
    else:
        args["outputDir"] = os.path.join(args["outputDir"],f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    
    if local_rank == 0: 
        os.makedirs(args["outputDir"], exist_ok=True)
        with open(os.path.join(args["outputDir"], "args"), "wb") as file:
            pickle.dump(args, file)
    
    # load data
    with open(args["datasetPath"], "rb") as handle:
        loadedData = pickle.load(handle)

    # data module
    dm = SpeechDataModule(loadedData, args["batchSize"], args["numWorkers"])

    # tensorboard logger
    logger = TensorBoardLogger(args["outputDir"], name="torch_dist_v0")
    
    # model
    model = JointAligner(
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
        outer_lr=args["outer_lr"],
        optimizer_type=args["optimizer_type"],
        use_scheduler=args["use_scheduler"],
        l2_decay=args["l2_decay"], 
        momentum=args["momentum"],
        nesterov=args["nesterov"],
        gamma=args["gamma"],
        stepSize=args["stepSize"],
        single_layer=args["single_layer"],
        align_hidden=args["align_hidden"]
    )
    
    # checkpoint callback
    checkpointCallback = ModelCheckpoint(filename=args["outputDir"]+"/modelWeights", monitor="val/ser", mode="min", save_top_k=1, every_n_train_steps=None)
    checkpointCallback.FILE_EXTENSION = ""


    # trainer
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=logger,
        min_steps=1,
        max_steps=args["nSteps"],
        accelerator=args["accelerator"],
        gradient_clip_val=args["gradClipNorm"],
        devices=args["devices"],
        precision=args["precision"],
        num_nodes=1,
        log_every_n_steps=1, 
        val_check_interval=100,
        check_val_every_n_epoch=None, 
        callbacks=[checkpointCallback, TimerCallback()]
    )

    # train
    trainer.fit(model, dm)

@hydra.main(version_base="1.1", config_path="conf", config_name="align_config")
def main(cfg):
    # local rank for distributed training
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if 'wandb' in cfg and cfg.wandb.enabled and local_rank == 0:
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        run = wandb.init(**cfg.wandb.setup,
                            config=wandb_config,
                            name=f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                            sync_tensorboard=True)
    # If we use multi-gpu, we need to define the batch size per gpu
    if cfg.devices == -1:
        nGPU = torch.cuda.device_count()
        cfg.batchSize = int(cfg.batchSize / nGPU)
    trainModel(cfg)
    if 'wandb' in cfg and cfg.wandb.enabled and local_rank == 0:
        run.finish()

if __name__ == "__main__":
    main()