"""
This file runs the main training/val loop, etc... using Lightning Trainer
"""
import hydra
from pytorch_lightning import Trainer
from research_seed.mnist.mnist import CoolSystem
import yaml

@hydra.main(config_path="../../conf/config.yml")
def main(cfg):
    # init module
    print(yaml.load(cfg.pretty()))
    logger = hydra.utils.instantiate(cfg.logger)
    logger.log_hyperparams(yaml.load(cfg.pretty()))
    model = CoolSystem(cfg.model)
    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=cfg.trainer.max_nb_epochs,
        gpus=cfg.trainer.gpus,
        nb_gpu_nodes=cfg.trainer.nodes,
        fast_dev_run=True,
        logger=logger,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
