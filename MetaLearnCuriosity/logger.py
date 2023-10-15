import datetime

import wandb


class WBLogger:
    def __init__(self, config, group, tags, notes=None):
        self.config = config
        self.tags = tags
        self.group = group
        self.notes = notes

    def log_episode_return(self, output):
        self.episode_returns = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
        )

        for returns in output["metrics"]["returned_episode_returns"].mean(-1).reshape(-1):
            self.episode_returns.log({"episode_return": returns})
        self.episode_returns.finish()

    def log_byol_losses(self, output):
        self.losses = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
        )

        for loss in range(len(output["rl_loss"][0].mean(-1).reshape(-1))):
            self.losses.log(
                {
                    "byol_loss": output["byol_loss"].mean(-1).reshape(-1)[loss],
                    "encoder_loss": output["encoder_loss"].mean(-1).reshape(-1)[loss],
                    "rl_loss": output["rl_loss"][0].mean(-1).reshape(-1)[loss],
                }
            )
        self.losses.finish()

    def log_rl_losses(self, output):
        self.losses = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
        )
        for loss in range(len(output["rl_loss"][0].mean(-1).reshape(-1))):
            self.losses.log(
                {
                    "rl_loss": output["rl_loss"][0].mean(-1).reshape(-1)[loss],
                }
            )
        self.losses.finish()
