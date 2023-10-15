import datetime

import wandb


class WBLogger:
    def __init__(self, config, group, tags, notes=None):

        self.episode_returns = wandb.init(
            project="MetaLearnCuriosity",
            reinit=True,
            config=config,
            group=group,
            tags=tags,
            notes=notes,
        )

        self.losses = wandb.init(
            project="MetaLearnCuriosity",
            reinit=True,
            config=config,
            group=group,
            tags=tags,
            notes=notes,
        )

    def log_episode_return(self, output):

        for returns in output["metrics"]["returned_episode_returns"].mean(-1).reshape(-1):
            self.episode_returns.log({"episode_return": returns})
        self.episode_returns.finish()

    def log_byol_losses(self, output):

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

        for loss in range(len(output["rl_loss"][0].mean(-1).reshape(-1))):
            self.losses.log(
                {
                    "rl_loss": output["rl_loss"][0].mean(-1).reshape(-1)[loss],
                }
            )
        self.losses.finish()
