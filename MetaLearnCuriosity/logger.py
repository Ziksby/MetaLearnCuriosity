import datetime

import jax
import jax.numpy as jnp

import wandb


class WBLogger:
    def __init__(self, config, group, tags, notes=None):
        self.config = config
        self.tags = tags
        self.group = group
        self.notes = notes

    def log_episode_return(self, output, num_seeds):
        self.episode_returns = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
        )

        if num_seeds > 1:
            outs_avg = jnp.mean(output["metrics"]["returned_episode_returns"], axis=0)
            for returns in outs_avg.mean(-1).reshape(-1):
                self.episode_returns.log(
                    {f"episode_return_{self.config['ENV_NAME']}_seeds": returns}
                )
            self.episode_returns.finish()

        else:

            for returns in output["metrics"]["returned_episode_returns"].mean(-1).reshape(-1):
                self.episode_returns.log({f"episode_return_{self.config['ENV_NAME']}": returns})
            self.episode_returns.finish()

    def log_byol_losses(self, output, num_seeds):
        self.losses = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
        )

        if num_seeds > 1:
            byol_avg = jnp.mean(output["byol_loss"], axis=0)
            rl_avg = jnp.mean(output["rl_loss"][0], axis=0)
            encoder_avg = jnp.mean(output["encoder_loss"], axis=0)
            for loss in range(len(rl_avg)):
                self.losses.log(
                    {
                        f"byol_loss_{self.config['ENV_NAME']}_seeds": byol_avg.mean(-1).reshape(-1)[
                            loss
                        ],
                        f"encoder_loss_{self.config['ENV_NAME']}_seeds": encoder_avg.mean(
                            -1
                        ).reshape(-1)[loss],
                        f"rl_loss_{self.config['ENV_NAME']}_seeds": rl_avg.mean(-1).reshape(-1)[
                            loss
                        ],
                    }
                )
            self.losses.finish()
        else:

            for loss in range(len(output["rl_loss"][0].mean(-1).reshape(-1))):
                self.losses.log(
                    {
                        f"byol_loss_{self.config['ENV_NAME']}": output["byol_loss"]
                        .mean(-1)
                        .reshape(-1)[loss],
                        f"encoder_loss_{self.config['ENV_NAME']}": output["encoder_loss"]
                        .mean(-1)
                        .reshape(-1)[loss],
                        f"rl_loss_{self.config['ENV_NAME']}": output["rl_loss"][0]
                        .mean(-1)
                        .reshape(-1)[loss],
                    }
                )
            self.losses.finish()

    def log_rl_losses(self, output, num_seeds):
        self.losses = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
        )
        if num_seeds > 1:
            rl_avg = jnp.mean(output["rl_loss"][0], axis=0)

            for loss in range(len(rl_avg)):
                self.losses.log(
                    {
                        f"rl_loss_{self.config['ENV_NAME']}_seeds": rl_avg.mean(-1).reshape(-1)[
                            loss
                        ],
                    }
                )
            self.losses.finish()
        else:

            for loss in range(len(output["rl_loss"][0].mean(-1).reshape(-1))):
                self.losses.log(
                    {
                        f"rl_loss_{self.config['ENV_NAME']}": output["rl_loss"][0]
                        .mean(-1)
                        .reshape(-1)[loss]
                    }
                )
            self.losses.finish()
