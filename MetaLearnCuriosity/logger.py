import jax.numpy as jnp
import wandb
from flax.jax_utils import unreplicate


class WBLogger:
    def __init__(self, config, group, tags, name, notes=None):
        self.config = config
        self.tags = tags
        self.group = group
        self.notes = notes
        self.name = name

    def log_episode_return(self, output, num_seeds):
        self.episode_returns = wandb.init(
            project="MetaLearnCuriosity",
            name=f"{self.name}_epi_ret",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
        )

        if num_seeds > 1:
            outs_avg = jnp.mean(output["metrics"]["returned_episode_returns"], axis=0)
            for returns in outs_avg.mean(-1).reshape(-1):
                self.episode_returns.log(
                    {f"episode_return_{self.config['ENV_NAME']}_{num_seeds}_seeds": returns}
                )
            self.episode_returns.finish()
        else:
            for returns in output["metrics"]["returned_episode_returns"].mean(-1).reshape(-1):
                self.episode_returns.log({f"episode_return_{self.config['ENV_NAME']}": returns})
            self.episode_returns.finish()

    def log_int_rewards(self, output, num_seeds):
        self.episode_int_rewards = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
            name=f"{self.name}_int_rew",
        )

        if num_seeds > 1:
            outs_avg = jnp.mean(output["int_reward"], axis=0)
            for returns in outs_avg.mean(-1).reshape(-1):
                self.episode_int_rewards.log(
                    {f"int_rewards_{self.config['ENV_NAME']}_{num_seeds}_seeds": returns}
                )
            self.episode_int_rewards.finish()
        else:
            for returns in output["int_reward"].mean(-1).reshape(-1):
                self.episode_int_rewards.log({f"int_rewards_{self.config['ENV_NAME']}": returns})
            self.episode_int_rewards.finish()

    def log_norm_int_rewards(self, output, num_seeds):
        self.episode_norm_int_rewards = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
            name=f"{self.name}_norm_int_rew",
        )

        if num_seeds > 1:
            outs_avg = jnp.mean(output["norm_int_reward"], axis=0)
            for returns in outs_avg.mean(-1).reshape(-1):
                self.episode_norm_int_rewards.log(
                    {f"norm_int_rewards_{self.config['ENV_NAME']}_{num_seeds}_seeds": returns}
                )
            self.episode_norm_int_rewards.finish()
        else:
            for returns in output["norm_int_reward"].mean(-1).reshape(-1):
                self.episode_norm_int_rewards.log(
                    {f"norm_int_rewards_{self.config['ENV_NAME']}": returns}
                )
            self.episode_norm_int_rewards.finish()

    def log_byol_losses(self, output, num_seeds):
        self.losses = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
            name=f"{self.name}_byol_loss",
        )

        if num_seeds > 1:
            byol_avg = jnp.mean(output["byol_loss"], axis=0)
            encoder_avg = jnp.mean(output["encoder_loss"], axis=0)
            for loss in range(len(byol_avg.mean(-1).mean(-1))):
                self.losses.log(
                    {
                        f"byol_loss_{self.config['ENV_NAME']}_{num_seeds}_seeds": byol_avg.mean(
                            -1
                        ).mean(-1)[loss],
                        f"encoder_loss_{self.config['ENV_NAME']}_{num_seeds}_seeds": encoder_avg.mean(
                            -1
                        ).mean(
                            -1
                        )[
                            loss
                        ],
                    }
                )
            self.losses.finish()
        else:
            for loss in range(len(output["byol_loss"].mean(-1).mean(-1))):
                self.losses.log(
                    {
                        f"byol_loss_{self.config['ENV_NAME']}": output["byol_loss"]
                        .mean(-1)
                        .mean(-1)[loss],
                        f"encoder_loss_{self.config['ENV_NAME']}": output["encoder_loss"]
                        .mean(-1)
                        .mean(-1)[loss],
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
            name=f"{self.name}_rl_loss",
        )

        if num_seeds > 1:
            rl_total_avg = jnp.mean(output["rl_total_loss"], axis=0)
            rl_value_avg = jnp.mean(output["rl_value_loss"], axis=0)
            rl_actor_avg = jnp.mean(output["rl_actor_loss"], axis=0)
            rl_entrophy_avg = jnp.mean(output["rl_entrophy_loss"], axis=0)
            for loss in range(len(rl_total_avg.mean(-1).mean(-1))):
                self.losses.log(
                    {
                        f"rl_total_loss_{self.config['ENV_NAME']}_{num_seeds}_seeds": rl_total_avg.mean(
                            -1
                        ).mean(
                            -1
                        )[
                            loss
                        ],
                        f"rl_value_loss_{self.config['ENV_NAME']}_{num_seeds}_seeds": rl_value_avg.mean(
                            1
                        ).mean(
                            -1
                        )[
                            loss
                        ],
                        f"rl_actor_loss_{self.config['ENV_NAME']}_{num_seeds}_seeds": rl_actor_avg.mean(
                            1
                        ).mean(
                            -1
                        )[
                            loss
                        ],
                        f"rl_entrophy_loss_{self.config['ENV_NAME']}_{num_seeds}_seeds": rl_entrophy_avg.mean(
                            1
                        ).mean(
                            -1
                        )[
                            loss
                        ],
                    }
                )
            self.losses.finish()
        else:
            for loss in range(len(output["rl_total_loss"].mean(-1).mean(-1))):
                self.losses.log(
                    {
                        f"rl_total_loss_{self.config['ENV_NAME']}": output["rl_total_loss"]
                        .mean(-1)
                        .mean(-1)[loss],
                        f"rl_value_loss_{self.config['ENV_NAME']}": output["rl_value_loss"]
                        .mean(-1)
                        .mean(-1)[loss],
                        f"rl_actor_loss_{self.config['ENV_NAME']}": output["rl_actor_loss"]
                        .mean(-1)
                        .mean(-1)[loss],
                        f"rl_entrophy_loss_{self.config['ENV_NAME']}": output["rl_entrophy_loss"]
                        .mean(-1)
                        .mean(-1)[loss],
                    }
                )
            self.losses.finish()

    def log_rnd_losses(self, output, num_seeds):
        self.losses = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
            name=f"{self.name}_rnd_loss",
        )

        if num_seeds > 1:
            rnd_avg = jnp.mean(output["rnd_loss"], axis=0)
            for loss in range(len(rnd_avg.mean(-1).mean(-1))):
                self.losses.log(
                    {
                        f"rnd_loss_{self.config['ENV_NAME']}_{num_seeds}_seeds": rnd_avg.mean(
                            -1
                        ).mean(-1)[loss],
                    }
                )
            self.losses.finish()
        else:
            for loss in range(len(output["rnd_loss"].mean(-1).mean(-1))):
                self.losses.log(
                    {
                        f"rnd_loss_{self.config['ENV_NAME']}": output["rnd_loss"]
                        .mean(-1)
                        .mean(-1)[loss],
                    }
                )
            self.losses.finish()

    def log_fast_losses(self, output, num_seeds):
        self.losses = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
            name=f"{self.name}_fast_loss",
        )

        if num_seeds > 1:
            fast_avg = jnp.mean(output["fast_loss"], axis=0)
            for loss in range(len(fast_avg.mean(-1).mean(-1))):
                self.losses.log(
                    {
                        f"fast_loss_{self.config['ENV_NAME']}_{num_seeds}_seeds": fast_avg.mean(
                            -1
                        ).mean(-1)[loss],
                    }
                )
            self.losses.finish()
        else:
            for loss in range(len(output["fast_loss"].mean(-1).mean(-1))):
                self.losses.log(
                    {
                        f"fast_loss_{self.config['ENV_NAME']}": output["fast_loss"]
                        .mean(-1)
                        .mean(-1)[loss],
                    }
                )
            self.losses.finish()

    def log_ccim_losses(self, output, num_seeds):
        self.losses = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
            name=f"{self.name}_ccim_loss",
        )

        if num_seeds > 1:
            for_avg = jnp.mean(output["forward_loss"], axis=0)
            back_avg = jnp.mean(output["backward_loss"], axis=0)
            for loss in range(len(for_avg.mean(-1).mean(-1))):
                self.losses.log(
                    {
                        f"forward_loss_{self.config['ENV_NAME']}_{num_seeds}_seeds": for_avg.mean(
                            -1
                        ).mean(-1)[loss],
                        f"backward_loss_{self.config['ENV_NAME']}_{num_seeds}_seeds": back_avg.mean(
                            -1
                        ).mean(-1)[loss],
                    }
                )
            self.losses.finish()
        else:
            for loss in range(len(output["forward_loss"].mean(-1).mean(-1))):
                self.losses.log(
                    {
                        f"forward_loss_{self.config['ENV_NAME']}": output["forward_loss"]
                        .mean(-1)
                        .mean(-1)[loss],
                        f"backward_loss_{self.config['ENV_NAME']}": output["backward_loss"]
                        .mean(-1)
                        .mean(-1)[loss],
                    }
                )
            self.losses.finish()

    def log_int_value_losses(self, output, num_seeds):
        self.losses = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
            name=f"{self.name}_int_value_loss",
        )

        if num_seeds > 1:
            int_value_avg = jnp.mean(output["rl_int_value_loss"], axis=0)
            for loss in range(len(int_value_avg.mean(-1).mean(-1))):
                self.losses.log(
                    {
                        f"int_value_loss_{self.config['ENV_NAME']}_{num_seeds}_seeds": int_value_avg.mean(
                            -1
                        ).mean(
                            -1
                        )[
                            loss
                        ]
                    }
                )
            self.losses.finish()
        else:
            for loss in range(len(output["rl_int_value_loss"].mean(-1).mean(-1))):
                self.losses.log(
                    {
                        f"int_value_loss_{self.config['ENV_NAME']}": output["rl_int_value_loss"]
                        .mean(-1)
                        .mean(-1)[loss],
                    }
                )
            self.losses.finish()

    def log_int_lambdas(self, output, num_seeds):
        self.int_lambdas = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
            name=f"{self.name}_int_lambdas",
        )

        if num_seeds > 1:
            int_value_avg = jnp.mean(output["int_lambdas"], axis=0)
            for int_lambda in range(len(int_value_avg.mean(-1).reshape(-1))):
                self.int_lambdas.log(
                    {
                        f"int_lambdas_{self.config['ENV_NAME']}_{num_seeds}_seeds": int_value_avg.mean(
                            -1
                        ).reshape(
                            -1
                        )[
                            int_lambda
                        ]
                    }
                )
            self.int_lambdas.finish()
        else:
            for int_lambda in range(len(output["int_lambdas"].mean(-1).reshape(-1))):
                self.int_lambdas.log(
                    {
                        f"int_lambdas_{self.config['ENV_NAME']}": output["int_lambdas"]
                        .mean(-1)
                        .reshape(-1)[int_lambda],
                    }
                )
            self.int_lambdas.finish()

    def log_total_reward(self, output, num_seeds):
        self.total_reward = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
            name=f"{self.name}_total_reward",
        )

        if num_seeds > 1:
            total_reward_avg = jnp.mean(output["total_reward"], axis=0)
            for total_rew in range(len(total_reward_avg.mean(-1).reshape(-1))):
                self.total_reward.log(
                    {
                        f"total_reward_{self.config['ENV_NAME']}_{num_seeds}_seeds": total_reward_avg.mean(
                            -1
                        ).reshape(
                            -1
                        )[
                            total_rew
                        ]
                    }
                )
            self.total_reward.finish()
        else:
            for total_rew in range(len(output["total_reward"].mean(-1).reshape(-1))):
                self.total_reward.log(
                    {
                        f"total_reward_{self.config['ENV_NAME']}": output["total_reward"]
                        .mean(-1)
                        .reshape(-1)[total_rew],
                    }
                )
            self.total_reward.finish()

    def log_rl_loss_minigrid(self, output, num_seeds):
        self.minigrid_losses = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
            name=f"{self.name}_rl_minigrid_loss",
        )

        if num_seeds == 1:
            for loss in range(len(output["rl_total_loss"])):
                self.minigrid_losses.log(
                    {
                        f"rl_total_loss_{self.config['ENV_NAME']}": output["rl_total_loss"][loss],
                        f"rl_value_loss_{self.config['ENV_NAME']}": output["rl_value_loss"][loss],
                        f"rl_actor_loss_{self.config['ENV_NAME']}": output["rl_actor_loss"][loss],
                        f"rl_entrophy_loss_{self.config['ENV_NAME']}": output["rl_entrophy_loss"][
                            loss
                        ],
                    }
                )
            self.minigrid_losses.finish()
        else:
            for loss in range(len(output["rl_total_loss"].mean(0))):
                self.minigrid_losses.log(
                    {
                        f"rl_total_loss_{self.config['ENV_NAME']}": output["rl_total_loss"].mean(0)[
                            loss
                        ],
                        f"rl_value_loss_{self.config['ENV_NAME']}": output["rl_value_loss"].mean(0)[
                            loss
                        ],
                        f"rl_actor_loss_{self.config['ENV_NAME']}": output["rl_actor_loss"].mean(0)[
                            loss
                        ],
                        f"rl_entrophy_loss_{self.config['ENV_NAME']}": output[
                            "rl_entrophy_loss"
                        ].mean(0)[loss],
                    }
                )
            self.minigrid_losses.finish()

    def save_artifact(self, path, type="dataset"):
        self.artifact = wandb.init(
            project="MetaLearnCuriosity",
            config=self.config,
            group=self.group,
            tags=self.tags,
            notes=self.notes,
            name=f"{self.name}_artifact",
        )
        # Create a new artifact
        artifact = wandb.Artifact(f"{self.name}_flax-checkpoints", type=type)

        # Add the checkpoint folder to the artifact
        artifact.add_dir(path)

        # Log the artifact to W&B
        self.artifact.log_artifact(artifact)

        # Finish the run
        self.artifact.finish()
