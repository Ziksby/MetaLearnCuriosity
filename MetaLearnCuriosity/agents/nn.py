# Taken from:
# https://github.com/corl-team/xland-minigrid/blob/main/training/nn.py

import math
from typing import Sequence, TypedDict

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, glorot_normal, orthogonal, zeros_init


class MiniGridGRU(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, xs, init_state):
        seq_len, input_dim = xs.shape
        # this init might not be optimal, for example bias for reset gate should be -1 (for now ok)
        Wi = self.param(
            "Wi", glorot_normal(in_axis=1, out_axis=0), (self.hidden_dim * 3, input_dim)
        )
        Wh = self.param("Wh", orthogonal(column_axis=0), (self.hidden_dim * 3, self.hidden_dim))
        bi = self.param("bi", zeros_init(), (self.hidden_dim * 3,))
        bn = self.param("bn", zeros_init(), (self.hidden_dim,))

        def _step_fn(h, x):
            igates = jnp.split(Wi @ x + bi, 3)
            hgates = jnp.split(Wh @ h, 3)

            reset = nn.sigmoid(igates[0] + hgates[0])
            update = nn.sigmoid(igates[1] + hgates[1])
            new = nn.tanh(igates[2] + reset * (hgates[2] + bn))
            next_h = (1 - update) * new + update * h

            return next_h, next_h

        last_state, all_states = jax.lax.scan(_step_fn, init=init_state, xs=xs)
        return all_states, last_state


class MiniGridRNNModel(nn.Module):
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, xs, init_state):
        # xs: [seq_len, input_dim]
        # init_state: [num_layers, hidden_dim]
        outs, states = [], []
        for layer in range(self.num_layers):
            xs, state = MiniGridGRU(hidden_dim=self.hidden_dim)(xs, init_state[layer])
            outs.append(xs)
            states.append(state)

        # sum outputs from all layers, kinda like in ResNet
        return jnp.array(outs).sum(0), jnp.array(states)


MiniGridBatchedRNNModel = flax.linen.vmap(
    MiniGridRNNModel,
    variable_axes={"params": None},
    split_rngs={"params": False},
    axis_name="batch",
)


class MiniGridActorCriticInput(TypedDict):
    observation: jax.Array
    prev_action: jax.Array
    prev_reward: jax.Array


class MiniGridActorCriticRNN(nn.Module):
    num_actions: int
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 1
    head_hidden_dim: int = 64
    use_cnns: bool = False

    @nn.compact
    def __call__(
        self, inputs: MiniGridActorCriticInput, hidden: jax.Array
    ) -> tuple[distrax.Categorical, jax.Array, jax.Array]:

        # encoder from https://github.com/lcswillems/rl-starter-files/blob/master/model.py
        B, S = inputs["observation"].shape[:2]
        if self.use_cnns:
            img_encoder = nn.Sequential(
                [
                    nn.Conv(16, (2, 2), padding="VALID", kernel_init=orthogonal(math.sqrt(2))),
                    nn.relu,
                    # use this only for image sizes >= 7
                    # MaxPool2d((2, 2)),
                    nn.Conv(32, (2, 2), padding="VALID", kernel_init=orthogonal(math.sqrt(2))),
                    nn.relu,
                    nn.Conv(64, (2, 2), padding="VALID", kernel_init=orthogonal(math.sqrt(2))),
                    nn.relu,
                ]
            )

            obs_emb = img_encoder(inputs["observation"]).reshape(B, S, -1)
        else:
            mlp_encoder = nn.Sequential(
                [
                    nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
                    nn.relu,
                ]
            )
            obs_emb = mlp_encoder(inputs["observation"])
        action_encoder = nn.Embed(self.num_actions, self.action_emb_dim)

        rnn_core = MiniGridBatchedRNNModel(self.rnn_hidden_dim, self.rnn_num_layers)
        actor = nn.Sequential(
            [
                nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(2)),
                nn.tanh,
                nn.Dense(self.num_actions, kernel_init=orthogonal(0.01)),
            ]
        )
        critic = nn.Sequential(
            [
                nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(2)),
                nn.tanh,
                nn.Dense(1, kernel_init=orthogonal(1.0)),
            ]
        )

        # [batch_size, seq_len, ...]

        act_emb = action_encoder(inputs["prev_action"])
        # [batch_size, seq_len, hidden_dim + act_emb_dim + 1]
        out = jnp.concatenate([obs_emb, act_emb, inputs["prev_reward"][..., None]], axis=-1)
        # core networks
        out, new_hidden = rnn_core(out, hidden)
        dist = distrax.Categorical(logits=actor(out))
        values = critic(out)

        return dist, jnp.squeeze(values, axis=-1), new_hidden

    def initialize_carry(self, batch_size):
        return jnp.zeros((batch_size, self.rnn_num_layers, self.rnn_hidden_dim))


class RewardCombiner(nn.Module):
    @nn.compact
    def __call__(self, x):

        int_lambda = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        int_lambda = nn.relu(int_lambda)
        int_lambda = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            int_lambda
        )
        int_lambda = nn.relu(int_lambda)
        int_lambda = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(int_lambda)
        # int_lambda = nn.tanh(int_lambda)

        return jnp.squeeze(int_lambda, axis=-1)


class TemporalRewardCombiner(nn.Module):
    @nn.compact
    def __call__(self, x):

        int_lambda = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        int_lambda = nn.relu(int_lambda)
        int_lambda = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            int_lambda
        )
        int_lambda = nn.relu(int_lambda)
        int_lambda = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(int_lambda)
        int_lambda = nn.tanh(int_lambda)

        return jnp.squeeze(int_lambda, axis=-1)


class TargetNetwork(nn.Module):
    encoder_layer_out_shape: Sequence[int]

    @nn.compact
    def __call__(self, x):
        encoded_obs = nn.Dense(self.encoder_layer_out_shape)(x)
        encoded_obs = nn.relu(encoded_obs)
        encoded_obs = nn.Dense(self.encoder_layer_out_shape)(encoded_obs)
        # encoded_obs = nn.relu(encoded_obs)
        # encoded_obs = nn.Dense(
        #     self.encoder_layer_out_shape
        # )(encoded_obs)

        return encoded_obs


class PredictorNetwork(nn.Module):
    encoder_layer_out_shape: Sequence[int]

    @nn.compact
    def __call__(self, x):
        encoded_obs = nn.Dense(self.encoder_layer_out_shape)(x)
        encoded_obs = nn.relu(encoded_obs)
        encoded_obs = nn.Dense(self.encoder_layer_out_shape)(encoded_obs)
        encoded_obs = nn.relu(encoded_obs)
        encoded_obs = nn.Dense(self.encoder_layer_out_shape)(encoded_obs)

        return encoded_obs
