# Taken from:
# https://github.com/corl-team/xland-minigrid/blob/main/training/nn.py

import functools
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


class MiniGridBYOLPredictor(nn.Module):
    encoder_layer_out_shape: Sequence[int]
    num_actions: int
    action_emb_dim: int = 16

    @nn.compact
    def __call__(self, close_hidden, open_hidden, x):
        action_encoder = nn.Embed(self.num_actions, self.action_emb_dim)

        bt, obs, action = x

        # Encoder
        en_obs = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            name="encoder_layer_1",
            bias_init=constant(0.0),
        )(obs)
        en_obs = nn.relu(en_obs)
        en_obs = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            name="encoder_layer_2",
            bias_init=constant(0.0),
        )(en_obs)

        # RL AGENT
        # actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
        #     en_obs
        # )
        # actor_mean = nn.relu(actor_mean)
        # actor_mean = nn.Dense(
        #     self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        # )(actor_mean)
        # pi = distrax.Categorical(logits=actor_mean)

        # critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(en_obs)
        # critic = nn.relu(critic)
        # critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        # Embed the action
        act_emb = action_encoder(action)

        # RNN stuff
        close_rnn_core = MiniGridBatchedRNNModel(self.encoder_layer_out_shape, 1)
        open_rnn_core = MiniGridBatchedRNNModel(self.encoder_layer_out_shape, 1)

        close_loop_input = jnp.concatenate((bt, en_obs, act_emb), axis=-1)
        new_bt, new_close_hidden = close_rnn_core(close_loop_input, close_hidden)
        open_loop_input = jnp.concatenate((new_bt, act_emb), axis=-1)
        bt_1, new_open_hidden = open_rnn_core(open_loop_input, open_hidden)

        # Predictor
        pred_fut = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            name="pred_layer_1",
            bias_init=constant(0.0),
        )(bt_1)
        pred_fut = nn.relu(pred_fut)
        pred_fut = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            name="pred_layer_2",
            bias_init=constant(0.0),
        )(pred_fut)

        return pred_fut, new_bt, new_close_hidden, new_open_hidden

    def initialize_carry(self, batch_size):
        return jnp.zeros((batch_size, 1, self.encoder_layer_out_shape))


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


# class RewardCombiner(nn.Module):
#     @nn.compact
#     def __call__(self, x):

#         # Input shape: (num_envs, history_length, 2)

#         x = nn.Conv(features=16, kernel_size=(3,), padding="valid")(x)
#         x = nn.relu(x)
#         # x = nn.max_pool(x, window_shape=(2,), strides=(2,), padding='valid')

#         x = nn.Conv(features=32, kernel_size=(3,), padding="valid")(x)
#         x = nn.relu(x)
#         # x = nn.max_pool(x, window_shape=(2,), strides=(2,), padding='valid')

#         x = x.reshape((x.shape[0], -1))  # Flatten

#         x = nn.Dense(features=128)(x)
#         x = nn.relu(x)

#         x = nn.Dense(features=1)(x)
#         return jnp.squeeze(nn.sigmoid(x), -1)


# class RewardCombiner(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         # Input shape: (num_envs, 128, 2)
#         x = nn.Conv(features=16, kernel_size=(3,), padding="valid")(x)
#         x = nn.relu(x)

#         x = nn.Conv(features=32, kernel_size=(3,), padding="valid")(x)
#         x = nn.relu(x)

#         # Flatten the output
#         x = x.reshape((x.shape[0], -1))  # Flatten all dimensions except batch

#         # Fully connected layer
#         x = nn.Dense(features=128)(x)
#         x = nn.relu(x)

#         # Output layer with sigmoid activation
#         x = nn.Dense(features=1)(x)
#         return jnp.squeeze(nn.sigmoid(x), -1)


class RewardCombiner(nn.Module):
    @nn.compact
    def __call__(self, carry, x):

        # Input is (1, num_envs, 2)

        carry, x = RCRNN()(carry, x)  # features is 16

        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return carry, jnp.squeeze(nn.sigmoid(x), -1)


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


class BYOLEncoder(nn.Module):
    encoder_layer_out_shape: Sequence[int]

    @nn.compact
    def __call__(self, x):
        encoded_obs = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        encoded_obs = nn.relu(encoded_obs)
        encoded_obs = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(encoded_obs)
        # encoded_obs = nn.relu(encoded_obs)
        return encoded_obs


class BYOLTarget(nn.Module):
    encoder_layer_out_shape: Sequence[int]

    @nn.compact
    def __call__(self, x):
        encoded_obs = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        encoded_obs = nn.relu(encoded_obs)
        encoded_obs = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(encoded_obs)
        # encoded_obs = nn.relu(encoded_obs)
        return encoded_obs


class RCRNN(nn.Module):
    features: int = 32  # Number of hidden units

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        new_rnn_state, y = nn.GRUCell(
            self.features,
        )(rnn_state, x)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class CloseScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module. Assumes the action and the
        encoded observation are concatenated."""
        features = carry[0].shape[-1]
        rnn_state = carry
        new_rnn_state, y = nn.GRUCell(features, kernel_init=orthogonal(np.sqrt(2)))(rnn_state, x)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*(batch_size,), hidden_size)
        )


class OpenScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module. Assumes the action and the
        previous history representation is concatenated."""
        features = carry[0].shape[-1]
        rnn_state = carry
        new_rnn_state, y = nn.GRUCell(features, kernel_init=orthogonal(np.sqrt(2)))(rnn_state, x)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*(batch_size,), hidden_size)
        )


class AtariBYOLPredictor(nn.Module):
    encoder_layer_out_shape: Sequence[int]
    num_actions: int
    action_emb_dim: int = 16

    @nn.compact
    def __call__(self, close_hidden, open_hidden, x):
        action_encoder = nn.Embed(self.num_actions, self.action_emb_dim)

        bt, obs, action = x

        # Encoder
        en_obs = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            name="encoder_layer_1",
            bias_init=constant(0.0),
        )(obs)
        en_obs = nn.relu(en_obs)
        en_obs = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            name="encoder_layer_2",
            bias_init=constant(0.0),
        )(en_obs)

        # RL AGENT
        # actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
        #     en_obs
        # )
        # actor_mean = nn.relu(actor_mean)
        # actor_mean = nn.Dense(
        #     self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        # )(actor_mean)
        # pi = distrax.Categorical(logits=actor_mean)

        # critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(en_obs)
        # critic = nn.relu(critic)
        # critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        # Embed the action
        act_emb = action_encoder(action)

        # RNN stuff
        close_loop_input = jnp.concatenate((bt, en_obs, act_emb), axis=-1)
        new_close_hidden, new_bt = CloseScannedRNN()(close_hidden, close_loop_input)
        open_loop_input = jnp.concatenate((new_bt, act_emb), axis=-1)
        new_open_hidden, bt_1 = OpenScannedRNN()(open_hidden, open_loop_input)

        # Predictor
        pred_fut = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            name="pred_layer_1",
            bias_init=constant(0.0),
        )(bt_1)
        pred_fut = nn.relu(pred_fut)
        pred_fut = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            name="pred_layer_2",
            bias_init=constant(0.0),
        )(pred_fut)

        return pred_fut, new_bt, new_close_hidden, new_open_hidden


class BraxBYOLPredictor(nn.Module):
    encoder_layer_out_shape: Sequence[int]

    @nn.compact
    def __call__(self, close_hidden, open_hidden, x):
        bt, obs, action = x

        # Encoder
        en_obs = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            name="encoder_layer_1",
            bias_init=constant(0.0),
        )(obs)
        en_obs = nn.relu(en_obs)
        en_obs = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            name="encoder_layer_2",
            bias_init=constant(0.0),
        )(en_obs)

        # RNN stuff
        close_loop_input = jnp.concatenate((bt, en_obs, action), axis=-1)
        new_close_hidden, new_bt = CloseScannedRNN()(close_hidden, close_loop_input)
        open_loop_input = jnp.concatenate((new_bt, action), axis=-1)
        new_open_hidden, bt_1 = OpenScannedRNN()(open_hidden, open_loop_input)

        # Predictor
        pred_fut = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            name="pred_layer_1",
            bias_init=constant(0.0),
        )(bt_1)
        pred_fut = nn.relu(pred_fut)
        pred_fut = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            name="pred_layer_2",
            bias_init=constant(0.0),
        )(pred_fut)

        return pred_fut, new_bt, new_close_hidden, new_open_hidden
