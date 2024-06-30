"""Distributed version of OpenAI-ES. Supports z-scoring fitness trafo only."""

from typing import Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from evosax import Strategy
from evosax.core import GradientOptimizer, OptParams, OptState, exp_decay
from flax import struct


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: float
    opt_state: OptState
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    opt_params: OptParams
    sigma_init: float = 0.04
    sigma_decay: float = 1.0
    sigma_limit: float = 0.01
    init_min: float = -2.0
    init_max: float = 2.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class OpenES(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        opt_name: str = "adam",
        lrate_init: float = 0.01,
        lrate_decay: float = 0.999,
        lrate_limit: float = 0.001,
        sigma_init: float = 0.04,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        mean_decay: float = 0.0,
        maximise_fitness: bool = True,
        n_devices: Optional[int] = None,
    ):
        """Pmapped version of OpenAI-ES (Salimans et al. (2017)
        Samples directly on different devices and updates mean using pmean grad.
        Reference: https://arxiv.org/pdf/1703.03864.pdf
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""
        super().__init__(
            popsize=popsize, num_dims=num_dims, pholder_params=pholder_params, n_devices=n_devices
        )
        assert not self.popsize & 1, "Population size must be even"
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup", "adan"]
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.strategy_name = "DistributedOpenES"
        self.lrate_init = lrate_init
        self.lrate_decay = lrate_decay
        self.lrate_limit = lrate_limit
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.maximise_fitness = maximise_fitness

        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices

        # Mean exponential decay coefficient m' = coeff * m
        # Implements form of weight decay regularization
        self.mean_decay = mean_decay
        self.use_mean_decay = mean_decay > 0.0

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        opt_params = self.optimizer.default_params.replace(
            lrate_init=self.lrate_init,
            lrate_decay=self.lrate_decay,
            lrate_limit=self.lrate_limit,
        )
        es_params = EvoParams(
            opt_params=opt_params,
            sigma_init=self.sigma_init,
            sigma_decay=self.sigma_decay,
            sigma_limit=self.sigma_limit,
        )
        if self.n_devices > 1:
            es_params = jax.tree_map(lambda x: jnp.array([x] * self.n_devices), es_params)
        else:
            es_params = es_params
        return es_params

    def initialize(
        self,
        rng: chex.PRNGKey,
        params: Optional[EvoParams] = None,
        init_mean: Optional[Union[chex.Array, chex.ArrayTree]] = None,
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Initialize strategy based on strategy-specific initialize method
        state = self.initialize_strategy(rng, params)

        if init_mean is not None:
            state = self.set_mean(state, init_mean)
        return state

    def initialize_strategy(self, rng: chex.PRNGKey, params: EvoParams) -> EvoState:
        """`initialize` the evolution strategy."""
        if self.n_devices > 1:
            mean, sigma, opt_state = self.multi_init(rng, params)
        else:
            mean, sigma, opt_state = self.single_init(rng, params)

        state = EvoState(
            mean=mean,
            sigma=sigma,
            opt_state=opt_state,
            best_member=mean,
        )
        return state

    def multi_init(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> Tuple[chex.Array, chex.Array, OptState]:
        """`initialize` the evolution strategy on multiple devices (same)."""
        # Use rng tile to create same random sample across devices
        batch_rng = jnp.tile(rng, (self.n_devices, 1))
        mean, sigma, opt_state = jax.pmap(self.single_init)(batch_rng, params)
        return mean, sigma, opt_state

    def single_init(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> Tuple[chex.Array, chex.Array, OptState]:
        """`initialize` the evolution strategy on a single device."""
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        opt_state = self.optimizer.initialize(params.opt_params)
        mean = initialization
        sigma = params.sigma_init
        return mean, sigma, opt_state

    def ask(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        x = self.multi_ask(rng, state.mean, state.sigma)
        x = self.param_reshaper.multi_reshape(x)
        return x, state

    def multi_ask(self, rng: chex.PRNGKey, mean: chex.Array, sigma: chex.Array) -> chex.Array:
        """Pmapped antithetic sampling of noise and reparametrization."""
        # Use rng split to create different random sample across devices
        batch_rng = jax.random.split(rng, self.n_devices)
        x = jax.pmap(self.single_ask)(batch_rng, mean, sigma)
        return x

    def single_ask(self, rng: chex.PRNGKey, mean: chex.Array, sigma: chex.Array) -> chex.Array:
        """Antithetic sampling of noise and reparametrization."""
        z_plus = jax.random.normal(
            rng,
            (int(self.popsize / (2 * self.n_devices)), self.num_dims),
        )
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        x = mean + sigma * z
        return x

    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        def update_fitness(fitness):
            # Use jax.lax.cond to handle conditional logic
            fitness = jax.lax.cond(self.maximise_fitness, lambda f: -f, lambda f: f, fitness)
            return fitness

        fitness = update_fitness(fitness)

        """`tell` performance data for strategy state update."""
        x = self.param_reshaper.multi_flatten(x)
        mean, sigma, opt_state = self.multi_tell(x, fitness, state, params)

        # Exponentially decay mean if coefficient < 1.0
        def update_mean(mean):
            # Conditional update using jax.lax.cond
            mean = jax.lax.cond(
                self.use_mean_decay,
                lambda m: m * (1 - self.mean_decay),  # true function
                lambda m: m,  # false function
                mean,
            )
            return mean

        mean = update_mean(mean)

        # TODO(RobertTLange): Add tracking of best member/fitness score
        return state.replace(mean=mean, sigma=sigma, opt_state=opt_state)

    def multi_tell(self, x, fitness, state, params) -> Tuple[chex.Array, chex.Array, OptState]:
        """Pmapped tell update call over multiple devices."""

        fitness = pmap_centered_rank(fitness)

        def calc_per_device_grad(x, fitness, mean, sigma):
            # Reconstruct noise from last mean/std estimates
            noise = (x - mean) / sigma
            theta_grad = 1.0 / (self.popsize * sigma) * jnp.dot(noise.T, fitness)
            return jax.lax.pmean(theta_grad, axis_name="p")

        theta_grad = jax.pmap(calc_per_device_grad, axis_name="p")(
            x, fitness, state.mean, state.sigma
        )

        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = jax.pmap(self.optimizer.step)(
            state.mean, theta_grad, state.opt_state, params.opt_params
        )
        opt_state = jax.pmap(self.optimizer.update)(opt_state, params.opt_params)
        sigma = jax.pmap(exp_decay)(state.sigma, params.sigma_decay, params.sigma_limit)
        return mean, sigma, opt_state


def pmap_zscore(fitness: chex.Array) -> chex.Array:
    """Pmappable version of z-scoring of fitness scores."""

    def zscore(fit: chex.Array) -> chex.Array:
        all_mean = jax.lax.pmean(fit, axis_name="p").mean()
        diff = fit - all_mean
        std = jnp.sqrt(jax.lax.pmean(diff**2, axis_name="p").mean())
        return diff / (std + 1e-10)

    out = jax.pmap(zscore, axis_name="p")(fitness)
    return out


@jax.jit
def compute_ranks(fitness: chex.Array) -> chex.Array:
    """Return fitness ranks in [0, len(fitness))."""
    ranks = jnp.zeros(len(fitness))
    ranks = ranks.at[fitness.argsort()].set(jnp.arange(len(fitness)))
    return ranks


@jax.jit
def pmap_centered_rank(fitness: chex.Array) -> chex.Array:
    """Return ~ -0.5 to 0.5 centered ranks (best to worst - min!).
    Assumes fitness is shaped as (num_devices, num_samples_per_device) and flattens for global rank.
    """
    global_fitness = fitness.reshape(-1)
    global_fitness = compute_ranks(global_fitness)
    global_fitness = global_fitness.reshape(fitness.shape)
    global_fitness /= global_fitness.flatten().size - 1
    return global_fitness - 0.5
