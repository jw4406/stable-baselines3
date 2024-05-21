from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.buffers import AdvRolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy, ActorActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance
import torch
import torch.autograd as autograd
SelfA2C = TypeVar("SelfA2C", bound="A2C")
#SelfAAC = TypeVar("SelfAAC", bound="A3")

class A3C_rarl(OnPolicyAlgorithm):
    """
    Advantage Actor Critic (A2C)

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
        "MlPAACPolicy": ActorActorCriticPolicy
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorActorCriticPolicy]],
        env: Union[GymEnv, str],
        c_learning_rate: Union[float, Schedule] = 1e-4,
        d_learning_rate: Union[float, Schedule] = 7e-4,
        v_learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[AdvRolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        adversarial=True,
        use_stackelberg=False
    ):
        super().__init__(
            policy,
            env,
            learning_rate=v_learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        self.adversarial=adversarial
        self.use_stackelberg = use_stackelberg
        self.normalize_advantage = normalize_advantage
        self.c_learning_rate = c_learning_rate
        self.v_learning_rate = v_learning_rate
        self.d_learning_rate = d_learning_rate
        self.learning_rate = [v_learning_rate, c_learning_rate, d_learning_rate]
        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """


        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.ctrl_optimizer)
        self._update_learning_rate(self.policy.dstb_optimizer)
        self._update_learning_rate(self.policy.value_optimizer)
        #self._update_learning_rate(self.policy.optimizer)
        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            dstb_actions = rollout_data.dstb_actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            values, ctrl_log_prob, ctrl_entropy, dstb_log_prob, dstb_entropy = self.policy.evaluate_actions(rollout_data.observations, actions, dstb_actions)
            if self.use_stackelberg is True: # need to get V^\pi (x0) and V_\omega (x0) from the scrambled data
                if self.rollout_buffer.split_trajectories is True:
                    # we need to discard the previous trajectory and set V_\omega (x_0) and V^\pi (x_0) to the correct values
                    self.rollout_buffer.split_trajectories = False
                    #assert self.rollout_buffer.sanity_value_start == self.rollout_buffer.split_value_start
                    #assert self.rollout_buffer.sanity_return_start == self.rollout_buffer.split_return_start
                    self.rollout_buffer.value_start = self.rollout_buffer.split_value_start
                    self.rollout_buffer.return_start = self.rollout_buffer.split_return_start
                    self.rollout_buffer.split_value_start = []
                    self.rollout_buffer.split_return_start = []

                if 1 in self.rollout_buffer.episode_starts:
                    self.rollout_buffer.has_multi_start = False
                    self.rollout_buffer.split_trajectories = False
                    self.rollout_buffer.next_traj_begin = None
                    if np.argwhere(self.rollout_buffer.episode_starts).shape[0] > 1:
                        self.rollout_buffer.has_multi_start = True
                    loc = np.argwhere(self.rollout_buffer.episode_starts)
                    self.rollout_buffer.next_traj_begin = loc[0, 0]
                    index_0 = np.argwhere(self.rollout_buffer.indices == loc[0, 0])
                    index_1 = np.argwhere(self.rollout_buffer.indices == loc[0,0]+1)
                    if loc[0, 0] != 0:
                        # this rollout buffer has the end of one trajectory and the start of another!
                        # example: traj1, traj1, traj1_end, traj2_begin, traj2, traj2, etc etc
                        self.rollout_buffer.split_trajectories = True
                        #with torch.no_grad():
                        #    assert torch.allclose(torch.tensor(self.rollout_buffer.split_value_start), values[index_0])
                        #    assert torch.allclose(torch.tensor(self.rollout_buffer.split_return_start), rollout_data.returns[index_0])
                        self.rollout_buffer.split_value_start = values[index_0]
                        self.rollout_buffer.split_return_start = rollout_data.returns[index_0]
                        self.rollout_buffer.rew_zero = self.rollout_buffer.rewards[loc[0,0]]
                        self.rollout_buffer.value_x1 = values[index_1]
                    else:
                        #rollout_data.observations[0] - torch.tensor(self.rollout_buffer.observations[self.rollout_buffer.indices[0], :])
                        #with torch.no_grad():
                        #    assert torch.allclose(torch.tensor(self.rollout_buffer.value_start), values[index_0])
                        #    assert torch.allclose(torch.tensor(self.rollout_buffer.return_start), rollout_data.returns[index_0])
                        self.rollout_buffer.value_start = values[index_0]
                        self.rollout_buffer.return_start = rollout_data.returns[index_0]
                        self.rollout_buffer.rew_zero = self.rollout_buffer.rewards[loc[0,0]]
                        self.rollout_buffer.value_x1 = values[index_1]
                        self.rollout_buffer.obs_x0 = rollout_data.observations[index_0]
                        self.rollout_buffer.obs_x1 = rollout_data.observations[index_1]
            values = values.flatten()
            advantages = rollout_data.advantages
            if self.use_stackelberg:
                x0_values,  _, _, _, _ = self.policy.evaluate_actions(self.rollout_buffer.obs_x0, actions[0], dstb_actions[0])
                x1_values, _, _, _, _ = self.policy.evaluate_actions(self.rollout_buffer.obs_x1, actions[0], dstb_actions[0])
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            if self.use_stackelberg is True:
                # Build h1 vector
                h1_upper_pre = self.prep_grad_theta_omega_J(x1_values, ctrl_log_prob)
                h1_upper_grad_batched = autograd.grad(h1_upper_pre,
                                                      self.policy.ctrl_optimizer.param_groups[0]['params'],
                                                      create_graph=True, retain_graph=True)
                h1_upper = torch.hstack([t.flatten() for t in h1_upper_grad_batched])
                h1_lower_pre = self.prep_grad_psi_omega_J(x1_values, dstb_log_prob)
                h1_lower_grad_batched = autograd.grad(h1_lower_pre,
                                                      self.policy.dstb_optimizer.param_groups[0]['params'],
                                                      create_graph=True, retain_graph=True)
                h1_lower = torch.hstack([t.flatten() for t in h1_lower_grad_batched])
                h1_pre_omega = torch.hstack((h1_upper, h1_lower))
                # TODO: jvp with h1

                # Build h2 vector
                h2_upper_pre = self.prep_grad_theta_L(advantages, ctrl_log_prob, x0_values)
                h2_upper_grad_batched = autograd.grad(h2_upper_pre,
                                                      self.policy.ctrl_optimizer.param_groups[0]['params'],
                                                      create_graph=True, retain_graph=True)
                h2_upper = torch.hstack([t.flatten() for t in h2_upper_grad_batched])
                h2_lower_pre = self.prep_grad_psi_L(advantages, dstb_log_prob, x0_values)
                h2_lower_grad_batched = autograd.grad(h2_lower_pre,
                                                      self.policy.dstb_optimizer.param_groups[0]['params'],
                                                      create_graph=True, retain_graph=True)
                h2_lower = torch.hstack([t.flatten() for t in h2_lower_grad_batched])
                h2 = torch.hstack((h2_upper, h2_lower))

                # Build big H matrix
                # H is a 2x2 block matrix. The diagonals are hessians -- H_\theta (J) and H_\psi (J). The off diagonal terms are
                # cross gradients -- \grad_{\theta\psi} J and \grad_{\psi\theta} J.
                # We assemble it block by block.

                # Diagonal terms (Hessians) first
                hess_theta_J_batched = autograd.grad(h1_upper, self.policy.ctrl_optimizer.param_groups[0]['params'],
                                                     torch.eye(h1_upper.shape[0], device=self.device),
                                                     is_grads_batched=True, create_graph=True, retain_graph=True)
                hess_psi_J_batched = autograd.grad(h1_lower, self.policy.dstb_optimizer.param_groups[0]['params'],
                                                   torch.eye(h1_lower.shape[0], device=self.device),
                                                   is_grads_batched=True, create_graph=True, retain_graph=True)
                num_params = 0
                for ele in self.policy.ctrl_optimizer.param_groups[0]['params']:
                    num_params = num_params + torch.numel(ele)

                hess_theta_J = self.matrix_unbatch(hess_theta_J_batched, num_params)  # this is the 1,1 position
                hess_psi_J = self.matrix_unbatch(hess_psi_J_batched, num_params)  # this is the 2,2 position

                # Cross terms next. This is the harder part
                cross_pre = self.prep_grad_theta_psi_J(advantages, ctrl_log_prob, dstb_log_prob)
                grad_theta_J_batched = autograd.grad(cross_pre,
                                                     self.policy.ctrl_optimizer.param_groups[0]['params'],
                                                     create_graph=True, retain_graph=True)
                grad_theta_J = torch.hstack([t.flatten() for t in grad_theta_J_batched])
                grad_theta_psi_J_batched = autograd.grad(grad_theta_J,
                                                         self.policy.dstb_optimizer.param_groups[0]['params'],
                                                         torch.eye(num_params, device=self.device),
                                                         is_grads_batched=True, create_graph=True,
                                                         retain_graph=True)

                grad_theta_psi_J = self.matrix_unbatch(grad_theta_psi_J_batched,
                                                       num_params)  # this is the 1,2 position

                grad_theta_psi_J_t = torch.transpose(grad_theta_psi_J, 0, 1)  # this is the 2,1 position
                x, y = torch.cat((hess_theta_J, grad_theta_psi_J, grad_theta_psi_J_t, hess_psi_J),
                                 dim=1).t().chunk(2)
                H = torch.cat((x, y), dim=1).t()
                ivp_H_h2 = torch.linalg.solve(H, h2)

                # TODO: need to test if doing a grad omega on h1 and then multiply that to ivpH_H2 is the same as
                # TODO: doing h1 times ivp_H_h2 and then doing a grad (basically whether grad is first or last)
                imp = autograd.grad(h1_pre_omega, self.policy.value_optimizer.param_groups[0]['params'], ivp_H_h2,
                                    create_graph=True, retain_graph=True)
                # imp is the stackelberg part of the total derivative

            # Policy gradient loss
            policy_loss = -(advantages * ctrl_log_prob).mean()
            dstb_policy_loss = (advantages * dstb_log_prob).mean()
            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)
            #if self.use_stackelberg is True:
            #    grad_omega_L_batched = autograd.grad(value_loss, self.policy.value_optimizer.param_groups[0]['params'])

            #    #stackelberg_loss = grad_omega_L_batched - imp
            #    stackelberg_loss = list()
            #    assert len(grad_omega_L_batched) == len(imp)
            #    for i in range(len(grad_omega_L_batched)):
            #        stackelberg_loss.append(grad_omega_L_batched[i] - imp[i])

            # Entropy loss favor exploration
            if ctrl_entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-ctrl_log_prob)
            else:
                entropy_loss = -th.mean(ctrl_entropy)

            #loss = policy_loss + dstb_policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step

            self.policy.value_optimizer.zero_grad()
            value_loss.backward()

            if self.use_stackelberg is True:
                for i in range(len(self.policy.value_optimizer.param_groups[0]['params'])):
                    self.policy.value_optimizer.param_groups[0]['params'][i].grad = self.policy.value_optimizer.param_groups[0]['params'][i].grad - imp[i]
                del imp
            #if self.use_stackelberg:
            #    for i in range(len(self.policy.value_optimizer.param_groups[0]['params'])):
            #        self.policy.value_optimizer.param_groups[0]['params'][i] = self.policy.value_optimizer.param_groups[0]['params'][i] - self.v_learning_rate * stackelberg_loss[i]

            self.policy.ctrl_optimizer.zero_grad()
            policy_loss.backward()
            self.policy.dstb_optimizer.zero_grad()
            dstb_policy_loss.backward()
            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            #th.nn.utils.clip_grad_norm_(self.policy.advantages.parameters(), self.max_grad_norm)
            #self.policy.optimizer.step()
            self.policy.value_optimizer.step()
            self.policy.ctrl_optimizer.step()
            self.policy.dstb_optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def learn(
        self: SelfA2C,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "A2C",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfA2C:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
    def prep_grad_theta_omega_J(self, x1_values, ctrl_logp):
        return ((torch.tensor(self.rollout_buffer.rew_zero, device=self.device) + x1_values) * ctrl_logp).mean()

    def prep_grad_psi_omega_J(self, x1_values, dstb_logp):
        return ((torch.tensor(self.rollout_buffer.rew_zero, device=self.device) + x1_values) * dstb_logp).mean()

    def prep_grad_theta_psi_J(self, advantages, ctrl_logp, dstb_logp):
        return (advantages * ctrl_logp * dstb_logp).mean()
    def prep_grad_theta_L(self, advantages, ctrl_logp, x0_values):
        # TODO: make sure returns is correct!

        if self.rollout_buffer.split_trajectories is True: # we have split trajectories in this buffer sample
            # this is the harder case
            grad_estimator = None
        else: # all belong to one trajectory; this is the easier case
            grad_estimator = 2 * (advantages * ctrl_logp).mean() * (self.rollout_buffer.return_start - x0_values)
        return grad_estimator
    def prep_grad_psi_L(self, advantages, dstb_logp, x0_values):
        return 2 * (advantages * dstb_logp).mean() * (self.rollout_buffer.return_start - x0_values)
    def matrix_unbatch(self, to_be_unbatched, size1, size2=None):
        if size2 is None:
            size2 = size1
        unbatched = torch.zeros((size1,size2), device=self.device)
        for jac_row_count in range(size1):
            curr = 0
            for count in range(len(to_be_unbatched)):
                unbatched[jac_row_count,
                curr:curr + len(
                    torch.flatten(to_be_unbatched[count][jac_row_count, :]))] = torch.flatten(
                    to_be_unbatched[count][jac_row_count, :])
                curr = curr + len(torch.flatten(to_be_unbatched[count][jac_row_count, :]))
        return unbatched