import time
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
from scipy.optimize import minimize

from traj_buffer import TrajBuffer


class MPO(object):
    """
        Maximum A Posteriori Policy Optimization (MPO) ; Discrete action-space ; Retrace

        Params:
            env: gym environment
            actor: actor network
            critic: critic network
            obs_shape: shape of observation (from env)
            action_shape: shape of action
            dual_constraint: learning rate of η in g(η)
            kl_constraint: Hard constraint on KL
            learning_rate: Bellman equation's decay for Q-retrace
            clip: 
            alpha: scaling factor of the lagrangian multiplier in the M-step
            episodes: number of iterations to sample episodes + do updates
            sample_episodes: number of episodes to sample
            episode_length: length of each episode
            lagrange_it: number of Lagrangian optimization steps
            runs: amount of training updates before updating target parameters
            device: pytorch device
            save_path: path to save model to
    """
    def __init__(self, env, actor, critic, obs_shape, action_shape,
                 dual_constraint=0.1, kl_constraint=0.01,
                 learning_rate=0.99, alpha=1.0,
                 episodes=1000, sample_episodes=1, episode_length=1000,
                 lagrange_it=5, runs=50, device='cpu',
                 save_path="./model/mpo"):
        # initialize env
        self.env = env

        # initialize some hyperparameters
        self.α = alpha  
        self.ε = dual_constraint 
        self.ε_kl = kl_constraint
        self.γ = learning_rate 
        self.episodes = episodes
        self.sample_episodes = sample_episodes
        self.episode_length = episode_length
        self.lagrange_it = lagrange_it
        self.mb_size = (episode_length-1) * env.num_envs
        self.runs = runs
        self.device = device

        # initialize networks and optimizer
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.critic = critic
        self.target_critic = deepcopy(critic)
        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.actor = actor
        self.target_actor = deepcopy(actor)
        for target_param, param in zip(self.target_actor.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # initialize Lagrange Multiplier
        self.η = np.random.rand()
        self.η_kl = 0.0

        # buffer and others
        self.buffer = TrajBuffer(env, episode_length, 100000)
        self.save_path = save_path

    def _sample_trajectory(self):
        mean_reward = 0

        for _ in range(self.sample_episodes):
            obs = self.env.reset()

            obs_b = np.zeros([self.episode_length, self.env.num_envs, self.obs_shape])
            action_b = np.zeros([self.episode_length, self.env.num_envs])
            reward_b = np.zeros([self.episode_length, self.env.num_envs])
            prob_b = np.zeros([self.episode_length, self.env.num_envs, self.action_shape])
            done_b = np.zeros([self.episode_length, self.env.num_envs])

            for steps in range(self.episode_length):
                action, prob = self.target_actor.action(torch.from_numpy(np.expand_dims(obs, axis=0)).to(self.device).float())
                action = np.reshape(action.cpu().numpy(), -1)
                prob = prob.cpu().numpy()

                new_obs, reward, done, _ = self.env.step(action)
                mean_reward += reward

                obs_b[steps] = obs
                action_b[steps] = action
                reward_b[steps] = reward
                prob_b[steps] = prob
                done_b[steps] = done

                obs = new_obs
            
            self.buffer.put(obs_b, action_b, reward_b, prob_b, done_b)

        return mean_reward

    def _update_critic_retrace(self, state_batch, action_batch, policies_batch, reward_batch, done_batch):
        action_size = policies_batch.shape[-1]
        nsteps = state_batch.shape[0]
        n_envs = state_batch.shape[1]

        self.critic_optimizer.zero_grad()

        with torch.no_grad():
            policies, a_log_prob, entropy = self.actor.evaluate_action(state_batch.view(-1, self.obs_shape), action_batch.view(-1, 1))
            target_policies, _, _ = self.target_actor.evaluate_action(state_batch.view(-1, self.obs_shape), action_batch.view(-1, 1))

        qval = self.critic(state_batch.view(-1, self.obs_shape))
        val = (qval * policies).sum(1, keepdim=True)

        old_policies = policies_batch.view(-1, action_size)
        policies = policies.view(-1, action_size)
        target_policies = target_policies.view(-1, action_size)

        val = val.view(-1, 1)
        qval = qval.view(-1, action_size)
        a_log_prob = a_log_prob.view(-1, 1)
        actions = action_batch.view(-1, 1)

        q_i = qval.gather(1, actions.long())
        rho = policies / (old_policies + 1e-10)
        rho_i = rho.gather(1, actions.long())

        with torch.no_grad():
            next_qval = self.critic(state_batch[-1]).detach()
            policies, a_log_prob, entropy = self.actor.evaluate_action(state_batch[-1], action_batch[-1])
            next_val = (next_qval * policies).sum(1, keepdim=True)
        
        q_retraces = reward_batch.new(nsteps + 1, n_envs, 1).zero_()
        q_retraces[-1] = next_val

        for step in reversed(range(nsteps)):
            q_ret = reward_batch[step] + self.γ * q_retraces[step + 1] * (1 - done_batch[step + 1])
            q_retraces[step] = q_ret
            q_ret = (rho_i[step] * (q_retraces[step] - q_i[step])) + val[step]
        
        q_retraces = q_retraces[:-1]
        q_retraces = q_retraces.view(-1, 1)

        q_loss = (q_i - q_retraces.detach()).pow(2).mean() * 0.5
        q_loss.backward()
        clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()

        return q_loss.detach()

    def _categorical_kl(self, p1, p2):
        p1 = torch.clamp_min(p1, 0.0001)
        p2 = torch.clamp_min(p2, 0.0001)
        return torch.mean((p1 * torch.log(p1 / p2)).sum(dim=-1))

    def _update_param(self):
        # Update policy parameters
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # Update critic parameters
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def train(self):
        # start training
        start_time = time.time()
        for episode in range(self.episodes):

            # Update replay buffer
            mean_reward = self._sample_trajectory()
            mean_q_loss = 0
            mean_policy = 0

            # Find better policy by gradient descent
            for _ in range(self.runs):
                state_batch, action_batch, reward_batch, policies_batch, done_batch = self.buffer.get()

                state_batch = torch.from_numpy(state_batch).to(self.device).float()[0:-1]
                action_batch = torch.from_numpy(action_batch).to(self.device).float()[0:-1]
                reward_batch = torch.from_numpy(reward_batch).to(self.device).float()[0:-1]
                policies_batch = torch.from_numpy(policies_batch).to(self.device).float()[0:-1]
                done_batch = torch.from_numpy(done_batch).to(self.device).float()[0:]

                reward_batch = torch.unsqueeze(reward_batch, dim=-1)
                done_batch = torch.unsqueeze(done_batch, dim=-1)

                # Update Q-function
                q_loss = self._update_critic_retrace(state_batch, action_batch, policies_batch, reward_batch, done_batch)
                mean_q_loss += q_loss

                # Sample values
                state_batch = state_batch.view(self.mb_size, *tuple(state_batch.shape[2:]))
                action_batch = action_batch.view(self.mb_size, *tuple(action_batch.shape[2:]))

                with torch.no_grad():
                    actions = torch.arange(self.action_shape)[..., None].expand(self.action_shape, self.mb_size).to(self.device)
                    b_p = self.target_actor.forward(state_batch)
                    b = Categorical(probs=b_p)
                    b_prob = b.expand((self.action_shape, self.mb_size)).log_prob(actions).exp()
                    target_q = self.target_critic.forward(state_batch)
                    target_q = target_q.transpose(0, 1)
                    b_prob_np = b_prob.cpu().numpy() 
                    target_q_np = target_q.cpu().numpy()
                
                # E-step
                # Update Dual-function
                def dual(η):
                    """
                    dual function of the non-parametric variational
                    g(η) = η*ε + η \sum \log (\sum \exp(Q(a, s)/η))
                    """
                    max_q = np.max(target_q_np, 0)
                    return η * self.ε + np.mean(max_q) \
                        + η * np.mean(np.log(np.sum(b_prob_np * np.exp((target_q_np - max_q) / η), axis=0)))

                bounds = [(1e-6, None)]
                res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
                self.η = res.x[0]

                # calculate the new q values
                qij = torch.softmax(target_q / self.η, dim=0)

                # M-step
                # update policy based on lagrangian
                for _ in range(self.lagrange_it):
                    π_p = self.actor.forward(state_batch)
                    π = Categorical(probs=π_p)
                    loss_p = torch.mean(
                        qij * π.expand((self.action_shape, self.mb_size)).log_prob(actions)
                    )
                
                    kl = self._categorical_kl(p1=π_p, p2=b_p)

                    # Update lagrange multipliers by gradient descent
                    self.η_kl -= self.α * (self.ε_kl - kl).detach().item()

                    if self.η_kl < 0.0:
                        self.η_kl = 0.0

                    self.actor_optimizer.zero_grad()
                    loss_policy = -(loss_p + self.η_kl * (self.ε_kl - kl))
                    loss_policy.backward()
                    clip_grad_norm_(self.actor.parameters(), 5.0)
                    self.actor_optimizer.step()
                    mean_policy += loss_policy.item()

            # Update target parameters
            self._update_param()

            print(f"Episode = {episode} ; "
                  f"Mean reward = {np.mean(mean_reward) / self.episode_length / self.sample_episodes} ; "
                  f"Mean Q loss = {mean_q_loss / self.runs} ; "
                  f"Policy loss = {mean_policy / self.runs} ; "
                  f"η = {self.η} ; η_kl = {self.η_kl} ; "
                  f"time = {(time.time() - start_time):.2f}")

            # Save model
            self.save_model()
            
    def load_model(self):
        checkpoint = torch.load(self.save_path)
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.η = checkpoint['lagrange_η']
        self.η_kl = checkpoint['lagrange_η_kl']
        self.critic.train()
        self.target_critic.train()
        self.actor.train()
        self.target_actor.train()

    def save_model(self):
        data = {
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict(),
            'lagrange_η': self.η,
            'lagrange_η_kl': self.η_kl
        }
        torch.save(data, self.save_path)
