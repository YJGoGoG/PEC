import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size
    ):
        super().__init__()
        # TODO: initialization
        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            self.add_module(f'fc{i}', fc)
            self.fcs.append(fc)
            in_size = next_size
        self.last_fc = nn.Linear(in_size, output_size)

    def forward(self, input):
        h = input
        for fc in self.fcs:
            h = F.relu(fc(h))
        output = self.last_fc(h)
        return output

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_quantiles, n_nets):
        super().__init__()
        self.nets = []
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets
        for i in range(n_nets):
            net = Mlp(state_dim + action_dim, [256,256], n_quantiles)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, state, action):
        sa = torch.cat((state, action), dim=1)
        quantiles = torch.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        first_net_output = self.nets[0](sa)
        return first_net_output

class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            num_q,
            critic_number,
            beta,
            headdrop_ratio,
            taildrop_ratio,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):
        self.num_q=int(num_q)
        self.headdrop_ratio=float(headdrop_ratio)
        self.taildrop_ratio=float(taildrop_ratio)
        self.critic_number = int(critic_number)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, self.num_q, self.critic_number).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.beta = beta

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            next_z = self.critic_target(next_state, next_action)
            # print("next_z",next_z)

            #Calculate mean + standard deviation
            mean_std_next_z=torch.mean(next_z, dim=2).unsqueeze(2) + self.beta*torch.std(next_z,dim=2).unsqueeze(2)

            #Get the minimum value and subscript
            mean_std_q,idx=torch.min(mean_std_next_z, dim=1, keepdim=True)

            #Select the corresponding target_q according to the subscripts
            expanded_idx = idx.expand(-1, -1, next_z.size(2))
            target_Q = torch.gather(next_z, 1, expanded_idx)

            #Calculate N*M mean
            all_avg_value= torch.mean(torch.mean(next_z, dim=2),dim=1).unsqueeze(1).unsqueeze(1)


            #Calculate the mean value of target_q
            target_avg_value=torch.mean(torch.mean(target_Q, dim=2),dim=1).unsqueeze(1).unsqueeze(1)
            comparison = target_avg_value < all_avg_value


            #sort
            target_Q1_sort, indices = torch.sort(target_Q, dim=2)

            #Trim number
            tail_drop = round(self.num_q * self.taildrop_ratio)
            head_drop = round(self.num_q * self.headdrop_ratio)
            num_elements_to_keep = self.num_q-tail_drop-head_drop

            # Initialize the target truncation value
            new_shape = list(target_Q1_sort.shape)
            new_shape[-1] = num_elements_to_keep
            truncate_Q = torch.randn(new_shape).to(device)
            # print("truncate_Q",truncate_Q)

            # Different truncations for different comparisons
            for i in range(comparison.size(0)):
                if comparison[i][0][0]:
                    truncate_Q[i][0] = target_Q1_sort[i][0][tail_drop:-head_drop]
                else:
                    truncate_Q[i][0] = target_Q1_sort[i][0][head_drop:-tail_drop]


            truncate_target=truncate_Q.reshape(batch_size, -1)
            target_Q = reward + not_done * self.discount * truncate_target
        cur_z = self.critic(state, action)
        critic_loss=TD3.quantile_huber_loss_f(self,cur_z, target_Q)
        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def quantile_huber_loss_f(self,quantiles, samples):
            pairwise_delta = samples[:, None, None, :] - quantiles[:, :,:, None]  # batch x nets x quantiles x samples
            abs_pairwise_delta = torch.abs(pairwise_delta)

            huber_loss = torch.where(abs_pairwise_delta > 1,
                                     abs_pairwise_delta - 0.5,
                                     pairwise_delta ** 2 * 0.5)
            n_quantiles = self.num_q
            tau = torch.arange(n_quantiles, device=device).float() / n_quantiles + 1 / 2 / n_quantiles
            loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float())
                    * huber_loss).mean()
            return loss