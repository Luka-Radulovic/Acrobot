import torch
import numpy as np
import matplotlib.pyplot as plt
import optuna

from pearl.neural_networks.sequential_decision_making.q_value_networks import VanillaQValueNetwork
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.policy_learners.sequential_decision_making.double_dqn import DoubleDQN
from pearl.replay_buffers import BasicReplayBuffer
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.pearl_agent import PearlAgent
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)

def objective(trial):
    set_seed(216028)

    env = GymEnvironment("Acrobot-v1")
    num_actions = env.action_space.n

    hidden_dims = trial.suggest_categorical("hidden_dims", [[128, 64, 32], [256, 128, 64], [64, 32]])
    soft_update_tau = trial.suggest_float("soft_update_tau", 0.01, 1.0)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)

    Q_network_DoubleDQN = VanillaQValueNetwork(
        state_dim=env.observation_space.shape[0],
        action_dim=num_actions,
        hidden_dims=hidden_dims,
        output_dim=1,
    )

    DoubleDQNagent = PearlAgent(
        policy_learner=DoubleDQN(
            state_dim=env.observation_space.shape[0],
            action_space=env.action_space,
            batch_size=64,
            training_rounds=10,
            soft_update_tau=soft_update_tau,
            network_instance=Q_network_DoubleDQN,
            action_representation_module=OneHotActionTensorRepresentationModule(
                max_number_actions=num_actions
            ),
            learning_rate=learning_rate,
        ),
        replay_buffer=BasicReplayBuffer(10_000),
    )

    info = online_learning(
        agent=DoubleDQNagent,
        env=env,
        number_of_episodes=240,
        print_every_x_episodes=20,
        learn_after_episode=True,
        seed=0,
    )

    return np.mean(info["return"])  

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)

best_params = study.best_params

set_seed(216028)

env = GymEnvironment("Acrobot-v1")
num_actions = env.action_space.n

Q_network_DoubleDQN = VanillaQValueNetwork(
    state_dim=env.observation_space.shape[0],
    action_dim=num_actions,
    hidden_dims=best_params["hidden_dims"],
    output_dim=1,
)

DoubleDQNagent = PearlAgent(
    policy_learner=DoubleDQN(
        state_dim=env.observation_space.shape[0],
        action_space=env.action_space,
        batch_size=64,
        training_rounds=10,
        soft_update_tau=best_params["soft_update_tau"],
        network_instance=Q_network_DoubleDQN,
        action_representation_module=OneHotActionTensorRepresentationModule(
            max_number_actions=num_actions
        ),
        learning_rate=best_params["learning_rate"],
    ),
    replay_buffer=BasicReplayBuffer(10_000),
)

info_DoubleDQN = online_learning(
    agent=DoubleDQNagent,
    env=env,
    number_of_episodes=500,
    print_every_x_episodes=20,
    learn_after_episode=True,
    seed=0,
)

NUM_EPISODES = 10
cumulative_return = 0

for i in range(NUM_EPISODES):

    env = GymEnvironment('Acrobot-v1', render_mode='human')
    observation, action_space = env.reset()
    DoubleDQNagent.reset(observation, action_space)
    done = False
    cumulative_reward = 0

    while not done:

        action = DoubleDQNagent.act(exploit=True)
        action_result = env.step(action)
        env.render()
        observation = action_result.observation
        reward = action_result.reward
        done = action_result.done
        DoubleDQNagent.observe(action_result)
        cumulative_reward += reward

    cumulative_return += cumulative_reward

print(f"Average return for DDQN: {cumulative_return / NUM_EPISODES}")
env.close()

torch.save(info_DoubleDQN["return"], "DoubleDQN-return.pt")
plt.plot(np.arange(len(info_DoubleDQN["return"])), info_DoubleDQN["return"], label="DoubleDQN")
plt.title("Episodic returns")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend()
plt.show()
