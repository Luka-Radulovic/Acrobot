import torch
import numpy as np
import matplotlib.pyplot as plt
import optuna

from pearl.neural_networks.sequential_decision_making.q_value_networks import VanillaQValueNetwork
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.policy_learners.sequential_decision_making.deep_q_learning import DeepQLearning
from pearl.replay_buffers import BasicReplayBuffer
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.pearl_agent import PearlAgent
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)

set_seed(216028)

def objective(trial):
    hidden_dims = trial.suggest_categorical("hidden_dims", [[128, 64, 32], [256, 128, 64], [64, 32]])
    soft_update_tau = trial.suggest_float("soft_update_tau", 0.1, 1.0)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)

    env = GymEnvironment("Acrobot-v1")
    num_actions = env.action_space.n

    Q_value_network = VanillaQValueNetwork(
        state_dim=env.observation_space.shape[0],
        action_dim=num_actions,
        hidden_dims=hidden_dims,
        output_dim=1,
    )

    DQNagent = PearlAgent(
        policy_learner=DeepQLearning(
            state_dim=env.observation_space.shape[0],
            action_space=env.action_space,
            batch_size=64,
            training_rounds=10,
            soft_update_tau=soft_update_tau,
            network_instance=Q_value_network,
            action_representation_module=OneHotActionTensorRepresentationModule(
                max_number_actions=num_actions
            ),
            learning_rate=learning_rate,
        ),
        replay_buffer=BasicReplayBuffer(10_000),
    )

    info = online_learning(
        agent=DQNagent,
        env=env,
        number_of_episodes=240,  
        print_every_x_episodes=20,
        learn_after_episode=True,
        seed=0,
    )

    return np.mean(info["return"]) 

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)  

print("Best parameters:", study.best_params)
print("Best return:", study.best_value)

best_params = study.best_params
hidden_dims = best_params["hidden_dims"]
soft_update_tau = best_params["soft_update_tau"]
learning_rate = best_params["learning_rate"]

env = GymEnvironment("Acrobot-v1")
num_actions = env.action_space.n

Q_value_network = VanillaQValueNetwork(
    state_dim=env.observation_space.shape[0],
    action_dim=num_actions,
    hidden_dims=hidden_dims,
    output_dim=1,
)

DQNagent = PearlAgent(
    policy_learner=DeepQLearning(
        state_dim=env.observation_space.shape[0],
        action_space=env.action_space,
        batch_size=64,
        training_rounds=10,
        soft_update_tau=soft_update_tau,
        network_instance=Q_value_network,
        action_representation_module=OneHotActionTensorRepresentationModule(
            max_number_actions=num_actions
        ),
        learning_rate=learning_rate,
    ),
    replay_buffer=BasicReplayBuffer(10_000),
)

info = online_learning(
    agent=DQNagent,
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
    DQNagent.reset(observation, action_space)
    done = False
    cumulative_reward = 0 

    while not done:

        action = DQNagent.act(exploit=True)  
        action_result = env.step(action)
        env.render() 
        observation = action_result.observation
        reward = action_result.reward
        done = action_result.done
        DQNagent.observe(action_result)
        cumulative_reward += reward

    cumulative_return += cumulative_reward


print(f"Average return for DQN: {cumulative_return/NUM_EPISODES}")
env.close()

torch.save(info["return"], "DQN-return.pt")   
plt.plot(np.arange(len(info["return"])), info["return"], label="DQN")
plt.title("Episodic returns")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend()
plt.show()