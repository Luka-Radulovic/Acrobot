import optuna
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback

env = make_vec_env("Acrobot-v1")

# def optimize_ppo(trial):
#     learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
#     n_steps = trial.suggest_int('n_steps', 512, 4096, step=512)
#     batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
#     gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
#     gae_lambda = trial.suggest_uniform('gae_lambda', 0.8, 0.99)
#     clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)

#     net_arch = trial.suggest_categorical('net_arch', [[64, 64], [128, 128], [256, 256]])

#     policy_kwargs = dict(net_arch=dict(pi=net_arch, vf=net_arch))

#     model = PPO(
#         "MlpPolicy",
#         env,
#         policy_kwargs=policy_kwargs,
#         learning_rate=learning_rate,
#         n_steps=n_steps,
#         batch_size=batch_size,
#         gamma=gamma,
#         gae_lambda=gae_lambda,
#         clip_range=clip_range,
#         n_epochs=10,  
#         verbose=0,  
#         device='cpu'
#     )

#     eval_callback = EvalCallback(env, eval_freq=500, best_model_save_path="./ppo_acrobot/", verbose=0)

#     all_rewards = []
#     total_episodes = 0

#     while total_episodes < 500:
#         timesteps = 2500
#         history = model.learn(total_timesteps=timesteps, callback=eval_callback)
        
#         rewards = [info['r'] for info in history.ep_info_buffer]  
#         all_rewards.append(rewards)
        
#         total_episodes += len(rewards)

#     mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

#     return mean_reward 

# study = optuna.create_study(direction='maximize')
# study.optimize(optimize_ppo, n_trials=100)

# print("Best hyperparameters:", study.best_params)

best_params = {'learning_rate': 9.54948730910873e-05, 'n_steps': 2560, 'batch_size': 128, 'gamma': 0.9769979995374652, 'gae_lambda': 0.9536992151248987, 'clip_range': 0.2369607630932279, 'net_arch': [64, 64]}
policy_kwargs = dict(net_arch=dict(pi=best_params['net_arch'], vf=best_params['net_arch']))

best_model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=best_params['learning_rate'],
    n_steps=best_params['n_steps'],
    batch_size=best_params['batch_size'],
    gamma=best_params['gamma'],
    gae_lambda=best_params['gae_lambda'],
    clip_range=best_params['clip_range'],
    n_epochs=10,
    verbose=1,
    device='cpu'
)

all_rewards = []
total_episodes = 0
eval_callback = EvalCallback(env, eval_freq=500, best_model_save_path="./ppo_acrobot/", verbose=0)
while total_episodes < 500:
    timesteps = 2500
    history = best_model.learn(total_timesteps=timesteps, callback=eval_callback)
    
    rewards = [info['r'] for info in history.ep_info_buffer]
    all_rewards.append(rewards)
    
    total_episodes += len(rewards)

best_model.save("ppo_acrobot_optimized")

all_rewards_flat = [r for sublist in all_rewards for r in sublist][:500]

plt.plot(np.arange(len(all_rewards_flat)), all_rewards_flat, label="PPO (Optimized)")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Episodic Returns (Optimized PPO)")
plt.legend()
plt.grid()
plt.show()
