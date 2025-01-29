import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


env = make_vec_env("Acrobot-v1")

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

policy_kwargs = dict(
    net_arch=dict(pi=[128, 128], vf=[128, 128]) 
    )

model = PPO(
    "MlpPolicy",         
    env,                 
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4, 
    n_steps=2048,       
    batch_size=64,      
    n_epochs=10,      
    gamma=0.99,       
    gae_lambda=0.95,   
    clip_range=0.2,    
    verbose=1,          
    device='cpu'
)

episode_rewards=[]


eval_callback = EvalCallback(
    env,
    eval_freq=500,             
    best_model_save_path="./ppo_acrobot/",
    verbose=1
)

all_rewards = []
total_episodes = 0 
while total_episodes < 500:
    timesteps = 2500
    history = model.learn(total_timesteps=timesteps, callback=eval_callback)
    
    rewards = [info['r'] for info in history.ep_info_buffer]  
    
    total_episodes += len(rewards)  


model.save("ppo_acrobot_final")
model = PPO.load("./ppo_acrobot/best_model")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} ± {std_reward}")




all_rewards_flat = []
for reward_list in all_rewards:
    for num in reward_list:
        if len(all_rewards_flat) < 500:
            all_rewards_flat.append(num)
        else:
            break  


plt.plot(np.arange((len(all_rewards_flat))), all_rewards_flat,  label="PPO")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Episodic returns")
plt.legend()
plt.grid()
plt.show()
