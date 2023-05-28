import os
import sys
import gym
import time


# print(f"Environment List : {gym.envs.registry.all()}")


from pathlib import Path

import tensorflow as tf
from tensorflow.keras import models


pwd = Path(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd.parent, "PolicyNetwork"))

class Environment:
    def prepareEnvironment(self):
        self.env = gym.make("CartPole-v1")

    def PrintEnvironmentParameters(self):
        print(f"Observations Space : {self.env.observation_space}")
        print(f"Action Space : {self.env.action_space}")

    def PrintStatistics(self, value):
        import numpy as np
        print(f"--------------------------------------")
        print(f"Mean Policy Reward {np.mean(value)} ")
        print(f"STD Policy Reward {np.std(value)} ")
        print(f"Min Policy Reward {np.min(value)} ")
        print(f"Max Policy Reward {np.max(value)} ")
        print(f"--------------------------------------")

    def Policy(self, obs):
        angle = obs[2]
        return 0 if angle < 0 else 1

    def NNPolicy(self, obs):
        from NNPolicy import NNPolicy
        if not hasattr(self, 'model') or self.model is None:
            self.nnPolicy = NNPolicy(_ObservationCount = self.env.observation_space.shape[0])
            self.model = self.nnPolicy.BuildPolicyNetwork()

        action, grad = self.nnPolicy.train_one_step(obs, self.model)
        return action, grad

    def RunEpisodes(self, n_iterations ,iteration, n_episodes, n_max_Steps, print_episode_reward = False):
        import numpy as np
        all_grads = []
        all_rewards = []

        aggregated_rewards = []
        
        for i_episode in range(n_episodes):
            step_obs = self.env.reset()

            step_obs = step_obs[0]

            current_grads = []
            current_rewards = []

            for t in range(n_max_Steps):


                self.env.render()
                action, grad = self.NNPolicy(step_obs)
                if action[0, 0].numpy(): 
                    action = 1 # Non-Zero(Move right) is true 
                else:
                    action = 0 # Zero(Move left) is false 
                step_obs, step_rew, done, info, _ = self.env.step(action) 
                current_grads.append(grad)
                current_rewards.append(step_rew)


                console_string = f"Iteration : {iteration+1}/{n_iterations} -> Episode : {i_episode+1}/{n_episodes} -> Step : {t}/{n_max_Steps} -> Rewards : {np.sum(current_rewards)}"

                print('\x1b[2K\r', end='\r')
                tf.print(console_string, end='\r')

                # print(f"Observations : {obs}")
                if done and print_episode_reward:
                    # print(f"--------------------------------------")
                    # print(f"Episode {i_episode} finished after {t+1} timesteps")
                    # print(f"Episode Rewards {np.sum(current_rewards)}")
                    # print(f"--------------------------------------")
                    break
                # time.sleep(0.02)

            all_grads.append(current_grads)
            all_rewards.append(current_rewards)

            import numpy as np
            aggregated_rewards.append(np.sum(current_rewards))

        if print_episode_reward:
            print(np.array(all_rewards).ndim)
            self.PrintStatistics(aggregated_rewards)

        return all_rewards, all_grads


    def discount_rewards(self, rewards, discount_factor):
        import numpy as np
        discounted = np.array(rewards)
        for step in range(len(rewards) -2 , -1, -1):
            discounted[step] += discounted[step+1] * discount_factor

        return discounted 

    def discount_and_normalize_reawards(self, rewards, discount_factor):
        import numpy as np
        all_discounted_rewards = [self.discount_rewards(reward, discount_factor)
            for reward in rewards
        ]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean  = flat_rewards.mean()
        reward_std = flat_rewards.std()

        return [(discounted_rewards - reward_mean)/reward_std 
                for discounted_rewards in all_discounted_rewards] 


    def train_policy(self, n_iterations, n_episodes_per_iteration, n_max_steps, discount_factor):
        for interation in range(n_iterations):
            all_rewards, all_grads = self.RunEpisodes(n_iterations = n_iterations, iteration = interation, n_episodes = n_episodes_per_iteration, 
            n_max_Steps = n_max_steps,
            print_episode_reward = True)
            all_final_rewards = self.discount_and_normalize_reawards(all_rewards, discount_factor)

            all_mean_grads = []
            for var_index in range(len(self.model.trainable_variables)):
                mean_grad = tf.reduce_mean(
                    [final_reward * all_grads[episode_index][step][var_index]

                    for episode_index, final_rewards in enumerate(all_final_rewards)  
                    
                        for step, final_reward in enumerate(final_rewards)] , axis=0)
                all_mean_grads.append(mean_grad)
        
            self.nnPolicy.optimizer.apply_gradients(zip(all_mean_grads, self.model.trainable_variables))

    def CloseEnvironment(self):
        self.env.close()


gym_Ev = Environment()
gym_Ev.prepareEnvironment()
gym_Ev.PrintEnvironmentParameters()
gym_Ev.train_policy(n_iterations = 150 ,
                    n_episodes_per_iteration = 10, 
                    n_max_steps = 200, 
                    discount_factor = 0.95)

gym_Ev.RunEpisodes(10, 200, 20, 199)
gym_Ev.CloseEnvironment()




# print(gym_Ev.discount_rewards([10, 0, -50], discount_factor=0.8))
# print(gym_Ev.discount_and_normalize_reawards([[10, 0, -50], [10, 20]], discount_factor=0.8))