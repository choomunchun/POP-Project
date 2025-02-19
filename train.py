import os
import random
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from datetime import datetime

# Load dataset
file_path = 'message.csv'
data = pd.read_csv(file_path)
feedback_file = 'user_feedback.csv'
model_dir = 'model_checkpoints'
log_dir = 'logs'

# Create necessary directories
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Ensure feedback file exists
if not os.path.exists(feedback_file):
    pd.DataFrame(columns=['message', 'persuasive_type', 'activity', 'user_label', 'timestamp']).to_csv(feedback_file,
                                                                                                       index=False)


# Define Persuasion Environment
class PersuasionEnv(gym.Env):
    def __init__(self, dataset, use_feedback=False):
        super(PersuasionEnv, self).__init__()
        self.data = dataset.sample(frac=1).reset_index(drop=True)
        self.index = 0
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.use_feedback = use_feedback
        self.rewards_history = []

        # Load feedback if available and requested
        if use_feedback and os.path.exists(feedback_file) and os.path.getsize(feedback_file) > 0:
            self.feedback_data = pd.read_csv(feedback_file)
            if not self.feedback_data.empty:
                print(f"Loaded {len(self.feedback_data)} feedback entries for training")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return self.get_observation(), {}

    def get_observation(self):
        row = self.data.iloc[self.index]
        msg_idx = hash(str(row['message'])) % 1000 / 1000
        ptype_idx = hash(str(row['persuasive_type'])) % 1000 / 1000
        act_idx = hash(str(row['activity'])) % 1000 / 1000
        return np.array([msg_idx, ptype_idx, act_idx], dtype=np.float32)

    def step(self, action):
        row = self.data.iloc[self.index]

        # Determine actual label from feedback if available
        if self.use_feedback and hasattr(self, 'feedback_data'):
            # Try to find this message in feedback
            matched_feedback = self.feedback_data[
                (self.feedback_data['message'] == row['message']) &
                (self.feedback_data['persuasive_type'] == row['persuasive_type']) &
                (self.feedback_data['activity'] == row['activity'])
                ]

            if not matched_feedback.empty:
                actual_label = matched_feedback.iloc[0]['user_label']
            else:
                actual_label = row.get('label', random.choice([0, 1]))
        else:
            actual_label = row.get('label', random.choice([0, 1]))

        reward = 1 if action == actual_label else -1
        self.rewards_history.append(reward)

        self.index += 1
        done = self.index >= len(self.data)
        truncated = False

        if done:
            next_obs, _ = self.reset()
        else:
            next_obs = self.get_observation()

        return next_obs, reward, done, truncated, {}


# Initialize PPO Model
def initialize_ppo(use_feedback=False):
    env = DummyVecEnv([lambda: PersuasionEnv(data, use_feedback=use_feedback)])

    # Set up callbacks for evaluation and checkpoints
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{model_dir}/best_model",
        log_path=log_dir,
        eval_freq=500,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=model_dir,
        name_prefix="ppo_persuasion_model"
    )

    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        tensorboard_log=log_dir
    )

    return model, env, [eval_callback, checkpoint_callback]


# Function to evaluate model on feedback data
def evaluate_on_feedback(model, feedback_data):
    if feedback_data.empty:
        return 0

    correct = 0
    for _, row in feedback_data.iterrows():
        obs = np.array([[
            hash(str(row['message'])) % 1000 / 1000,
            hash(str(row['persuasive_type'])) % 1000 / 1000,
            hash(str(row['activity'])) % 1000 / 1000
        ]], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)
        if action[0] == row['user_label']:
            correct += 1

    return correct / len(feedback_data) if len(feedback_data) > 0 else 0


# Function to plot learning progress
def plot_learning_progress(accuracies, rewards_history=None):
    plt.figure(figsize=(12, 6))

    # Plot accuracies
    plt.subplot(1, 2 if rewards_history else 1, 1)
    plt.plot(accuracies, 'b-')
    plt.xlabel('Training Iterations')
    plt.ylabel('Accuracy on Feedback Data')
    plt.title('Model Accuracy Progress')
    plt.grid(True)

    # Plot rewards if available
    if rewards_history:
        plt.subplot(1, 2, 2)
        plt.plot(rewards_history, 'r-')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.title('Reward History')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{log_dir}/learning_progress.png')
    print(f"Progress plot saved to {log_dir}/learning_progress.png")


# Main Interactive Evaluation
def main():
    print("Persuasion Message Evaluation with PPO")

    # Check if we have an existing model
    latest_model_path = "persuasion_ppo_model.zip"
    if os.path.exists(latest_model_path):
        print("Loading existing model...")
        ppo_model = PPO.load(latest_model_path)
        # Create env for prediction
        env = DummyVecEnv([lambda: PersuasionEnv(data)])
    else:
        print("Creating new model...")
        ppo_model, env, _ = initialize_ppo()

    # Load feedback data if exists
    feedback_data = pd.DataFrame()
    if os.path.exists(feedback_file) and os.path.getsize(feedback_file) > 0:
        feedback_data = pd.read_csv(feedback_file)
        print(f"Loaded {len(feedback_data)} feedback entries.")

    # Track learning progress
    accuracies = []
    training_iterations = 0

    while True:
        msg_idx = random.randint(0, len(data) - 1)
        msg = data.iloc[msg_idx]
        print(f"\nMessage: {msg['message']}")
        print(f"Persuasive Type: {msg['persuasive_type']}")
        print(f"Activity: {msg['activity']}")

        obs = np.array([[
            hash(str(msg['message'])) % 1000 / 1000,
            hash(str(msg['persuasive_type'])) % 1000 / 1000,
            hash(str(msg['activity'])) % 1000 / 1000
        ]], dtype=np.float32)

        action, _ = ppo_model.predict(obs)
        print(f"Model predicts: {'Persuasive' if action[0] == 1 else 'Not Persuasive'}")

        user_input = input("Is this persuasive? (1 for Yes, 0 for No, q to quit, t to train): ")

        if user_input.lower() == 'q':
            break
        elif user_input.lower() == 't':
            # Train model on all feedback
            if not feedback_data.empty:
                print(f"Training model on {len(feedback_data)} feedback entries...")

                # Create a new model with feedback
                new_model, train_env, callbacks = initialize_ppo(use_feedback=True)

                # Train and evaluate
                new_model.learn(total_timesteps=5000, callback=callbacks)

                # Evaluate and record accuracy
                accuracy = evaluate_on_feedback(new_model, feedback_data)
                accuracies.append(accuracy)
                training_iterations += 1
                print(f"Training iteration {training_iterations}, Accuracy: {accuracy:.2f}")

                # Save the trained model
                new_model.save(latest_model_path)
                ppo_model = new_model

                # Get reward history from environment for plotting
                rewards_history = train_env.get_attr('rewards_history')[0]

                # Plot progress
                plot_learning_progress(accuracies, rewards_history)

                print("Model trained and saved successfully!")
            else:
                print("No feedback data available for training!")
        else:
            try:
                user_label = int(user_input)

                # Save feedback with timestamp
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                feedback_df = pd.DataFrame.from_records([{
                    'message': msg['message'],
                    'persuasive_type': msg['persuasive_type'],
                    'activity': msg['activity'],
                    'user_label': user_label,
                    'timestamp': timestamp
                }])

                # Load existing feedback if any
                if os.path.exists(feedback_file) and os.path.getsize(feedback_file) > 0:
                    existing_feedback = pd.read_csv(feedback_file)
                    feedback_df = pd.concat([existing_feedback, feedback_df], ignore_index=True)

                # Save combined feedback
                feedback_df.to_csv(feedback_file, index=False)
                feedback_data = feedback_df  # Update in-memory feedback data
                print("User feedback saved!")

            except ValueError:
                print("Invalid input. Please enter 1, 0, q, or t.")

    # Final evaluation and plotting
    if training_iterations > 0:
        print("Final model evaluation:")
        final_accuracy = evaluate_on_feedback(ppo_model, feedback_data)
        accuracies.append(final_accuracy)
        plot_learning_progress(accuracies)
        print(f"Final Accuracy: {final_accuracy:.2f}")

    print("Session complete. Feedback collected in 'user_feedback.csv'")


if __name__ == "__main__":
    main()