import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import time
import pickle
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import warnings
import os

warnings.filterwarnings('ignore')

# Enable CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


class Box:
    def __init__(self, size, position=None):
        self.original_size = size
        self.size = size
        self.position = position

    def set_position(self, pos):
        self.position = pos

    def rotate(self, rotation_type):
        """Apply rotation to the box - simplified to 3 main rotations"""
        l, w, h = self.original_size
        if rotation_type == 0:  # No rotation
            self.size = (l, w, h)
        elif rotation_type == 1:  # Rotate around z-axis
            self.size = (w, l, h)
        elif rotation_type == 2:  # Rotate around y-axis
            self.size = (h, w, l)

    def get_bounds(self):
        x, y, z = self.position
        l, w, h = self.size
        return (x, x + l), (y, y + w), (z, z + h)

    def get_volume(self):
        return self.size[0] * self.size[1] * self.size[2]


class ImprovedPalletEnv(gym.Env):
    def __init__(self, enable_rotation=True):
        super(ImprovedPalletEnv, self).__init__()
        self.pallet_size = (5, 5, 5)
        self.n_boxes = 100
        self.enable_rotation = enable_rotation

        # Simplified observation: 3D grid + current box info + placement stats
        pallet_dims = self.pallet_size[0] * self.pallet_size[1] * self.pallet_size[2]

        # Observation: flattened 3D grid + current box (3) + progress info (3) = 125 + 6 = 131
        self.observation_space = spaces.Box(
            low=0, high=10,
            shape=(pallet_dims + 6,),
            dtype=np.float32
        )

        # Simplified action space: position + rotation (if enabled)
        n_positions = self.pallet_size[0] * self.pallet_size[1]  # 25 positions
        n_rotations = 3 if enable_rotation else 1  # 3 rotations or just 1
        self.action_space = spaces.Discrete(n_positions * n_rotations)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.occupied = np.zeros(self.pallet_size, dtype=np.int32)
        self.placed_boxes = []
        self.current_box_idx = 0
        self.box_queue = [Box(tuple(random.choices([1, 2], k=3))) for _ in range(self.n_boxes)]
        self.total_volume_placed = 0
        self.failed_placements = 0
        self.successful_placements = 0

        return self._get_obs(), {}

    def _get_obs(self):
        # Flatten 3D grid
        flat_occupied = self.occupied.flatten().astype(np.float32)

        # Current box info
        if self.current_box_idx < self.n_boxes:
            box = self.box_queue[self.current_box_idx]
            box_info = np.array(list(box.original_size), dtype=np.float32)
        else:
            box_info = np.array([0, 0, 0], dtype=np.float32)

        # Progress info: current index, successful placements, total volume
        progress_info = np.array([
            self.current_box_idx / self.n_boxes,  # Progress ratio
            self.successful_placements / max(1, self.current_box_idx),  # Success rate so far
            self.total_volume_placed / 125.0  # Volume ratio (max possible = 5*5*5)
        ], dtype=np.float32)

        return np.concatenate([flat_occupied, box_info, progress_info])

    def _decode_action(self, action):
        """Decode action into position and rotation"""
        n_positions = self.pallet_size[0] * self.pallet_size[1]
        position_idx = action % n_positions
        rotation_idx = action // n_positions if self.enable_rotation else 0

        x = position_idx // self.pallet_size[1]
        y = position_idx % self.pallet_size[1]

        return x, y, rotation_idx

    def is_valid_placement(self, box):
        (x1, x2), (y1, y2), (z1, z2) = box.get_bounds()

        # Bounds check
        if x2 > self.pallet_size[0] or y2 > self.pallet_size[1] or z2 > self.pallet_size[2]:
            return False

        # Collision check
        if np.any(self.occupied[x1:x2, y1:y2, z1:z2] != 0):
            return False

        # Must sit on floor or on another box
        if z1 == 0:
            return True
        return np.all(self.occupied[x1:x2, y1:y2, z1 - 1:z1] == 1)

    def step(self, action):
        if self.current_box_idx >= self.n_boxes:
            return self._get_obs(), 0, True, False, {}

        box = self.box_queue[self.current_box_idx]
        x, y, rotation = self._decode_action(action)

        # Apply rotation
        box.rotate(rotation)

        # Try to place at the lowest possible z
        placed = False
        reward = 0

        for z in range(self.pallet_size[2]):
            box.set_position((x, y, z))
            if self.is_valid_placement(box):
                # Place the box
                self.placed_boxes.append(box)
                (x1, x2), (y1, y2), (z1, z2) = box.get_bounds()
                self.occupied[x1:x2, y1:y2, z1:z2] = 1

                # Improved reward function
                volume = box.get_volume()
                base_reward = volume * 2.0  # Base reward for volume
                height_bonus = max(0, (self.pallet_size[2] - z) * 0.5)  # Bonus for lower placement
                efficiency_bonus = 1.0  # Bonus for successful placement

                reward = base_reward + height_bonus + efficiency_bonus
                self.total_volume_placed += volume
                self.successful_placements += 1
                placed = True
                break

        if not placed:
            # Penalty for failed placement
            reward = -2.0
            self.failed_placements += 1

        self.current_box_idx += 1
        done = self.current_box_idx >= self.n_boxes

        # End-of-episode bonus
        if done and self.successful_placements > 0:
            efficiency_ratio = self.successful_placements / self.n_boxes
            volume_ratio = self.total_volume_placed / 125.0  # Max possible volume
            bonus = (efficiency_ratio + volume_ratio) * 10
            reward += bonus

        return self._get_obs(), reward, done, False, {
            'placed': placed,
            'volume': box.get_volume() if placed else 0,
            'total_volume': self.total_volume_placed,
            'successful_placements': self.successful_placements,
            'failed_placements': self.failed_placements
        }

    def render(self, title="Box Placement"):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        if len(self.placed_boxes) > 0:
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.placed_boxes)))

            for i, box in enumerate(self.placed_boxes):
                x, y, z = box.position
                l, w, h = box.size
                self._draw_box(ax, x, y, z, l, w, h, colors[i])

        ax.set_xlim(0, self.pallet_size[0])
        ax.set_ylim(0, self.pallet_size[1])
        ax.set_zlim(0, self.pallet_size[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        success_rate = self.successful_placements / max(1, self.current_box_idx)
        volume_efficiency = self.total_volume_placed / 125.0

        ax.set_title(f'{title}\nBoxes: {self.successful_placements}/{self.current_box_idx}, '
                     f'Volume: {self.total_volume_placed}, '
                     f'Success Rate: {success_rate:.2%}, '
                     f'Volume Efficiency: {volume_efficiency:.2%}')
        plt.tight_layout()
        plt.show()

    def _draw_box(self, ax, x, y, z, l, w, h, color):
        vertices = np.array([
            [x, y, z], [x + l, y, z], [x + l, y + w, z], [x, y + w, z],
            [x, y, z + h], [x + l, y, z + h], [x + l, y + w, z + h], [x, y + w, z + h]
        ])

        faces = [
            [vertices[j] for j in [0, 1, 2, 3]], [vertices[j] for j in [4, 5, 6, 7]],
            [vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [2, 3, 7, 6]],
            [vertices[j] for j in [1, 2, 6, 5]], [vertices[j] for j in [4, 7, 3, 0]]
        ]

        box_collection = Poly3DCollection(faces, linewidths=0.5, edgecolors='black', alpha=0.8)
        box_collection.set_facecolor(color)
        ax.add_collection3d(box_collection)


class AdvancedMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.volumes_placed = []
        self.training_step = 0

    def _on_step(self) -> bool:
        self.training_step += 1
        return True

    def _on_rollout_end(self):
        # Collect episode statistics
        if len(self.model.ep_info_buffer) > 0:
            recent_episode = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(recent_episode['r'])
            self.episode_lengths.append(recent_episode['l'])


def create_optimized_dqn_model(env, neurons_per_layer, device="auto"):
    """Create DQN model with optimized hyperparameters"""

    policy_kwargs = {
        'net_arch': [neurons_per_layer, neurons_per_layer, neurons_per_layer],
        'activation_fn': torch.nn.ReLU,
    }

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,  # Slightly higher learning rate
        buffer_size=50000,  # Larger buffer
        learning_starts=5000,  # More exploration before learning
        batch_size=64,  # Larger batch size
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2000,  # Less frequent target updates
        exploration_fraction=0.4,  # Longer exploration
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,  # Higher final exploration
        max_grad_norm=10,
        verbose=0,
        device=device,
        tensorboard_log="./dqn_tensorboard/"
    )

    return model


def run_experiment_with_network_config(neurons_per_layer, experiment_id, total_timesteps=100000, enable_rotation=True):
    """Run a single experiment with specific network configuration"""
    print(f"\n--- Experiment {experiment_id}: {neurons_per_layer} neurons per layer ---")
    print(f"Rotation enabled: {enable_rotation}")

    # Create environment with monitoring
    env = ImprovedPalletEnv(enable_rotation=enable_rotation)
    monitored_env = Monitor(env)
    vec_env = DummyVecEnv([lambda: monitored_env])

    # Create optimized DQN model
    model = create_optimized_dqn_model(vec_env, neurons_per_layer, device=device)

    # Training with callback
    start_time = time.time()
    callback = AdvancedMetricsCallback()

    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.2f}s")

    return model, training_time, callback


def evaluate_model_comprehensive(model, n_episodes=15, render_one=False, enable_rotation=True):
    """Comprehensive model evaluation"""
    results = {
        'volumes': [],
        'success_rates': [],
        'decision_times': [],
        'boxes_placed': [],
        'failed_placements': [],
        'volume_efficiency': [],
        'episode_rewards': []
    }

    print(f"Evaluating model over {n_episodes} episodes...")

    for episode in range(n_episodes):
        env = ImprovedPalletEnv(enable_rotation=enable_rotation)
        obs, _ = env.reset(seed=episode + 42)  # Fixed seeds for consistency

        total_volume = 0
        boxes_placed = 0
        failed_placements = 0
        episode_time = 0
        episode_reward = 0

        for step in range(env.n_boxes):
            start_time = time.time()
            action, _ = model.predict(obs, deterministic=True)
            decision_time = time.time() - start_time
            episode_time += decision_time

            obs, reward, done, _, info = env.step(action)
            episode_reward += reward

            if info['placed']:
                boxes_placed += 1
                total_volume += info['volume']
            else:
                failed_placements += 1

            if done:
                break

        success_rate = boxes_placed / env.n_boxes
        volume_efficiency = total_volume / 125.0  # Max possible volume

        results['volumes'].append(total_volume)
        results['success_rates'].append(success_rate)
        results['decision_times'].append(episode_time / env.n_boxes)
        results['boxes_placed'].append(boxes_placed)
        results['failed_placements'].append(failed_placements)
        results['volume_efficiency'].append(volume_efficiency)
        results['episode_rewards'].append(episode_reward)

        # Render one episode for visualization
        if render_one and episode == 0:
            env.render(f"DQN Result - Episode {episode + 1}")

    return results


def run_comprehensive_experiments():
    """Run experiments with different network configurations"""

    # Different network configurations
    network_configs = [32, 64, 128, 256, 512]

    all_results = {}
    training_times = {}

    print("Starting improved DQN experiments with CUDA acceleration...")
    print("=" * 70)

    # First test without rotation for baseline
    print("\n🔧 Running baseline DQN (no rotation)...")
    baseline_model, baseline_time, _ = run_experiment_with_network_config(
        128, "baseline", total_timesteps=80000, enable_rotation=False
    )
    baseline_results = evaluate_model_comprehensive(baseline_model, n_episodes=10, enable_rotation=False)

    print(f"Baseline DQN Results (no rotation):")
    print(f"  Average Volume: {np.mean(baseline_results['volumes']):.2f}")
    print(f"  Average Success Rate: {np.mean(baseline_results['success_rates']):.3f}")
    print(f"  Training Time: {baseline_time:.2f}s")

    # Now test with rotation and different network sizes
    for i, neurons in enumerate(network_configs):
        # Train model with rotation
        model, train_time, callback = run_experiment_with_network_config(
            neurons, i + 1, total_timesteps=80000, enable_rotation=True
        )

        training_times[neurons] = train_time

        # Evaluate model
        print(f"Evaluating model with {neurons} neurons...")
        results = evaluate_model_comprehensive(
            model, n_episodes=12, render_one=(i == 2), enable_rotation=True
        )

        all_results[neurons] = results

        # Print summary
        avg_volume = np.mean(results['volumes'])
        avg_success = np.mean(results['success_rates'])
        avg_efficiency = np.mean(results['volume_efficiency'])
        avg_time = np.mean(results['decision_times'])
        avg_reward = np.mean(results['episode_rewards'])

        print(f"Results for {neurons} neurons:")
        print(f"  Average Volume: {avg_volume:.2f}")
        print(f"  Average Success Rate: {avg_success:.3f}")
        print(f"  Volume Efficiency: {avg_efficiency:.3f}")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Decision Time: {avg_time:.6f}s")
        print(f"  Training Time: {train_time:.2f}s")
        print("-" * 50)

    return all_results, training_times, baseline_results


def create_comprehensive_plots(results, training_times, baseline_results):
    """Create comprehensive comparison plots"""

    configs = list(results.keys())

    # Prepare data for plotting
    avg_volumes = [np.mean(results[config]['volumes']) for config in configs]
    std_volumes = [np.std(results[config]['volumes']) for config in configs]

    avg_success_rates = [np.mean(results[config]['success_rates']) for config in configs]
    std_success_rates = [np.std(results[config]['success_rates']) for config in configs]

    avg_efficiency = [np.mean(results[config]['volume_efficiency']) for config in configs]
    std_efficiency = [np.std(results[config]['volume_efficiency']) for config in configs]

    avg_decision_times = [np.mean(results[config]['decision_times']) for config in configs]
    std_decision_times = [np.std(results[config]['decision_times']) for config in configs]

    avg_rewards = [np.mean(results[config]['episode_rewards']) for config in configs]

    train_times = [training_times[config] for config in configs]

    # Baseline comparisons
    baseline_volume = np.mean(baseline_results['volumes'])
    baseline_success = np.mean(baseline_results['success_rates'])
    baseline_efficiency = np.mean(baseline_results['volume_efficiency'])

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced DQN Performance Analysis (With Rotation vs Baseline)', fontsize=16)

    # Volume comparison
    axes[0, 0].errorbar(configs, avg_volumes, yerr=std_volumes, marker='o', capsize=5, label='With Rotation')
    axes[0, 0].axhline(y=baseline_volume, color='red', linestyle='--',
                       label=f'Baseline (No Rotation): {baseline_volume:.1f}')
    axes[0, 0].axhline(y=74.5, color='gray', linestyle=':', label='Random: 74.5')
    axes[0, 0].set_xlabel('Neurons per Layer')
    axes[0, 0].set_ylabel('Average Volume Placed')
    axes[0, 0].set_title('Volume Efficiency Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Success rate comparison
    axes[0, 1].errorbar(configs, avg_success_rates, yerr=std_success_rates, marker='s', color='green', capsize=5,
                        label='With Rotation')
    axes[0, 1].axhline(y=baseline_success, color='red', linestyle='--', label=f'Baseline: {baseline_success:.3f}')
    axes[0, 1].axhline(y=0.353, color='gray', linestyle=':', label='Random: 0.353')
    axes[0, 1].set_xlabel('Neurons per Layer')
    axes[0, 1].set_ylabel('Average Success Rate')
    axes[0, 1].set_title('Placement Success Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Volume efficiency
    axes[0, 2].errorbar(configs, avg_efficiency, yerr=std_efficiency, marker='^', color='purple', capsize=5,
                        label='With Rotation')
    axes[0, 2].axhline(y=baseline_efficiency, color='red', linestyle='--', label=f'Baseline: {baseline_efficiency:.3f}')
    axes[0, 2].axhline(y=0.596, color='gray', linestyle=':', label='Random: 0.596')
    axes[0, 2].set_xlabel('Neurons per Layer')
    axes[0, 2].set_ylabel('Volume Efficiency (fraction of max)')
    axes[0, 2].set_title('Space Utilization Efficiency')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Episode rewards
    axes[1, 0].plot(configs, avg_rewards, marker='d', color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Neurons per Layer')
    axes[1, 0].set_ylabel('Average Episode Reward')
    axes[1, 0].set_title('Learning Performance (Episode Rewards)')
    axes[1, 0].grid(True, alpha=0.3)

    # Decision time comparison
    axes[1, 1].errorbar(configs, avg_decision_times, yerr=std_decision_times, marker='*', color='red', capsize=5)
    axes[1, 1].set_xlabel('Neurons per Layer')
    axes[1, 1].set_ylabel('Average Decision Time (s)')
    axes[1, 1].set_title('Decision Speed')
    axes[1, 1].grid(True, alpha=0.3)

    # Training time comparison
    axes[1, 2].bar(range(len(configs)), train_times, color='brown', alpha=0.7)
    axes[1, 2].set_xlabel('Network Configuration')
    axes[1, 2].set_ylabel('Training Time (s)')
    axes[1, 2].set_title('Training Efficiency')
    axes[1, 2].set_xticks(range(len(configs)))
    axes[1, 2].set_xticklabels([f'{c}' for c in configs])
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Create detailed results table
    print("\n" + "=" * 100)
    print("DETAILED RESULTS SUMMARY")
    print("=" * 100)
    print(
        f"{'Config':<8} {'Volume':<10} {'Success':<10} {'Efficiency':<12} {'Reward':<10} {'Dec.Time':<12} {'Train.Time':<12}")
    print("-" * 100)

    for i, config in enumerate(configs):
        print(f"{config:<8} {avg_volumes[i]:<10.2f} {avg_success_rates[i]:<10.3f} "
              f"{avg_efficiency[i]:<12.3f} {avg_rewards[i]:<10.1f} {avg_decision_times[i]:<12.6f} {train_times[i]:<12.1f}")

    print(
        f"\nBaseline (No Rotation): Volume={baseline_volume:.2f}, Success={baseline_success:.3f}, Efficiency={baseline_efficiency:.3f}")
    print(f"Random Algorithm: Volume=74.5, Success=0.353, Efficiency=0.596")


def baseline_comparison():
    """Run baseline random algorithm for comparison"""
    print("\n" + "=" * 60)
    print("RUNNING BASELINE (RANDOM) COMPARISON")
    print("=" * 60)

    random_results = {
        'volumes': [],
        'success_rates': [],
        'boxes_placed': [],
        'failed_placements': [],
        'volume_efficiency': []
    }

    for episode in range(15):
        env = ImprovedPalletEnv(enable_rotation=True)
        obs, _ = env.reset(seed=episode + 100)

        total_volume = 0
        boxes_placed = 0
        failed_placements = 0

        for step in range(env.n_boxes):
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)

            if info['placed']:
                boxes_placed += 1
                total_volume += info['volume']
            else:
                failed_placements += 1

            if done:
                break

        success_rate = boxes_placed / env.n_boxes
        volume_efficiency = total_volume / 125.0

        random_results['volumes'].append(total_volume)
        random_results['success_rates'].append(success_rate)
        random_results['boxes_placed'].append(boxes_placed)
        random_results['failed_placements'].append(failed_placements)
        random_results['volume_efficiency'].append(volume_efficiency)

    print(f"Random Algorithm Results:")
    print(f"  Average Volume: {np.mean(random_results['volumes']):.2f}")
    print(f"  Average Success Rate: {np.mean(random_results['success_rates']):.3f}")
    print(f"  Average Volume Efficiency: {np.mean(random_results['volume_efficiency']):.3f}")
    print(f"  Average Boxes Placed: {np.mean(random_results['boxes_placed']):.1f}")

    return random_results


# Main execution
if __name__ == "__main__":
    print("🚀 Enhanced 3D Palletizing with DQN, Rotation, and CUDA")
    print("=" * 60)

    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Run baseline comparison
    baseline_results = baseline_comparison()

    # Run comprehensive experiments
    results, training_times, dqn_baseline = run_comprehensive_experiments()

    # Create visualizations
    create_comprehensive_plots(results, training_times, dqn_baseline)

    # Save results for report
    experiment_data = {
        'dqn_results': results,
        'training_times': training_times,
        'baseline_results': baseline_results,
        'dqn_baseline': dqn_baseline,
        'device_used': str(device)
    }

    with open('enhanced_dqn_results.pkl', 'wb') as f:
        pickle.dump(experiment_data, f)

    print("\n" + "=" * 60)
    print("🎉 ENHANCED EXPERIMENT COMPLETED!")
    print("Results saved to 'enhanced_dqn_results.pkl'")
    print(f"Training performed on: {device}")
    print("=" * 60)