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
import warnings

warnings.filterwarnings('ignore')


class Box:
    def __init__(self, size, position=None):
        self.original_size = size  # Store original dimensions
        self.size = size  # Current dimensions (after rotation)
        self.position = position

    def set_position(self, pos):
        self.position = pos

    def rotate(self, rotation_type):
        """Apply rotation to the box"""
        l, w, h = self.original_size
        if rotation_type == 0:  # No rotation
            self.size = (l, w, h)
        elif rotation_type == 1:  # Rotate around z-axis
            self.size = (w, l, h)
        elif rotation_type == 2:  # Rotate around y-axis
            self.size = (h, w, l)
        elif rotation_type == 3:  # Rotate around x-axis
            self.size = (l, h, w)
        elif rotation_type == 4:  # Combined rotation 1
            self.size = (w, h, l)
        elif rotation_type == 5:  # Combined rotation 2
            self.size = (h, l, w)

    def get_bounds(self):
        x, y, z = self.position
        l, w, h = self.size
        return (x, x + l), (y, y + w), (z, z + h)

    def get_volume(self):
        return self.size[0] * self.size[1] * self.size[2]


class Enhanced3DBoxEnv(gym.Env):
    def __init__(self):
        super(Enhanced3DBoxEnv, self).__init__()
        self.pallet_size = (5, 5, 5)
        self.n_boxes = 100

        # Enhanced observation space: pallet + current box info + height map
        pallet_dims = self.pallet_size[0] * self.pallet_size[1] * self.pallet_size[2]
        height_map_dims = self.pallet_size[0] * self.pallet_size[1]

        self.observation_space = spaces.Box(
            low=0, high=5,
            shape=(pallet_dims + height_map_dims + 6,),
            dtype=np.float32
        )

        # Enhanced action space: position (25) + rotation (6) = 150 total actions
        n_positions = self.pallet_size[0] * self.pallet_size[1]
        n_rotations = 6
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

        return self._get_obs(), {}

    def _get_height_map(self):
        """Get height map of the current state"""
        height_map = np.zeros((self.pallet_size[0], self.pallet_size[1]))
        for x in range(self.pallet_size[0]):
            for y in range(self.pallet_size[1]):
                for z in range(self.pallet_size[2] - 1, -1, -1):
                    if self.occupied[x, y, z] == 1:
                        height_map[x, y] = z + 1
                        break
        return height_map

    def _get_obs(self):
        flat_occupied = self.occupied.flatten().astype(np.float32)
        height_map = self._get_height_map().flatten().astype(np.float32)

        if self.current_box_idx < self.n_boxes:
            box = self.box_queue[self.current_box_idx]
            box_info = np.array(list(box.original_size) + [self.current_box_idx,
                                                           self.failed_placements,
                                                           self.total_volume_placed], dtype=np.float32)
        else:
            box_info = np.array([0, 0, 0, self.n_boxes, self.failed_placements,
                                 self.total_volume_placed], dtype=np.float32)

        return np.concatenate([flat_occupied, height_map, box_info])

    def _decode_action(self, action):
        """Decode action into position and rotation"""
        n_positions = self.pallet_size[0] * self.pallet_size[1]
        position_idx = action % n_positions
        rotation_idx = action // n_positions

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
        best_reward = -10  # Penalty for failed placement

        for z in range(self.pallet_size[2]):
            box.set_position((x, y, z))
            if self.is_valid_placement(box):
                self.placed_boxes.append(box)
                (x1, x2), (y1, y2), (z1, z2) = box.get_bounds()
                self.occupied[x1:x2, y1:y2, z1:z2] = 1

                # Enhanced reward function
                volume_reward = box.get_volume() * 10
                height_penalty = -z * 0.5  # Encourage lower placement
                efficiency_bonus = 5 if z == 0 else 0  # Bonus for ground placement

                best_reward = volume_reward + height_penalty + efficiency_bonus
                self.total_volume_placed += box.get_volume()
                placed = True
                break

        if not placed:
            self.failed_placements += 1

        self.current_box_idx += 1
        done = self.current_box_idx >= self.n_boxes

        return self._get_obs(), best_reward, done, False, {
            'placed': placed,
            'volume': box.get_volume() if placed else 0,
            'total_volume': self.total_volume_placed,
            'failed_placements': self.failed_placements
        }

    def render(self, title="Box Placement"):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

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
        ax.set_title(f'{title}\nBoxes: {len(self.placed_boxes)}, Volume: {self.total_volume_placed}')
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

        box_collection = Poly3DCollection(faces, linewidths=0.5, edgecolors='black', alpha=0.7)
        box_collection.set_facecolor(color)
        ax.add_collection3d(box_collection)


class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.volumes_placed = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        # Collect metrics from the environment
        if hasattr(self.training_env, 'get_attr'):
            infos = self.training_env.get_attr('total_volume_placed')
            if infos:
                self.volumes_placed.extend(infos)


def run_experiment_with_network_config(neurons_per_layer, experiment_id, total_timesteps=50000):
    """Run a single experiment with specific network configuration"""
    print(f"\n--- Experiment {experiment_id}: {neurons_per_layer} neurons per layer ---")

    # Create environment
    env = Enhanced3DBoxEnv()
    vec_env = DummyVecEnv([lambda: Enhanced3DBoxEnv()])

    # Create DQN model with custom network
    policy_kwargs = {
        'net_arch': [neurons_per_layer, neurons_per_layer, neurons_per_layer]
    }

    model = DQN(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.001,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=0
    )

    # Training metrics
    start_time = time.time()
    callback = MetricsCallback()

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callback)
    training_time = time.time() - start_time

    return model, training_time, callback


def evaluate_model(model, n_episodes=10, render_one=False):
    """Evaluate trained model"""
    results = {
        'volumes': [],
        'success_rates': [],
        'decision_times': [],
        'boxes_placed': [],
        'failed_placements': []
    }

    for episode in range(n_episodes):
        env = Enhanced3DBoxEnv()
        obs, _ = env.reset(seed=episode)

        total_volume = 0
        boxes_placed = 0
        failed_placements = 0
        episode_time = 0

        for step in range(env.n_boxes):
            start_time = time.time()
            action, _ = model.predict(obs, deterministic=True)
            decision_time = time.time() - start_time
            episode_time += decision_time

            obs, reward, done, _, info = env.step(action)

            if info['placed']:
                boxes_placed += 1
                total_volume += info['volume']
            else:
                failed_placements += 1

            if done:
                break

        success_rate = boxes_placed / env.n_boxes

        results['volumes'].append(total_volume)
        results['success_rates'].append(success_rate)
        results['decision_times'].append(episode_time / env.n_boxes)
        results['boxes_placed'].append(boxes_placed)
        results['failed_placements'].append(failed_placements)

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

    print("Starting comprehensive DQN experiments...")
    print("=" * 60)

    for i, neurons in enumerate(network_configs):
        # Train model
        model, train_time, callback = run_experiment_with_network_config(
            neurons, i + 1, total_timesteps=30000  # Reduced for time constraint
        )

        training_times[neurons] = train_time

        # Evaluate model
        print(f"Evaluating model with {neurons} neurons...")
        results = evaluate_model(model, n_episodes=10, render_one=(i == 2))  # Render middle config

        all_results[neurons] = results

        # Print summary
        avg_volume = np.mean(results['volumes'])
        avg_success = np.mean(results['success_rates'])
        avg_time = np.mean(results['decision_times'])

        print(f"Results for {neurons} neurons:")
        print(f"  Average Volume: {avg_volume:.2f}")
        print(f"  Average Success Rate: {avg_success:.3f}")
        print(f"  Average Decision Time: {avg_time:.6f}s")
        print(f"  Training Time: {train_time:.2f}s")
        print("-" * 40)

    return all_results, training_times


def create_comparison_plots(results, training_times):
    """Create comprehensive comparison plots"""

    configs = list(results.keys())

    # Prepare data for plotting
    avg_volumes = [np.mean(results[config]['volumes']) for config in configs]
    std_volumes = [np.std(results[config]['volumes']) for config in configs]

    avg_success_rates = [np.mean(results[config]['success_rates']) for config in configs]
    std_success_rates = [np.std(results[config]['success_rates']) for config in configs]

    avg_decision_times = [np.mean(results[config]['decision_times']) for config in configs]
    std_decision_times = [np.std(results[config]['decision_times']) for config in configs]

    train_times = [training_times[config] for config in configs]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DQN Performance Analysis Across Network Configurations', fontsize=16)

    # Volume comparison
    axes[0, 0].errorbar(configs, avg_volumes, yerr=std_volumes, marker='o', capsize=5)
    axes[0, 0].set_xlabel('Neurons per Layer')
    axes[0, 0].set_ylabel('Average Volume Placed')
    axes[0, 0].set_title('Volume Efficiency')
    axes[0, 0].grid(True, alpha=0.3)

    # Success rate comparison
    axes[0, 1].errorbar(configs, avg_success_rates, yerr=std_success_rates, marker='s', color='green', capsize=5)
    axes[0, 1].set_xlabel('Neurons per Layer')
    axes[0, 1].set_ylabel('Average Success Rate')
    axes[0, 1].set_title('Placement Success Rate')
    axes[0, 1].grid(True, alpha=0.3)

    # Decision time comparison
    axes[1, 0].errorbar(configs, avg_decision_times, yerr=std_decision_times, marker='^', color='red', capsize=5)
    axes[1, 0].set_xlabel('Neurons per Layer')
    axes[1, 0].set_ylabel('Average Decision Time (s)')
    axes[1, 0].set_title('Decision Speed')
    axes[1, 0].grid(True, alpha=0.3)

    # Training time comparison
    axes[1, 1].bar(range(len(configs)), train_times, color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Network Configuration')
    axes[1, 1].set_ylabel('Training Time (s)')
    axes[1, 1].set_title('Training Efficiency')
    axes[1, 1].set_xticks(range(len(configs)))
    axes[1, 1].set_xticklabels([f'{c} neurons' for c in configs], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Create detailed results table
    print("\n" + "=" * 80)
    print("DETAILED RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Config':<12} {'Avg Volume':<12} {'Success Rate':<15} {'Decision Time':<15} {'Train Time':<12}")
    print("-" * 80)

    for i, config in enumerate(configs):
        print(f"{config:<12} {avg_volumes[i]:<12.2f} {avg_success_rates[i]:<15.3f} "
              f"{avg_decision_times[i]:<15.6f} {train_times[i]:<12.2f}")


def baseline_comparison():
    """Run baseline random algorithm for comparison"""
    print("\n" + "=" * 60)
    print("RUNNING BASELINE (RANDOM) COMPARISON")
    print("=" * 60)

    random_results = {
        'volumes': [],
        'success_rates': [],
        'decision_times': [],
        'boxes_placed': [],
        'failed_placements': []
    }

    for episode in range(10):
        env = Enhanced3DBoxEnv()
        obs, _ = env.reset(seed=episode)

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

        random_results['volumes'].append(total_volume)
        random_results['success_rates'].append(success_rate)
        random_results['boxes_placed'].append(boxes_placed)
        random_results['failed_placements'].append(failed_placements)

    print(f"Random Algorithm Results:")
    print(f"  Average Volume: {np.mean(random_results['volumes']):.2f}")
    print(f"  Average Success Rate: {np.mean(random_results['success_rates']):.3f}")
    print(f"  Average Boxes Placed: {np.mean(random_results['boxes_placed']):.1f}")

    return random_results


# Main execution
if __name__ == "__main__":
    print("3D Palletizing with DQN and Box Rotation")
    print("=" * 50)

    # Run baseline comparison
    baseline_results = baseline_comparison()

    # Run comprehensive experiments
    results, training_times = run_comprehensive_experiments()

    # Create visualizations
    create_comparison_plots(results, training_times)

    # Save results for report
    experiment_data = {
        'dqn_results': results,
        'training_times': training_times,
        'baseline_results': baseline_results
    }

    with open('dqn_experiment_results.pkl', 'wb') as f:
        pickle.dump(experiment_data, f)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED!")
    print("Results saved to 'dqn_experiment_results.pkl'")
    print("=" * 60)