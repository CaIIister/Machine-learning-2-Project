import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import torch
import torch.nn as nn
import time
import pickle
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import warnings

warnings.filterwarnings('ignore')

# Enable CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Box:
    def __init__(self, size, position=None):
        self.original_size = size
        self.size = size
        self.position = position

    def set_position(self, pos):
        self.position = pos

    def rotate(self, rotation_type):
        """Apply rotation - only 2 meaningful rotations to keep it simple"""
        l, w, h = self.original_size
        if rotation_type == 0:  # No rotation (original)
            self.size = (l, w, h)
        elif rotation_type == 1:  # Swap length and width
            self.size = (w, l, h)

    def get_bounds(self):
        x, y, z = self.position
        l, w, h = self.size
        return (x, x + l), (y, y + w), (z, z + h)

    def get_volume(self):
        return self.size[0] * self.size[1] * self.size[2]


class OptimizedPalletEnv(gym.Env):
    def __init__(self, enable_rotation=True):
        super(OptimizedPalletEnv, self).__init__()
        self.pallet_size = (5, 5, 5)
        self.n_boxes = 100
        self.enable_rotation = enable_rotation

        # Simplified observation: just the 3D grid + current box
        pallet_dims = self.pallet_size[0] * self.pallet_size[1] * self.pallet_size[2]

        self.observation_space = spaces.Box(
            low=0, high=2,
            shape=(pallet_dims + 3,),  # 125 + 3 = 128 features
            dtype=np.float32
        )

        # Simplified action space: 25 positions × 2 rotations = 50 actions
        n_positions = self.pallet_size[0] * self.pallet_size[1]
        n_rotations = 2 if enable_rotation else 1
        self.action_space = spaces.Discrete(n_positions * n_rotations)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.occupied = np.zeros(self.pallet_size, dtype=np.int32)
        self.placed_boxes = []
        self.current_box_idx = 0
        self.box_queue = [Box(tuple(random.choices([1, 2], k=3))) for _ in range(self.n_boxes)]
        self.total_volume_placed = 0
        self.successful_placements = 0

        return self._get_obs(), {}

    def _get_obs(self):
        # Just the 3D grid + current box dimensions
        flat_occupied = self.occupied.flatten().astype(np.float32)

        if self.current_box_idx < self.n_boxes:
            box = self.box_queue[self.current_box_idx]
            box_info = np.array(list(box.original_size), dtype=np.float32)
        else:
            box_info = np.array([0, 0, 0], dtype=np.float32)

        return np.concatenate([flat_occupied, box_info])

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

                # MUCH SIMPLER REWARD: Just give points for volume
                volume = box.get_volume()
                reward = volume * 10.0  # 10 points per unit volume

                self.total_volume_placed += volume
                self.successful_placements += 1
                placed = True
                break

        if not placed:
            # Small penalty for failed placement
            reward = -1.0

        self.current_box_idx += 1
        done = self.current_box_idx >= self.n_boxes

        return self._get_obs(), reward, done, False, {
            'placed': placed,
            'volume': box.get_volume() if placed else 0,
            'total_volume': self.total_volume_placed,
            'successful_placements': self.successful_placements
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

        ax.set_title(f'{title}\n'
                     f'Boxes: {self.successful_placements}/{self.current_box_idx} '
                     f'(Success: {success_rate:.1%})\n'
                     f'Volume: {self.total_volume_placed}/125 '
                     f'(Efficiency: {volume_efficiency:.1%})')
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


def create_final_dqn_model(env, neurons_per_layer):
    """Create DQN with aggressive hyperparameters for better learning"""

    policy_kwargs = {
        'net_arch': [neurons_per_layer, neurons_per_layer],  # Only 2 layers
        'activation_fn': torch.nn.ReLU,
    }

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,  # Higher learning rate
        buffer_size=100000,  # Large buffer
        learning_starts=10000,  # More exploration
        batch_size=128,  # Larger batches
        tau=1.0,
        gamma=0.95,  # Less focus on long-term rewards
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.5,  # Half the time exploring
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=0,
        device=device
    )

    return model


def quick_final_experiment():
    """Run a quick experiment focusing on the best configurations"""
    print("🎯 FINAL EXPERIMENT - FOCUSING ON BEST PERFORMING CONFIGS")
    print("=" * 60)

    # Test the three most promising approaches
    configs = [
        ("Baseline DQN (No Rotation)", False, 128),
        ("Optimized DQN (Simple Rotation)", True, 256),
        ("Optimized DQN (Simple Rotation)", True, 128)
    ]

    results = {}

    for name, rotation, neurons in configs:
        print(f"\n🚀 Testing: {name} ({neurons} neurons)")
        print("-" * 50)

        # Create environment
        env = OptimizedPalletEnv(enable_rotation=rotation)
        vec_env = DummyVecEnv([lambda: OptimizedPalletEnv(enable_rotation=rotation)])

        # Create model
        model = create_final_dqn_model(vec_env, neurons)

        # Train
        start_time = time.time()
        print("Training...")
        model.learn(total_timesteps=120000, progress_bar=True)  # More training
        training_time = time.time() - start_time

        # Evaluate
        print("Evaluating...")
        volumes = []
        success_rates = []

        for episode in range(15):
            test_env = OptimizedPalletEnv(enable_rotation=rotation)
            obs, _ = test_env.reset(seed=episode + 50)

            total_volume = 0
            boxes_placed = 0

            for step in range(test_env.n_boxes):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = test_env.step(action)

                if info['placed']:
                    boxes_placed += 1
                    total_volume += info['volume']

                if done:
                    break

            success_rate = boxes_placed / test_env.n_boxes
            volumes.append(total_volume)
            success_rates.append(success_rate)

            # Show one visualization
            if episode == 0:
                test_env.render(f"{name} - Episode {episode + 1}")

        avg_volume = np.mean(volumes)
        avg_success = np.mean(success_rates)

        results[name] = {
            'avg_volume': avg_volume,
            'std_volume': np.std(volumes),
            'avg_success': avg_success,
            'std_success': np.std(success_rates),
            'training_time': training_time,
            'neurons': neurons,
            'rotation': rotation
        }

        print(f"Results: Volume={avg_volume:.2f}±{np.std(volumes):.2f}, "
              f"Success={avg_success:.3f}±{np.std(success_rates):.3f}")
        print(f"Training time: {training_time:.1f}s")

    return results


def run_random_baseline():
    """Quick random baseline"""
    print("🎲 Random Baseline (15 episodes)")

    volumes = []
    success_rates = []

    for episode in range(15):
        env = OptimizedPalletEnv(enable_rotation=True)
        obs, _ = env.reset(seed=episode + 200)

        total_volume = 0
        boxes_placed = 0

        for step in range(env.n_boxes):
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)

            if info['placed']:
                boxes_placed += 1
                total_volume += info['volume']

            if done:
                break

        success_rate = boxes_placed / env.n_boxes
        volumes.append(total_volume)
        success_rates.append(success_rate)

    avg_volume = np.mean(volumes)
    avg_success = np.mean(success_rates)

    print(f"Random: Volume={avg_volume:.2f}±{np.std(volumes):.2f}, "
          f"Success={avg_success:.3f}±{np.std(success_rates):.3f}")

    return {
        'avg_volume': avg_volume,
        'std_volume': np.std(volumes),
        'avg_success': avg_success,
        'std_success': np.std(success_rates)
    }


def create_final_comparison_plot(results, random_results):
    """Create final comparison plot"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    methods = list(results.keys()) + ['Random']
    volumes = [results[method]['avg_volume'] for method in results.keys()] + [random_results['avg_volume']]
    volume_stds = [results[method]['std_volume'] for method in results.keys()] + [random_results['std_volume']]

    success_rates = [results[method]['avg_success'] for method in results.keys()] + [random_results['avg_success']]
    success_stds = [results[method]['std_success'] for method in results.keys()] + [random_results['std_success']]

    # Volume comparison
    bars1 = ax1.bar(range(len(methods)), volumes, yerr=volume_stds, capsize=5,
                    color=['skyblue', 'lightgreen', 'orange', 'gray'])
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Average Volume Placed')
    ax1.set_title('Volume Efficiency Comparison')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([m.replace('Optimized DQN (Simple Rotation)', 'DQN+Rot') for m in methods],
                        rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (v, std) in enumerate(zip(volumes, volume_stds)):
        ax1.text(i, v + std + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    # Success rate comparison
    bars2 = ax2.bar(range(len(methods)), success_rates, yerr=success_stds, capsize=5,
                    color=['skyblue', 'lightgreen', 'orange', 'gray'])
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Average Success Rate')
    ax2.set_title('Placement Success Rate Comparison')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels([m.replace('Optimized DQN (Simple Rotation)', 'DQN+Rot') for m in methods],
                        rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (v, std) in enumerate(zip(success_rates, success_stds)):
        ax2.text(i, v + std + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def print_final_summary(results, random_results):
    """Print final summary for report"""
    print("\n" + "=" * 80)
    print("🎯 FINAL RESULTS SUMMARY FOR YOUR REPORT")
    print("=" * 80)

    print(f"{'Method':<35} {'Volume':<12} {'Success Rate':<15} {'Improvement':<15}")
    print("-" * 80)

    baseline_volume = random_results['avg_volume']
    baseline_success = random_results['avg_success']

    print(f"{'Random Baseline':<35} {baseline_volume:<12.2f} {baseline_success:<15.3f} {'N/A':<15}")

    for method, data in results.items():
        volume_improvement = (data['avg_volume'] - baseline_volume) / baseline_volume * 100
        success_improvement = (data['avg_success'] - baseline_success) / baseline_success * 100

        print(f"{method[:34]:<35} {data['avg_volume']:<12.2f} {data['avg_success']:<15.3f} "
              f"{f'{volume_improvement:+6.1f}% vol':<15}")

    # Find best method
    best_method = max(results.keys(), key=lambda x: results[x]['avg_volume'])
    best_result = results[best_method]

    print(f"\n🏆 BEST PERFORMING METHOD: {best_method}")
    print(f"   Volume: {best_result['avg_volume']:.2f} (vs Random: {baseline_volume:.2f})")
    print(f"   Success Rate: {best_result['avg_success']:.3f} (vs Random: {baseline_success:.3f})")
    print(f"   Improvement: {(best_result['avg_volume'] - baseline_volume) / baseline_volume * 100:+.1f}% volume")
    print(f"   Network: {best_result['neurons']} neurons, Rotation: {best_result['rotation']}")


# Main execution
if __name__ == "__main__":
    print("🎯 FINAL OPTIMIZED 3D PALLETIZING EXPERIMENT")
    print("=" * 60)

    # Set seeds
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    # Run random baseline
    random_results = run_random_baseline()

    # Run final experiments
    results = quick_final_experiment()

    # Create comparison plot
    create_final_comparison_plot(results, random_results)

    # Print summary
    print_final_summary(results, random_results)

    # Save results
    final_data = {
        'dqn_results': results,
        'random_results': random_results,
        'device_used': str(device)
    }

    with open('final_dqn_results.pkl', 'wb') as f:
        pickle.dump(final_data, f)

    print(f"\n🎉 FINAL EXPERIMENT COMPLETED!")
    print(f"Results saved to 'final_dqn_results.pkl'")
    print(f"Training performed on: {device}")