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
import warnings

warnings.filterwarnings('ignore')

# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class Box:
    def __init__(self, size, position=None):
        self.original_size = size
        self.size = size
        self.position = position

    def set_position(self, pos):
        self.position = pos

    def get_bounds(self):
        x, y, z = self.position
        l, w, h = self.size
        return (x, x + l), (y, y + w), (z, z + h)

    def get_volume(self):
        return self.size[0] * self.size[1] * self.size[2]


class OptimizedPalletEnv(gym.Env):
    """Optimized 3D palletizing environment with improved reward shaping"""

    def __init__(self):
        super(OptimizedPalletEnv, self).__init__()
        self.pallet_size = (5, 5, 5)
        self.n_boxes = 100
        self.max_volume = 125  # 5*5*5

        # Enhanced observation space
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=1, shape=self.pallet_size, dtype=np.float32),
            'height_map': spaces.Box(low=0, high=self.pallet_size[2], shape=(self.pallet_size[0], self.pallet_size[1]), dtype=np.float32),
            'current_box': spaces.Box(low=1, high=2, shape=(3,), dtype=np.float32),
            'metrics': spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        })

        # Action space: 25 positions
        self.action_space = spaces.Discrete(self.pallet_size[0] * self.pallet_size[1])

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.occupied = np.zeros(self.pallet_size, dtype=np.int32)
        self.placed_boxes = []
        self.current_box_idx = 0
        self.box_queue = [Box(tuple(random.choices([1, 2], k=3))) for _ in range(self.n_boxes)]
        self.total_volume_placed = 0
        self.successful_placements = 0
        self.failed_placements = 0
        self.total_reward = 0

        return self._get_obs(), {}

    def _get_height_map(self):
        """Get 2D height map of current state"""
        height_map = np.zeros((self.pallet_size[0], self.pallet_size[1]))
        for x in range(self.pallet_size[0]):
            for y in range(self.pallet_size[1]):
                for z in range(self.pallet_size[2] - 1, -1, -1):
                    if self.occupied[x, y, z] == 1:
                        height_map[x, y] = z + 1
                        break
        return height_map

    def _get_obs(self):
        # Convert to proper observation format
        obs = {
            'grid': self.occupied.astype(np.float32),
            'height_map': self._get_height_map().astype(np.float32),
            'current_box': np.array(self.box_queue[self.current_box_idx].original_size, dtype=np.float32) if self.current_box_idx < self.n_boxes else np.zeros(3, dtype=np.float32),
            'metrics': np.array([
                self.current_box_idx / self.n_boxes,  # Progress
                self.successful_placements / max(1, self.current_box_idx),  # Success rate
                self.total_volume_placed / self.max_volume,  # Volume efficiency
                np.mean(self._get_height_map()) / self.pallet_size[2]  # Average height utilization
            ], dtype=np.float32)
        }
        
        # Flatten for the neural network
        return np.concatenate([
            obs['grid'].flatten(),
            obs['height_map'].flatten(),
            obs['current_box'],
            obs['metrics']
        ])

    def is_valid_placement(self, box):
        (x1, x2), (y1, y2), (z1, z2) = box.get_bounds()

        # Bounds check
        if x2 > self.pallet_size[0] or y2 > self.pallet_size[1] or z2 > self.pallet_size[2]:
            return False

        # Collision check
        if np.any(self.occupied[x1:x2, y1:y2, z1:z2] != 0):
            return False

        # Support check
        if z1 == 0:
            return True
        return np.all(self.occupied[x1:x2, y1:y2, z1 - 1:z1] == 1)

    def _calculate_advanced_reward(self, box, z, placed):
        """Simplified and more balanced reward function"""
        if not placed:
            return -1.0  # Reduced penalty for failed placement

        volume = box.get_volume()
        
        # Base reward for successful placement
        base_reward = 2.0
        
        # Volume-based reward (normalized)
        volume_reward = volume / 8.0  # Normalized by max possible box volume
        
        # Height efficiency reward (encourage efficient use of vertical space)
        height_efficiency = 1.0 - (z / self.pallet_size[2])
        height_reward = height_efficiency * 0.5
        
        # Stability reward
        stability_reward = 0.0
        if z == 0:  # Ground level placement
            stability_reward = 0.5
        else:
            # Check support from below
            x, y = box.position[0], box.position[1]
            l, w = box.size[0], box.size[1]
            support_area = np.sum(self.occupied[x:x+l, y:y+w, z-1]) / (l * w)
            stability_reward = support_area * 0.5

        total_reward = base_reward + volume_reward + height_reward + stability_reward
        return total_reward

    def step(self, action):
        if self.current_box_idx >= self.n_boxes:
            return self._get_obs(), 0, True, False, {}

        box = self.box_queue[self.current_box_idx]
        x = action // self.pallet_size[1]
        y = action % self.pallet_size[1]

        placed = False
        best_z = None

        # Find lowest valid placement
        for z in range(self.pallet_size[2]):
            box.set_position((x, y, z))
            if self.is_valid_placement(box):
                self.placed_boxes.append(box)
                (x1, x2), (y1, y2), (z1, z2) = box.get_bounds()
                self.occupied[x1:x2, y1:y2, z1:z2] = 1

                self.total_volume_placed += box.get_volume()
                self.successful_placements += 1
                placed = True
                best_z = z
                break

        if not placed:
            self.failed_placements += 1
            best_z = 0

        # Calculate reward
        reward = self._calculate_advanced_reward(box, best_z, placed)
        self.total_reward += reward

        self.current_box_idx += 1
        done = self.current_box_idx >= self.n_boxes

        # End-of-episode reward
        if done:
            final_success_rate = self.successful_placements / self.n_boxes
            final_volume_efficiency = self.total_volume_placed / self.max_volume

            # Bonus for high performance
            if final_success_rate > 0.5:
                reward += 50.0 * final_success_rate
            if final_volume_efficiency > 0.7:
                reward += 100.0 * final_volume_efficiency

            # Penalty for poor performance
            if final_success_rate < 0.3:
                reward -= 25.0

        return self._get_obs(), reward, done, False, {
            'placed': placed,
            'volume': box.get_volume() if placed else 0,
            'total_volume': self.total_volume_placed,
            'successful_placements': self.successful_placements,
            'failed_placements': self.failed_placements,
            'success_rate': self.successful_placements / max(1, self.current_box_idx),
            'volume_efficiency': self.total_volume_placed / self.max_volume
        }

    def render(self, title="3D Palletizing Result"):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        if len(self.placed_boxes) > 0:
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.placed_boxes)))

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
        volume_efficiency = self.total_volume_placed / self.max_volume

        ax.set_title(f'{title}\n'
                     f'Placed: {self.successful_placements}/{self.current_box_idx} boxes '
                     f'(Success: {success_rate:.1%})\n'
                     f'Volume: {self.total_volume_placed}/{self.max_volume} '
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


class LearningRateScheduler(BaseCallback):
    """Adaptive learning rate scheduler"""

    def __init__(self, initial_lr=1e-3, decay_factor=0.95, decay_interval=10000):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.decay_interval = decay_interval

    def _on_step(self) -> bool:
        if self.num_timesteps % self.decay_interval == 0:
            new_lr = self.initial_lr * (self.decay_factor ** (self.num_timesteps // self.decay_interval))
            self.model.policy.optimizer.param_groups[0]['lr'] = new_lr
        return True


def create_optimized_dqn_model(env, neurons_per_layer):
    """Create optimized DQN with improved hyperparameters"""
    
    # Deeper network architecture
    net_arch = [
        neurons_per_layer,
        neurons_per_layer,
        neurons_per_layer,
        neurons_per_layer // 2
    ]

    policy_kwargs = {
        'net_arch': net_arch,
        'activation_fn': torch.nn.ReLU,
        'normalize_images': False,
    }

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,  # Lower learning rate for stability
        buffer_size=500000,  # Larger buffer for better experience replay
        learning_starts=50000,  # More initial exploration
        batch_size=128,  # Smaller batch size for better generalization
        tau=0.001,  # Slower target updates
        gamma=0.98,  # Slightly lower discount factor
        train_freq=4,
        gradient_steps=4,  # More gradient steps per update
        target_update_interval=10000,
        exploration_fraction=0.3,  # Faster exploration decay
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,  # Slightly higher final exploration
        max_grad_norm=0.5,  # Lower gradient clipping
        verbose=0,
        device=device
    )

    return model


def evaluate_model_comprehensive(model, n_episodes=15):
    """Comprehensive model evaluation with detailed metrics"""
    results = {
        'volumes': [],
        'success_rates': [],
        'volume_efficiencies': [],
        'decision_times': [],
        'boxes_placed': [],
        'failed_placements': []
    }

    for episode in range(n_episodes):
        env = OptimizedPalletEnv()
        obs, _ = env.reset(seed=episode + 1000)  # Different seeds for evaluation

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
        volume_efficiency = total_volume / env.max_volume

        results['volumes'].append(total_volume)
        results['success_rates'].append(success_rate)
        results['volume_efficiencies'].append(volume_efficiency)
        results['decision_times'].append(episode_time / env.n_boxes)
        results['boxes_placed'].append(boxes_placed)
        results['failed_placements'].append(failed_placements)

    return results


def test_network_configuration(neurons, config_id, total_timesteps=150000):
    """Test a single network configuration with optimized training"""
    print(f"\nConfiguration {config_id}: {neurons} neurons per layer")
    print("-" * 50)

    # Create environment
    def make_env():
        return OptimizedPalletEnv()

    env = OptimizedPalletEnv()
    vec_env = DummyVecEnv([make_env])

    # Create optimized model
    model = create_optimized_dqn_model(vec_env, neurons)

    # Setup callbacks
    lr_scheduler = LearningRateScheduler()

    # Training
    start_time = time.time()
    print(f"Training for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=lr_scheduler, progress_bar=True)
    training_time = time.time() - start_time

    # Evaluation
    print("Evaluating performance...")
    results = evaluate_model_comprehensive(model, n_episodes=15)

    # Calculate statistics
    avg_volume = np.mean(results['volumes'])
    std_volume = np.std(results['volumes'])
    avg_success = np.mean(results['success_rates'])
    std_success = np.std(results['success_rates'])
    avg_efficiency = np.mean(results['volume_efficiencies'])
    avg_decision_time = np.mean(results['decision_times'])

    print(f"Results:")
    print(f"  Volume: {avg_volume:.2f} ± {std_volume:.2f}")
    print(f"  Success Rate: {avg_success:.3f} ± {std_success:.3f}")
    print(f"  Volume Efficiency: {avg_efficiency:.3f}")
    print(f"  Decision Time: {avg_decision_time:.6f}s")
    print(f"  Training Time: {training_time:.1f}s")

    # Show visualization for first configuration
    if config_id == 1:
        test_env = OptimizedPalletEnv()
        obs, _ = test_env.reset(seed=999)

        for step in range(test_env.n_boxes):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = test_env.step(action)
            if done:
                break

        test_env.render(f"Optimized DQN - {neurons} neurons")

    return {
        'neurons': neurons,
        'avg_volume': avg_volume,
        'std_volume': std_volume,
        'avg_success': avg_success,
        'std_success': std_success,
        'avg_efficiency': avg_efficiency,
        'avg_decision_time': avg_decision_time,
        'training_time': training_time,
        'all_results': results
    }


def run_complete_network_analysis():
    """Run comprehensive analysis with increased training time"""
    print("DQN Network Architecture Analysis")
    print("=" * 50)

    # Test 5 different network configurations with more focused sizes
    network_configs = [64, 128, 256, 384, 512]  # Adjusted network sizes
    results = {}

    # Random baseline
    print("\nRandom Baseline Evaluation")
    print("-" * 30)

    random_results = []
    for episode in range(15):
        env = OptimizedPalletEnv()
        obs, _ = env.reset(seed=episode + 2000)

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
        volume_efficiency = total_volume / env.max_volume
        random_results.append({
            'volume': total_volume,
            'success_rate': success_rate,
            'volume_efficiency': volume_efficiency
        })

    random_baseline = {
        'avg_volume': np.mean([r['volume'] for r in random_results]),
        'std_volume': np.std([r['volume'] for r in random_results]),
        'avg_success': np.mean([r['success_rate'] for r in random_results]),
        'std_success': np.std([r['success_rate'] for r in random_results]),
        'avg_efficiency': np.mean([r['volume_efficiency'] for r in random_results])
    }

    print(f"Random Performance:")
    print(f"  Volume: {random_baseline['avg_volume']:.2f} ± {random_baseline['std_volume']:.2f}")
    print(f"  Success Rate: {random_baseline['avg_success']:.3f} ± {random_baseline['std_success']:.3f}")
    print(f"  Volume Efficiency: {random_baseline['avg_efficiency']:.3f}")

    # Test each network configuration with increased training time
    for i, neurons in enumerate(network_configs):
        result = test_network_configuration(neurons, i + 1, total_timesteps=300000)  # Doubled training time
        results[neurons] = result

    return results, random_baseline


def create_performance_analysis(results, random_baseline):
    """Create comprehensive performance analysis plots"""

    configs = sorted(results.keys())

    # Extract metrics
    volumes = [results[config]['avg_volume'] for config in configs]
    volume_stds = [results[config]['std_volume'] for config in configs]
    success_rates = [results[config]['avg_success'] for config in configs]
    success_stds = [results[config]['std_success'] for config in configs]
    efficiencies = [results[config]['avg_efficiency'] for config in configs]
    training_times = [results[config]['training_time'] for config in configs]

    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DQN Network Architecture Performance Analysis', fontsize=16, fontweight='bold')

    # Volume performance
    x_pos = range(len(configs))
    bars1 = axes[0, 0].bar(x_pos, volumes, yerr=volume_stds, capsize=5,
                           color='steelblue', alpha=0.8, edgecolor='black')
    axes[0, 0].axhline(y=random_baseline['avg_volume'], color='red', linestyle='--',
                       linewidth=2, label=f"Random: {random_baseline['avg_volume']:.1f}")
    axes[0, 0].set_xlabel('Network Size (neurons per layer)')
    axes[0, 0].set_ylabel('Average Volume Placed')
    axes[0, 0].set_title('Volume Efficiency by Network Size')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(configs)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Add improvement percentages
    for i, (v, std) in enumerate(zip(volumes, volume_stds)):
        improvement = (v - random_baseline['avg_volume']) / random_baseline['avg_volume'] * 100
        color = 'green' if improvement > 0 else 'red'
        axes[0, 0].text(i, v + std + 1, f'{improvement:+.1f}%',
                        ha='center', va='bottom', fontweight='bold', color=color)

    # Success rates
    bars2 = axes[0, 1].bar(x_pos, success_rates, yerr=success_stds, capsize=5,
                           color='forestgreen', alpha=0.8, edgecolor='black')
    axes[0, 1].axhline(y=random_baseline['avg_success'], color='red', linestyle='--',
                       linewidth=2, label=f"Random: {random_baseline['avg_success']:.3f}")
    axes[0, 1].set_xlabel('Network Size (neurons per layer)')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_title('Placement Success Rate')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(configs)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Volume efficiency
    bars3 = axes[1, 0].bar(x_pos, efficiencies, color='orange', alpha=0.8, edgecolor='black')
    axes[1, 0].axhline(y=random_baseline['avg_efficiency'], color='red', linestyle='--',
                       linewidth=2, label=f"Random: {random_baseline['avg_efficiency']:.3f}")
    axes[1, 0].set_xlabel('Network Size (neurons per layer)')
    axes[1, 0].set_ylabel('Volume Efficiency (fraction of max)')
    axes[1, 0].set_title('Space Utilization Efficiency')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(configs)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Training efficiency
    bars4 = axes[1, 1].bar(x_pos, training_times, color='purple', alpha=0.8, edgecolor='black')
    axes[1, 1].set_xlabel('Network Size (neurons per layer)')
    axes[1, 1].set_ylabel('Training Time (seconds)')
    axes[1, 1].set_title('Training Efficiency')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(configs)
    axes[1, 1].grid(True, alpha=0.3)

    # Add training time labels
    for i, t in enumerate(training_times):
        axes[1, 1].text(i, t + 5, f'{t:.0f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def print_comprehensive_summary(results, random_baseline):
    """Print detailed performance summary"""

    print("\n" + "=" * 90)
    print("COMPREHENSIVE DQN PERFORMANCE ANALYSIS")
    print("=" * 90)
    print(f"{'Config':<8} {'Volume':<12} {'Success':<10} {'Efficiency':<12} {'Improvement':<12} {'Status':<10}")
    print("-" * 90)

    # Random baseline
    print(f"{'Random':<8} {random_baseline['avg_volume']:<12.2f} "
          f"{random_baseline['avg_success']:<10.3f} {random_baseline['avg_efficiency']:<12.3f} "
          f"{'Baseline':<12} {'Reference':<10}")

    successful_configs = 0
    best_volume = 0
    best_config = None
    improvements = []

    for neurons in sorted(results.keys()):
        data = results[neurons]
        volume_improvement = (data['avg_volume'] - random_baseline['avg_volume']) / random_baseline['avg_volume'] * 100
        improvements.append(volume_improvement)

        if data['avg_volume'] > random_baseline['avg_volume']:
            status = "SUCCESS"
            successful_configs += 1
            if data['avg_volume'] > best_volume:
                best_volume = data['avg_volume']
                best_config = neurons
        else:
            status = "FAILED"

        print(f"{neurons:<8} {data['avg_volume']:<12.2f} {data['avg_success']:<10.3f} "
              f"{data['avg_efficiency']:<12.3f} {volume_improvement:<+11.1f}% {status:<10}")

    print("\n" + "=" * 90)
    print("PERFORMANCE SUMMARY")
    print("=" * 90)
    print(f"Successful configurations: {successful_configs}/5")
    print(f"Best performing network: {best_config} neurons")
    print(f"Best volume achieved: {best_volume:.2f}")
    print(f"Best improvement: {max(improvements):+.1f}% over random")
    print(f"Average improvement: {np.mean(improvements):+.1f}% over random")
    print(f"Training device: {device}")

    if successful_configs >= 4:
        print("RESULT: Excellent performance across network sizes")
    elif successful_configs >= 3:
        print("RESULT: Good performance with optimal network size identification")
    elif successful_configs >= 2:
        print("RESULT: Moderate success with clear network size preferences")
    else:
        print("RESULT: Limited success - may require further optimization")


# Main execution
if __name__ == "__main__":
    print("Professional DQN Implementation for 3D Palletizing")
    print("=" * 60)

    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Run comprehensive analysis
    results, random_baseline = run_complete_network_analysis()

    # Create visualizations
    create_performance_analysis(results, random_baseline)

    # Print summary
    print_comprehensive_summary(results, random_baseline)

    # Save results
    final_data = {
        'dqn_network_analysis': results,
        'random_baseline': random_baseline,
        'device_used': str(device),
        'configurations_tested': sorted(results.keys()),
        'training_timesteps': 300000,
        'environment_type': 'OptimizedPalletEnv'
    }

    with open('dqn_network_analysis_results.pkl', 'wb') as f:
        pickle.dump(final_data, f)

    print("\nAnalysis completed and saved to 'dqn_network_analysis_results.pkl'")