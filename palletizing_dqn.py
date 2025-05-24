import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import torch
import time
import pickle
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings

warnings.filterwarnings('ignore')

# Enable CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 EMERGENCY FIX - Using device: {device}")


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


class WorkingPalletEnv(gym.Env):
    """Using the EXACT setup that worked before (80.87 volume)"""

    def __init__(self):
        super(WorkingPalletEnv, self).__init__()
        self.pallet_size = (5, 5, 5)
        self.n_boxes = 100

        # EXACT same observation as before
        pallet_dims = self.pallet_size[0] * self.pallet_size[1] * self.pallet_size[2]
        self.observation_space = spaces.Box(
            low=0, high=10,
            shape=(pallet_dims + 6,),  # 125 + 6 = 131 (same as working version)
            dtype=np.float32
        )

        # Simple action space (no rotation - this worked!)
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

        return self._get_obs(), {}

    def _get_obs(self):
        # EXACT same observation as working version
        flat_occupied = self.occupied.flatten().astype(np.float32)

        if self.current_box_idx < self.n_boxes:
            box = self.box_queue[self.current_box_idx]
            box_info = np.array(list(box.original_size), dtype=np.float32)
        else:
            box_info = np.array([0, 0, 0], dtype=np.float32)

        # Progress info (same as working version)
        progress_info = np.array([
            self.current_box_idx / self.n_boxes,  # Progress ratio
            self.successful_placements / max(1, self.current_box_idx),  # Success rate so far
            self.total_volume_placed / 125.0  # Volume ratio
        ], dtype=np.float32)

        return np.concatenate([flat_occupied, box_info, progress_info])

    def is_valid_placement(self, box):
        (x1, x2), (y1, y2), (z1, z2) = box.get_bounds()

        if x2 > self.pallet_size[0] or y2 > self.pallet_size[1] or z2 > self.pallet_size[2]:
            return False

        if np.any(self.occupied[x1:x2, y1:y2, z1:z2] != 0):
            return False

        if z1 == 0:
            return True
        return np.all(self.occupied[x1:x2, y1:y2, z1 - 1:z1] == 1)

    def step(self, action):
        if self.current_box_idx >= self.n_boxes:
            return self._get_obs(), 0, True, False, {}

        box = self.box_queue[self.current_box_idx]
        x = action // self.pallet_size[1]
        y = action % self.pallet_size[1]

        placed = False
        reward = 0

        # Try to place at lowest z
        for z in range(self.pallet_size[2]):
            box.set_position((x, y, z))
            if self.is_valid_placement(box):
                self.placed_boxes.append(box)
                (x1, x2), (y1, y2), (z1, z2) = box.get_bounds()
                self.occupied[x1:x2, y1:y2, z1:z2] = 1

                # WORKING reward function (from successful version)
                volume = box.get_volume()
                base_reward = volume * 2.0
                height_bonus = max(0, (self.pallet_size[2] - z) * 0.5)
                efficiency_bonus = 1.0

                reward = base_reward + height_bonus + efficiency_bonus
                self.total_volume_placed += volume
                self.successful_placements += 1
                placed = True
                break

        if not placed:
            reward = -2.0
            self.failed_placements += 1

        self.current_box_idx += 1
        done = self.current_box_idx >= self.n_boxes

        # End-of-episode bonus (from working version)
        if done and self.successful_placements > 0:
            efficiency_ratio = self.successful_placements / self.n_boxes
            volume_ratio = self.total_volume_placed / 125.0
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


def create_working_dqn_model(env, neurons_per_layer):
    """Use EXACT hyperparameters that worked before"""

    policy_kwargs = {
        'net_arch': [neurons_per_layer, neurons_per_layer],  # 2 layers
        'activation_fn': torch.nn.ReLU,
    }

    # EXACT same hyperparameters as working version
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,
        buffer_size=100000,  # Large buffer
        learning_starts=10000,  # More exploration
        batch_size=128,  # Larger batches
        tau=1.0,
        gamma=0.95,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2000,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        max_grad_norm=10,
        verbose=0,
        device=device
    )

    return model


def test_working_config(neurons, config_id, total_timesteps=120000):
    """Test with WORKING configuration"""
    print(f"\n--- Configuration {config_id}: {neurons} neurons (WORKING VERSION) ---")

    # Create environment using working setup
    def make_env():
        return WorkingPalletEnv()

    env = WorkingPalletEnv()
    vec_env = DummyVecEnv([make_env])

    # Create model with working hyperparameters
    model = create_working_dqn_model(vec_env, neurons)

    # Train with MORE timesteps (this worked before!)
    start_time = time.time()
    print("Training (120k timesteps)...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    training_time = time.time() - start_time

    # Evaluate
    print("Evaluating...")
    volumes = []
    success_rates = []

    for episode in range(12):
        test_env = WorkingPalletEnv()
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

        # Show visualization for first config
        if config_id == 1 and episode == 0:
            test_env.render(f"Working DQN {neurons} neurons - Episode {episode + 1}")

    avg_volume = np.mean(volumes)
    avg_success = np.mean(success_rates)
    std_volume = np.std(volumes)
    std_success = np.std(success_rates)

    print(f"Results: Volume={avg_volume:.2f}±{std_volume:.2f}, "
          f"Success={avg_success:.3f}±{std_success:.3f}, "
          f"Training time: {training_time:.1f}s")

    return {
        'neurons': neurons,
        'avg_volume': avg_volume,
        'std_volume': std_volume,
        'avg_success': avg_success,
        'std_success': std_success,
        'training_time': training_time,
        'volumes': volumes,
        'success_rates': success_rates
    }


def run_emergency_working_experiment():
    """Emergency experiment using WORKING configuration"""

    print("🚨 EMERGENCY: USING WORKING DQN CONFIGURATION")
    print("=" * 60)

    # 5 network configurations using WORKING setup
    network_configs = [32, 64, 128, 256, 512]

    results = {}

    # Random baseline
    print("\n🎲 Random Baseline (Working Environment)")
    random_volumes = []
    random_success = []

    for episode in range(12):
        env = WorkingPalletEnv()
        obs, _ = env.reset(seed=episode + 100)

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

        random_volumes.append(total_volume)
        random_success.append(boxes_placed / env.n_boxes)

    random_result = {
        'avg_volume': np.mean(random_volumes),
        'std_volume': np.std(random_volumes),
        'avg_success': np.mean(random_success),
        'std_success': np.std(random_success)
    }

    print(f"Random: Volume={random_result['avg_volume']:.2f}±{random_result['std_volume']:.2f}, "
          f"Success={random_result['avg_success']:.3f}±{random_result['std_success']:.3f}")

    # Test all 5 configurations with WORKING setup
    for i, neurons in enumerate(network_configs):
        result = test_working_config(neurons, i + 1, total_timesteps=120000)
        results[neurons] = result

    return results, random_result


def create_emergency_plots(results, random_result):
    """Create plots for emergency results"""

    configs = sorted(results.keys())

    avg_volumes = [results[config]['avg_volume'] for config in configs]
    std_volumes = [results[config]['std_volume'] for config in configs]

    avg_success_rates = [results[config]['avg_success'] for config in configs]
    std_success_rates = [results[config]['std_success'] for config in configs]

    training_times = [results[config]['training_time'] for config in configs]

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('🚨 EMERGENCY FIX: Working DQN - 5 Network Configurations', fontsize=16)

    # Volume comparison
    x_pos = range(len(configs))
    bars1 = axes[0].bar(x_pos, avg_volumes, yerr=std_volumes, capsize=5,
                        color='green', alpha=0.8, label='Working DQN')
    axes[0].axhline(y=random_result['avg_volume'], color='red', linestyle='--',
                    label=f"Random: {random_result['avg_volume']:.1f}")
    axes[0].set_xlabel('Neurons per Layer')
    axes[0].set_ylabel('Average Volume Placed')
    axes[0].set_title('Volume Efficiency (FIXED)')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(configs)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Add improvement indicators
    for i, (v, std) in enumerate(zip(avg_volumes, std_volumes)):
        improvement = (v - random_result['avg_volume']) / random_result['avg_volume'] * 100
        color = 'green' if improvement > 0 else 'red'
        axes[0].text(i, v + std + 1, f'{v:.1f}\n({improvement:+.1f}%)',
                     ha='center', va='bottom', fontweight='bold', color=color)

    # Success rate comparison
    bars2 = axes[1].bar(x_pos, avg_success_rates, yerr=std_success_rates, capsize=5,
                        color='blue', alpha=0.8, label='Working DQN')
    axes[1].axhline(y=random_result['avg_success'], color='red', linestyle='--',
                    label=f"Random: {random_result['avg_success']:.3f}")
    axes[1].set_xlabel('Neurons per Layer')
    axes[1].set_ylabel('Average Success Rate')
    axes[1].set_title('Success Rate (FIXED)')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(configs)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Training time
    bars3 = axes[2].bar(x_pos, training_times, color='orange', alpha=0.8)
    axes[2].set_xlabel('Neurons per Layer')
    axes[2].set_ylabel('Training Time (seconds)')
    axes[2].set_title('Training Time (120k steps)')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(configs)
    axes[2].grid(True, alpha=0.3)

    for i, v in enumerate(training_times):
        axes[2].text(i, v + 5, f'{v:.0f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def print_emergency_summary(results, random_result):
    """Print emergency results summary"""

    print("\n" + "=" * 90)
    print("🚨 EMERGENCY FIX RESULTS - WORKING DQN CONFIGURATION")
    print("=" * 90)
    print(f"{'Config':<8} {'Volume':<12} {'Success':<12} {'Improvement':<15} {'Status':<15} {'Time':<8}")
    print("-" * 90)

    baseline_volume = random_result['avg_volume']
    baseline_success = random_result['avg_success']

    print(
        f"{'Random':<8} {baseline_volume:<12.2f} {baseline_success:<12.3f} {'Baseline':<15} {'Reference':<15} {'N/A':<8}")

    successful_configs = 0
    best_volume = 0
    best_config = None

    for neurons in sorted(results.keys()):
        data = results[neurons]
        volume_improvement = (data['avg_volume'] - baseline_volume) / baseline_volume * 100

        if data['avg_volume'] > baseline_volume:
            status = "✅ SUCCESS"
            successful_configs += 1
            if data['avg_volume'] > best_volume:
                best_volume = data['avg_volume']
                best_config = neurons
        else:
            status = "❌ FAILED"

        print(f"{neurons:<8} {data['avg_volume']:<12.2f} {data['avg_success']:<12.3f} "
              f"{volume_improvement:+6.1f}%{'':<8} {status:<15} {data['training_time']:<7.0f}s")

    print(f"\n🎯 EMERGENCY RESULTS SUMMARY:")
    print(f"   ✅ Successful configs: {successful_configs}/5")
    print(f"   🏆 Best config: {best_config} neurons ({best_volume:.2f} volume)")
    print(f"   📈 Best improvement: {((best_volume - baseline_volume) / baseline_volume * 100):+.1f}%")

    if successful_configs >= 3:
        print(f"   🎉 RECOVERY SUCCESSFUL! You have working results for your report!")
    else:
        print(f"   ⚠️  Only {successful_configs} configs working. May need more debugging.")


# Main execution
if __name__ == "__main__":
    print("🚨 EMERGENCY DQN FIX - USING PROVEN WORKING CONFIGURATION")
    print("=" * 70)

    # Set seeds
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Run emergency experiment
    results, random_result = run_emergency_working_experiment()

    # Create visualization
    create_emergency_plots(results, random_result)

    # Print summary
    print_emergency_summary(results, random_result)

    # Save results
    emergency_data = {
        'emergency_dqn_results': results,
        'random_baseline': random_result,
        'device_used': str(device),
        'configurations_tested': sorted(results.keys()),
        'note': 'Emergency fix using working DQN configuration'
    }

    with open('emergency_working_dqn_results.pkl', 'wb') as f:
        pickle.dump(emergency_data, f)

    print(f"\n🚨 EMERGENCY EXPERIMENT COMPLETED!")
    print(f"Results saved to 'emergency_working_dqn_results.pkl'")
    print(f"Training performed on: {device}")
    print("=" * 70)