import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Box:
    def __init__(self, size, position=None):
        self.original_size = size
        self.size = size
        self.position = position

    def set_position(self, pos):
        self.position = pos

    def rotate(self, rotation_type):
        """Apply rotation: 0=no rotation, 1=90°, 2=180°"""
        l, w, h = self.original_size
        if rotation_type == 0:
            self.size = (l, w, h)
        elif rotation_type == 1:
            self.size = (w, l, h)
        elif rotation_type == 2:
            self.size = (h, w, l)

    def get_bounds(self):
        x, y, z = self.position
        l, w, h = self.size
        return (x, x + l), (y, y + w), (z, z + h)

    def get_volume(self):
        return self.size[0] * self.size[1] * self.size[2]


class AdvancedPalletEnv(gym.Env):
    """Advanced 3D palletizing environment with rotation and enhanced features"""

    def __init__(self, enable_rotation=True, curriculum_level=1):
        super(AdvancedPalletEnv, self).__init__()
        self.pallet_size = (5, 5, 5)
        self.n_boxes = 100
        self.max_volume = 125
        self.enable_rotation = enable_rotation
        self.curriculum_level = curriculum_level  # 1=easy, 2=medium, 3=hard

        # Enhanced observation space with spatial features
        grid_size = np.prod(self.pallet_size)  # 125
        height_map_size = self.pallet_size[0] * self.pallet_size[1]  # 25
        stability_map_size = self.pallet_size[0] * self.pallet_size[1]  # 25

        obs_size = (
                grid_size +  # 3D occupancy grid
                height_map_size +  # Height map
                stability_map_size +  # Stability map
                6 +  # Current box info (original + rotated sizes)
                8 +  # Enhanced metrics
                self.pallet_size[0] * 2 + self.pallet_size[1] * 2  # Edge utilization
        )

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )

        # Action space: positions × rotations
        n_positions = self.pallet_size[0] * self.pallet_size[1]
        n_rotations = 3 if enable_rotation else 1
        self.action_space = spaces.Discrete(n_positions * n_rotations)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.occupied = np.zeros(self.pallet_size, dtype=np.int32)
        self.stability_map = np.zeros(self.pallet_size, dtype=np.float32)
        self.placed_boxes = []
        self.current_box_idx = 0

        # Curriculum learning: easier boxes at lower levels
        if self.curriculum_level == 1:
            # Easier: more 1x1x1 and 1x1x2 boxes
            box_types = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1)]
            weights = [0.4, 0.3, 0.2, 0.1]
        elif self.curriculum_level == 2:
            # Medium: balanced distribution
            box_types = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 1, 1), (2, 1, 2), (2, 2, 1)]
            weights = [0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1]
        else:
            # Hard: include larger boxes
            box_types = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)]
            weights = [0.15, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05]

        self.box_queue = [Box(random.choices(box_types, weights=weights)[0])
                          for _ in range(self.n_boxes)]

        self.total_volume_placed = 0
        self.successful_placements = 0
        self.failed_placements = 0
        self.placement_history = []

        return self._get_obs(), {}

    def _get_height_map(self):
        """Get 2D height map"""
        height_map = np.zeros((self.pallet_size[0], self.pallet_size[1]))
        for x in range(self.pallet_size[0]):
            for y in range(self.pallet_size[1]):
                for z in range(self.pallet_size[2] - 1, -1, -1):
                    if self.occupied[x, y, z] == 1:
                        height_map[x, y] = (z + 1) / self.pallet_size[2]
                        break
        return height_map

    def _get_stability_analysis(self):
        """Analyze structural stability"""
        stability = np.zeros((self.pallet_size[0], self.pallet_size[1]))

        for x in range(self.pallet_size[0]):
            for y in range(self.pallet_size[1]):
                # Calculate support strength
                support_strength = 0
                for z in range(self.pallet_size[2]):
                    if self.occupied[x, y, z] == 1:
                        # Check support from below
                        if z == 0:
                            support_strength += 1.0  # Ground support
                        else:
                            adjacent_support = 0
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    nx, ny = x + dx, y + dy
                                    if (0 <= nx < self.pallet_size[0] and
                                            0 <= ny < self.pallet_size[1] and
                                            self.occupied[nx, ny, z - 1] == 1):
                                        adjacent_support += 1
                            support_strength += adjacent_support / 9.0

                stability[x, y] = min(support_strength, 1.0)

        return stability

    def _get_edge_utilization(self):
        """Analyze edge and corner utilization"""
        edges_x = np.mean(self.occupied[0, :, :]) + np.mean(self.occupied[-1, :, :])
        edges_y = np.mean(self.occupied[:, 0, :]) + np.mean(self.occupied[:, -1, :])

        edge_features = np.array([
            edges_x / 2.0, edges_y / 2.0,
            np.mean(self.occupied[0, 0, :]),  # Corner utilization
            np.mean(self.occupied[0, -1, :]),
            np.mean(self.occupied[-1, 0, :]),
            np.mean(self.occupied[-1, -1, :])
        ])

        return edge_features

    def _get_obs(self):
        # 3D occupancy grid (normalized)
        grid = self.occupied.astype(np.float32)

        # Height map
        height_map = self._get_height_map()

        # Stability analysis
        stability = self._get_stability_analysis()

        # Current box information with potential rotations
        if self.current_box_idx < self.n_boxes:
            box = self.box_queue[self.current_box_idx]
            original_size = np.array(box.original_size, dtype=np.float32) / 2.0

            # Show all possible rotated sizes
            rotated_sizes = []
            for rot in range(3 if self.enable_rotation else 1):
                temp_box = Box(box.original_size)
                temp_box.rotate(rot)
                rotated_sizes.extend(temp_box.size)

            box_info = np.array(rotated_sizes[:6], dtype=np.float32) / 2.0  # Normalize
        else:
            box_info = np.zeros(6, dtype=np.float32)

        # Enhanced metrics
        progress = self.current_box_idx / self.n_boxes
        success_rate = self.successful_placements / max(1, self.current_box_idx)
        volume_efficiency = self.total_volume_placed / self.max_volume
        height_variance = np.var(height_map)
        stability_score = np.mean(stability)
        compactness = self._calculate_compactness()
        density = self.total_volume_placed / (np.sum(self.occupied) + 1e-6)

        metrics = np.array([
            progress, success_rate, volume_efficiency, height_variance,
            stability_score, compactness, density,
            len(self.placed_boxes) / 100.0
        ], dtype=np.float32)

        # Edge utilization
        edge_features = self._get_edge_utilization()

        return np.concatenate([
            grid.flatten(),
            height_map.flatten(),
            stability.flatten(),
            box_info,
            metrics,
            edge_features
        ])

    def _calculate_compactness(self):
        """Calculate how compact the placement is"""
        if len(self.placed_boxes) < 2:
            return 1.0

        # Calculate average distance between box centers
        centers = []
        for box in self.placed_boxes:
            x, y, z = box.position
            l, w, h = box.size
            center = (x + l / 2, y + w / 2, z + h / 2)
            centers.append(center)

        total_distance = 0
        count = 0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(centers[i], centers[j])))
                total_distance += dist
                count += 1

        avg_distance = total_distance / max(count, 1)
        # Normalize and invert (lower distance = higher compactness)
        return max(0, 1 - avg_distance / 10.0)

    def _decode_action(self, action):
        """Decode action into position and rotation"""
        n_positions = self.pallet_size[0] * self.pallet_size[1]
        if self.enable_rotation:
            position_idx = action % n_positions
            rotation_idx = action // n_positions
        else:
            position_idx = action
            rotation_idx = 0

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

        # Support check
        if z1 == 0:
            return True
        return np.all(self.occupied[x1:x2, y1:y2, z1 - 1:z1] == 1)

    def _calculate_advanced_reward(self, box, z, placed, rotation_used):
        """Multi-objective reward function"""
        if not placed:
            return -2.0

        volume = box.get_volume()

        # Base rewards
        placement_reward = 5.0
        volume_reward = volume * 3.0

        # Height efficiency
        height_efficiency = (self.pallet_size[2] - z) / self.pallet_size[2]
        height_reward = height_efficiency * 2.0

        # Stability reward
        stability_reward = 0
        x, y = box.position[0], box.position[1]
        l, w = box.size[0], box.size[1]

        if z == 0:
            stability_reward = 3.0
        else:
            # Check how well supported this placement is
            support_area = 0
            for dx in range(l):
                for dy in range(w):
                    if self.occupied[x + dx, y + dy, z - 1] == 1:
                        support_area += 1
            support_ratio = support_area / (l * w)
            stability_reward = support_ratio * 2.0

        # Compactness reward (encourage adjacent placements)
        compactness_reward = 0
        if self.successful_placements > 0:
            adjacent_count = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == dy == dz == 0:
                            continue
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (0 <= nx < self.pallet_size[0] and
                                0 <= ny < self.pallet_size[1] and
                                0 <= nz < self.pallet_size[2] and
                                self.occupied[nx, ny, nz] == 1):
                            adjacent_count += 1

            compactness_reward = min(adjacent_count * 0.5, 3.0)

        # Rotation bonus (reward for using rotation effectively)
        rotation_reward = 0
        if self.enable_rotation and rotation_used > 0:
            # Check if rotation improved the fit
            original_box = Box(box.original_size)
            original_box.set_position(box.position)
            if (original_box.size != box.size and
                    box.size[0] * box.size[1] <= original_box.original_size[0] * original_box.original_size[1]):
                rotation_reward = 1.0

        # Edge utilization bonus
        edge_bonus = 0
        if (x == 0 or x + l == self.pallet_size[0] or
                y == 0 or y + w == self.pallet_size[1]):
            edge_bonus = 1.0

        total_reward = (placement_reward + volume_reward + height_reward +
                        stability_reward + compactness_reward + rotation_reward + edge_bonus)

        return total_reward

    def step(self, action):
        if self.current_box_idx >= self.n_boxes:
            return self._get_obs(), 0, True, False, {}

        box = self.box_queue[self.current_box_idx]
        x, y, rotation = self._decode_action(action)

        # Apply rotation
        box.rotate(rotation)

        placed = False
        best_z = 0

        # Find lowest valid placement
        for z in range(self.pallet_size[2]):
            box.set_position((x, y, z))
            if self.is_valid_placement(box):
                self.placed_boxes.append(box)
                (x1, x2), (y1, y2), (z1, z2) = box.get_bounds()
                self.occupied[x1:x2, y1:y2, z1:z2] = 1

                self.total_volume_placed += box.get_volume()
                self.successful_placements += 1
                self.placement_history.append({
                    'position': (x, y, z),
                    'size': box.size,
                    'volume': box.get_volume(),
                    'rotation': rotation
                })
                placed = True
                best_z = z
                break

        if not placed:
            self.failed_placements += 1

        # Calculate reward
        reward = self._calculate_advanced_reward(box, best_z, placed, rotation)

        self.current_box_idx += 1
        done = self.current_box_idx >= self.n_boxes

        # End-of-episode rewards
        if done:
            final_success_rate = self.successful_placements / self.n_boxes
            final_volume_efficiency = self.total_volume_placed / self.max_volume
            final_compactness = self._calculate_compactness()

            # Multi-objective end bonuses
            if final_success_rate > 0.5:
                reward += 100 * final_success_rate
            if final_volume_efficiency > 0.8:
                reward += 150 * final_volume_efficiency
            if final_compactness > 0.7:
                reward += 50 * final_compactness

            # Penalty for poor performance
            if final_success_rate < 0.3:
                reward -= 50

        return self._get_obs(), reward, done, False, {
            'placed': placed,
            'volume': box.get_volume() if placed else 0,
            'total_volume': self.total_volume_placed,
            'successful_placements': self.successful_placements,
            'failed_placements': self.failed_placements,
            'success_rate': self.successful_placements / max(1, self.current_box_idx),
            'volume_efficiency': self.total_volume_placed / self.max_volume,
            'compactness': self._calculate_compactness(),
            'rotation_used': rotation if placed else 0
        }

    def render(self, title="Advanced 3D Palletizing"):
        fig = plt.figure(figsize=(15, 12))

        # Main 3D plot
        ax1 = fig.add_subplot(221, projection='3d')

        if len(self.placed_boxes) > 0:
            colors = plt.cm.tab20(np.linspace(0, 1, len(self.placed_boxes)))

            for i, box in enumerate(self.placed_boxes):
                x, y, z = box.position
                l, w, h = box.size
                self._draw_box(ax1, x, y, z, l, w, h, colors[i])

        ax1.set_xlim(0, self.pallet_size[0])
        ax1.set_ylim(0, self.pallet_size[1])
        ax1.set_zlim(0, self.pallet_size[2])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Placement')

        # Height map
        ax2 = fig.add_subplot(222)
        height_map = self._get_height_map()
        im1 = ax2.imshow(height_map, cmap='viridis', origin='lower')
        ax2.set_title('Height Map')
        ax2.set_xlabel('Y')
        ax2.set_ylabel('X')
        plt.colorbar(im1, ax=ax2, label='Height')

        # Stability map
        ax3 = fig.add_subplot(223)
        stability = self._get_stability_analysis()
        im2 = ax3.imshow(stability, cmap='RdYlGn', origin='lower')
        ax3.set_title('Stability Analysis')
        ax3.set_xlabel('Y')
        ax3.set_ylabel('X')
        plt.colorbar(im2, ax=ax3, label='Stability')

        # Performance metrics
        ax4 = fig.add_subplot(224)
        success_rate = self.successful_placements / max(1, self.current_box_idx)
        volume_efficiency = self.total_volume_placed / self.max_volume
        compactness = self._calculate_compactness()

        metrics = ['Success\nRate', 'Volume\nEfficiency', 'Compactness']
        values = [success_rate, volume_efficiency, compactness]
        colors_bar = ['skyblue', 'lightgreen', 'coral']

        bars = ax4.bar(metrics, values, color=colors_bar, alpha=0.7)
        ax4.set_ylim(0, 1)
        ax4.set_title('Performance Metrics')
        ax4.set_ylabel('Score')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.suptitle(f'{title}\nBoxes: {self.successful_placements}/{self.current_box_idx}, '
                     f'Volume: {self.total_volume_placed}/{self.max_volume}', fontsize=14)
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


def create_advanced_dqn_model(env, neurons_per_layer):
    """Create advanced DQN with curriculum learning support"""

    # Adaptive architecture based on observation space
    obs_size = env.observation_space.shape[0]

    if obs_size > 200:  # Complex observation space
        net_arch = [neurons_per_layer * 2, neurons_per_layer, neurons_per_layer, neurons_per_layer // 2]
    else:
        net_arch = [neurons_per_layer, neurons_per_layer, neurons_per_layer // 2]

    policy_kwargs = {
        'net_arch': net_arch,
        'activation_fn': torch.nn.ReLU,
    }

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=5e-4,  # Optimized learning rate
        buffer_size=500000,  # Large experience buffer
        learning_starts=50000,  # Extended exploration
        batch_size=128,
        tau=0.001,  # Soft target updates
        gamma=0.99,
        train_freq=4,
        gradient_steps=4,
        target_update_interval=10000,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.03,
        max_grad_norm=0.5,
        verbose=0,
        device=device
    )

    return model


def run_enhanced_experiment():
    """Run enhanced experiment with advanced features"""

    print("Advanced DQN with Rotation and Enhanced Features")
    print("=" * 60)

    # Test configurations
    configs = [
        {'neurons': 128, 'rotation': False, 'curriculum': 1, 'name': 'Baseline DQN'},
        {'neurons': 256, 'rotation': True, 'curriculum': 1, 'name': 'DQN + Rotation'},
        {'neurons': 256, 'rotation': True, 'curriculum': 2, 'name': 'DQN + Rotation + Curriculum'},
        {'neurons': 384, 'rotation': True, 'curriculum': 2, 'name': 'Large DQN + All Features'},
    ]

    results = {}

    # Random baseline
    print("\nRandom Baseline")
    env = AdvancedPalletEnv(enable_rotation=True, curriculum_level=2)
    random_results = []

    for episode in range(10):
        obs, _ = env.reset(seed=episode + 3000)
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

        random_results.append({
            'volume': total_volume,
            'success_rate': boxes_placed / env.n_boxes,
            'volume_efficiency': total_volume / env.max_volume
        })

    random_baseline = {
        'avg_volume': np.mean([r['volume'] for r in random_results]),
        'avg_success': np.mean([r['success_rate'] for r in random_results]),
        'avg_efficiency': np.mean([r['volume_efficiency'] for r in random_results])
    }

    print(f"Random: Volume={random_baseline['avg_volume']:.2f}, "
          f"Success={random_baseline['avg_success']:.3f}")

    # Test each configuration
    for i, config in enumerate(configs):
        print(f"\n{config['name']} ({config['neurons']} neurons)")
        print("-" * 50)

        # Create environment
        def make_env():
            return AdvancedPalletEnv(
                enable_rotation=config['rotation'],
                curriculum_level=config['curriculum']
            )

        env = AdvancedPalletEnv(
            enable_rotation=config['rotation'],
            curriculum_level=config['curriculum']
        )
        vec_env = DummyVecEnv([make_env])

        # Create model
        model = create_advanced_dqn_model(vec_env, config['neurons'])

        # Train
        start_time = time.time()
        print("Training...")
        model.learn(total_timesteps=200000, progress_bar=True)
        training_time = time.time() - start_time

        # Evaluate
        print("Evaluating...")
        volumes = []
        success_rates = []
        efficiencies = []
        compactness_scores = []

        for episode in range(12):
            test_env = AdvancedPalletEnv(
                enable_rotation=config['rotation'],
                curriculum_level=config['curriculum']
            )
            obs, _ = test_env.reset(seed=episode + 4000)

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

            volumes.append(total_volume)
            success_rates.append(boxes_placed / test_env.n_boxes)
            efficiencies.append(total_volume / test_env.max_volume)
            compactness_scores.append(info.get('compactness', 0))

            # Show visualization for first config, first episode
            if i == 0 and episode == 0:
                test_env.render(f"{config['name']} - Episode {episode + 1}")

        results[config['name']] = {
            'avg_volume': np.mean(volumes),
            'std_volume': np.std(volumes),
            'avg_success': np.mean(success_rates),
            'avg_efficiency': np.mean(efficiencies),
            'avg_compactness': np.mean(compactness_scores),
            'training_time': training_time,
            'config': config
        }

        improvement = (np.mean(volumes) - random_baseline['avg_volume']) / random_baseline['avg_volume'] * 100

        print(f"Results:")
        print(f"  Volume: {np.mean(volumes):.2f} ± {np.std(volumes):.2f}")
        print(f"  Success Rate: {np.mean(success_rates):.3f}")
        print(f"  Volume Efficiency: {np.mean(efficiencies):.3f}")
        print(f"  Compactness: {np.mean(compactness_scores):.3f}")
        print(f"  Improvement: {improvement:+.1f}%")
        print(f"  Training Time: {training_time:.0f}s")

    return results, random_baseline


def print_enhanced_summary(results, random_baseline):
    """Print comprehensive summary of enhanced experiments"""

    print("\n" + "=" * 80)
    print("ENHANCED DQN PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"{'Configuration':<30} {'Volume':<10} {'Success':<10} {'Efficiency':<12} {'Improvement':<12}")
    print("-" * 80)

    print(f"{'Random Baseline':<30} {random_baseline['avg_volume']:<10.2f} "
          f"{random_baseline['avg_success']:<10.3f} {random_baseline['avg_efficiency']:<12.3f} {'--':<12}")

    best_volume = 0
    best_config = None

    for name, data in results.items():
        improvement = (data['avg_volume'] - random_baseline['avg_volume']) / random_baseline['avg_volume'] * 100

        if data['avg_volume'] > best_volume:
            best_volume = data['avg_volume']
            best_config = name

        print(f"{name:<30} {data['avg_volume']:<10.2f} {data['avg_success']:<10.3f} "
              f"{data['avg_efficiency']:<12.3f} {improvement:<+11.1f}%")

    print("\n" + "=" * 80)
    print("ENHANCEMENT ANALYSIS")
    print("=" * 80)
    print(f"Best Configuration: {best_config}")
    print(f"Best Volume: {best_volume:.2f}")
    print(
        f"Maximum Improvement: {((best_volume - random_baseline['avg_volume']) / random_baseline['avg_volume'] * 100):+.1f}%")

    # Feature analysis
    print("\nFeature Impact Analysis:")
    baseline_vol = results.get('Baseline DQN', {}).get('avg_volume', 0)
    rotation_vol = results.get('DQN + Rotation', {}).get('avg_volume', 0)
    curriculum_vol = results.get('DQN + Rotation + Curriculum', {}).get('avg_volume', 0)

    if rotation_vol > baseline_vol:
        rotation_impact = (rotation_vol - baseline_vol) / baseline_vol * 100
        print(f"  Rotation Impact: +{rotation_impact:.1f}% volume improvement")

    if curriculum_vol > rotation_vol:
        curriculum_impact = (curriculum_vol - rotation_vol) / rotation_vol * 100
        print(f"  Curriculum Learning Impact: +{curriculum_impact:.1f}% additional improvement")


# Main execution
if __name__ == "__main__":
    print("Enhanced 3D Palletizing with Advanced Features")
    print("=" * 60)

    # Set seeds
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    # Run enhanced experiments
    results, random_baseline = run_enhanced_experiment()

    # Print analysis
    print_enhanced_summary(results, random_baseline)

    # Save results
    enhanced_data = {
        'enhanced_results': results,
        'random_baseline': random_baseline,
        'features_tested': ['rotation', 'curriculum_learning', 'advanced_rewards', 'stability_analysis'],
        'device_used': str(device)
    }

    with open('enhanced_dqn_results.pkl', 'wb') as f:
        pickle.dump(enhanced_data, f)

    print("\nEnhanced analysis completed and saved to 'enhanced_dqn_results.pkl'")