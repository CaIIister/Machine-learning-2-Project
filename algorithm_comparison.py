#!/usr/bin/env python3
"""
Comprehensive comparison of 3D Palletizing algorithms:
1. PPO (upgraded to OptimizedPalletEnv)
2. DQN (your existing implementation)
3. Greedy Algorithm (friend's approach, upgraded)
4. MLP Imitation Learning (friend's approach, upgraded)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import torch
import pickle
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Import your DQN implementation
from palletizing_dqn import OptimizedPalletEnv, create_optimized_dqn_model


class LossTrackingCallback(BaseCallback):
    """Enhanced callback for tracking training losses"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.loss_history = {
            'value_loss': [],
            'policy_gradient_loss': [],
            'entropy_loss': []
        }

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            logs = self.model.logger.name_to_value
            for key in self.loss_history.keys():
                if f"train/{key}" in logs:
                    self.loss_history[key].append(logs[f"train/{key}"])


class UpgradedGreedyAlgorithm:
    """Friend's greedy algorithm upgraded for OptimizedPalletEnv"""

    def __init__(self):
        self.name = "Greedy Algorithm"

    @staticmethod
    def _get_height_map(occupied):
        """Extract height map from 3D occupation grid"""
        W, L, Z = occupied.shape
        height_map = np.zeros((W, L), dtype=int)
        for x in range(W):
            for y in range(L):
                for z in range(Z - 1, -1, -1):
                    if occupied[x, y, z] == 1:
                        height_map[x, y] = z + 1
                        break
        return height_map

    def predict(self, obs, env):
        """Predict best action using greedy strategy"""
        # Extract current box info from observation
        box = env.box_queue[env.current_box_idx]
        l, w, h_box = box.original_size
        W, L, Z = env.pallet_size

        # Get current height map
        height_map = self._get_height_map(env.occupied)

        best_action = None
        best_score = float('inf')

        # Try all possible positions
        for x in range(W - l + 1):
            for y in range(L - w + 1):
                # Find the height at this position
                z = height_map[x:x + l, y:y + w].max()

                # Check if box fits
                if z + h_box > Z:
                    continue

                # Check support (if not on ground)
                if z > 0 and not env.occupied[x:x + l, y:y + w, z - 1].all():
                    continue

                # Score: prefer lower heights (greedy for stability)
                score = z + h_box
                if score < best_score:
                    best_score = score
                    best_action = x * L + y

        # If no valid placement found, return random action
        if best_action is None:
            return env.action_space.sample()

        return best_action


class UpgradedMLPAlgorithm:
    """Friend's MLP algorithm upgraded for OptimizedPalletEnv"""

    def __init__(self, input_dim=157, hidden_dim=256, output_dim=25):
        self.name = "MLP Imitation Learning"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize network weights
        rng = np.random.default_rng(42)
        self.W1 = (rng.standard_normal((input_dim, hidden_dim)) *
                   np.sqrt(2.0 / (input_dim + hidden_dim))).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden_dim, output_dim)) *
                   np.sqrt(2.0 / (hidden_dim + output_dim))).astype(np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)

        self.trained = False

    def forward(self, x):
        """Forward pass through the network"""
        h = np.tanh(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits, h

    def predict(self, obs, env=None):
        """Predict action using trained MLP"""
        if not self.trained:
            # If not trained, use greedy as fallback
            greedy = UpgradedGreedyAlgorithm()
            return greedy.predict(obs, env)

        x = obs.reshape(1, -1).astype(np.float32)
        logits, _ = self.forward(x)
        return int(logits.argmax(axis=1)[0])

    def train_on_demonstrations(self, n_episodes=1000, epochs=200, lr=3e-4):
        """Train MLP using imitation learning on greedy demonstrations"""
        print(f"Collecting demonstrations from greedy algorithm...")

        X, y = self._collect_demonstrations(n_episodes)

        print(f"Training MLP on {len(X)} demonstrations...")
        self._train_mlp(X, y, epochs, lr)
        self.trained = True
        print("MLP training completed!")

    def _collect_demonstrations(self, n_episodes):
        """Collect training data from greedy algorithm"""
        greedy = UpgradedGreedyAlgorithm()
        X, y = [], []

        for episode in range(n_episodes):
            env = OptimizedPalletEnv()
            obs, _ = env.reset(seed=episode + 1000)  # Different seeds for training

            while env.current_box_idx < env.n_boxes:
                action = greedy.predict(obs, env)
                X.append(obs.copy())
                y.append(action)

                obs, reward, done, _, _ = env.step(action)
                if done:
                    break

        return np.array(X), np.array(y)

    def _train_mlp(self, X, y, epochs, lr, batch_size=512):
        """Train MLP using collected demonstrations"""
        N = X.shape[0]
        indices = np.arange(N)

        for epoch in range(epochs):
            np.random.shuffle(indices)

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_idx = indices[start:end]
                x_batch, y_batch = X[batch_idx], y[batch_idx]

                # Forward pass
                logits, h = self.forward(x_batch)

                # Compute loss and gradients
                logits_max = logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits - logits_max)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                # Cross-entropy gradient
                grad_logits = probs.copy()
                grad_logits[np.arange(len(y_batch)), y_batch] -= 1
                grad_logits /= len(y_batch)

                # Backpropagation
                dW2 = h.T @ grad_logits
                db2 = grad_logits.sum(axis=0)
                dh = grad_logits @ self.W2.T * (1 - h ** 2)
                dW1 = x_batch.T @ dh
                db1 = dh.sum(axis=0)

                # Update weights
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2

            if epoch % 50 == 0:
                # Compute accuracy
                pred_logits, _ = self.forward(X[:1000])
                preds = pred_logits.argmax(axis=1)
                acc = (preds == y[:1000]).mean()
                print(f"Epoch {epoch:3d}: Accuracy = {acc:.3f}")


def create_ppo_model(env):
    """Create PPO model for OptimizedPalletEnv"""
    policy_kwargs = {
        'net_arch': [256, 256, 128],  # Deeper network for complex environment
        'activation_fn': torch.nn.ReLU,
    }

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0
    )

    return model


def evaluate_algorithm(algorithm, algorithm_name, n_episodes=15, seeds=None):
    """Evaluate algorithm performance over multiple episodes"""
    if seeds is None:
        seeds = list(range(1000, 1000 + n_episodes))

    results = {
        'volumes': [],
        'success_rates': [],
        'volume_efficiencies': [],
        'decision_times': [],
        'boxes_placed': [],
        'failed_placements': []
    }

    print(f"Evaluating {algorithm_name}...")

    for i, seed in enumerate(seeds):
        env = OptimizedPalletEnv()
        obs, _ = env.reset(seed=seed)

        total_volume = 0
        boxes_placed = 0
        failed_placements = 0
        episode_time = 0

        for step in range(env.n_boxes):
            start_time = time.time()

            if hasattr(algorithm, 'predict') and callable(algorithm.predict):
                if algorithm_name in ["Greedy Algorithm", "MLP Imitation Learning"]:
                    action = algorithm.predict(obs, env)
                else:
                    action, _ = algorithm.predict(obs, deterministic=True)
            else:
                action = algorithm(obs, env)

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

        if i % 5 == 0:
            print(f"  Episode {i + 1}/{n_episodes}: Volume = {total_volume:.1f}, Success = {success_rate:.3f}")

    return results


def create_comparison_visualization(results_dict, save_path=None):
    """Create comprehensive comparison visualization"""
    algorithms = list(results_dict.keys())
    n_algs = len(algorithms)

    # Extract metrics
    volumes = [np.array(results_dict[alg]['volumes']) for alg in algorithms]
    success_rates = [np.array(results_dict[alg]['success_rates']) for alg in algorithms]
    efficiencies = [np.array(results_dict[alg]['volume_efficiencies']) for alg in algorithms]
    decision_times = [np.array(results_dict[alg]['decision_times']) for alg in algorithms]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('3D Palletizing Algorithm Comparison', fontsize=16, fontweight='bold')

    colors = ['steelblue', 'forestgreen', 'orange', 'purple']

    # Volume comparison
    ax = axes[0, 0]
    means = [np.mean(vol) for vol in volumes]
    stds = [np.std(vol) for vol in volumes]
    x_pos = range(n_algs)

    bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                  color=colors[:n_algs], alpha=0.8, edgecolor='black')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Average Volume Placed')
    ax.set_title('Volume Efficiency Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 1, f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')

    # Success rate comparison
    ax = axes[0, 1]
    success_means = [np.mean(sr) for sr in success_rates]
    success_stds = [np.std(sr) for sr in success_rates]

    bars = ax.bar(x_pos, success_means, yerr=success_stds, capsize=5,
                  color=colors[:n_algs], alpha=0.8, edgecolor='black')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Success Rate')
    ax.set_title('Placement Success Rate')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Volume efficiency comparison
    ax = axes[1, 0]
    eff_means = [np.mean(eff) for eff in efficiencies]
    eff_stds = [np.std(eff) for eff in efficiencies]

    bars = ax.bar(x_pos, eff_means, yerr=eff_stds, capsize=5,
                  color=colors[:n_algs], alpha=0.8, edgecolor='black')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Volume Efficiency (fraction of max)')
    ax.set_title('Space Utilization Efficiency')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Decision time comparison
    ax = axes[1, 1]
    time_means = [np.mean(dt) * 1000 for dt in decision_times]  # Convert to ms
    time_stds = [np.std(dt) * 1000 for dt in decision_times]

    bars = ax.bar(x_pos, time_means, yerr=time_stds, capsize=5,
                  color=colors[:n_algs], alpha=0.8, edgecolor='black')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Average Decision Time (ms)')
    ax.set_title('Computational Efficiency')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_side_by_side_visualization(algorithms_dict, seed=42):
    """Create side-by-side 3D visualizations for comparison"""
    n_algorithms = len(algorithms_dict)
    fig = plt.figure(figsize=(6 * n_algorithms, 8))

    algorithm_names = list(algorithms_dict.keys())

    for i, (name, algorithm) in enumerate(algorithms_dict.items()):
        # Run episode with same seed
        env = OptimizedPalletEnv()
        obs, _ = env.reset(seed=seed)

        for step in range(env.n_boxes):
            if hasattr(algorithm, 'predict') and callable(algorithm.predict):
                if name in ["Greedy Algorithm", "MLP Imitation Learning"]:
                    action = algorithm.predict(obs, env)
                else:
                    action, _ = algorithm.predict(obs, deterministic=True)
            else:
                action = algorithm(obs, env)

            obs, reward, done, _, info = env.step(action)
            if done:
                break

        # Create subplot
        ax = fig.add_subplot(2, n_algorithms, i + 1, projection='3d')

        # Render the result
        if len(env.placed_boxes) > 0:
            colors = plt.cm.tab10(np.linspace(0, 1, len(env.placed_boxes)))

            for j, box in enumerate(env.placed_boxes):
                x, y, z = box.position
                l, w, h = box.size
                env._draw_box(ax, x, y, z, l, w, h, colors[j])

        ax.set_xlim(0, env.pallet_size[0])
        ax.set_ylim(0, env.pallet_size[1])
        ax.set_zlim(0, env.pallet_size[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        success_rate = env.successful_placements / env.n_boxes
        volume_efficiency = env.total_volume_placed / env.max_volume

        ax.set_title(f'{name}\n'
                     f'Placed: {env.successful_placements}/{env.n_boxes}\n'
                     f'Volume: {env.total_volume_placed} '
                     f'(Eff: {volume_efficiency:.1%})')

        # Create height map subplot
        ax2 = fig.add_subplot(2, n_algorithms, i + 1 + n_algorithms)
        height_map = env._get_height_map()
        im = ax2.imshow(height_map, cmap='viridis', origin='lower')
        ax2.set_title(f'{name} - Height Map')
        ax2.set_xlabel('Y')
        ax2.set_ylabel('X')
        plt.colorbar(im, ax=ax2, shrink=0.6)

    plt.tight_layout()
    plt.show()


def print_comparison_summary(results_dict):
    """Print detailed comparison summary"""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE ALGORITHM COMPARISON SUMMARY")
    print("=" * 100)

    print(f"{'Algorithm':<25} {'Volume':<15} {'Success':<12} {'Efficiency':<12} {'Time (ms)':<12} {'Status':<10}")
    print("-" * 100)

    algorithms = list(results_dict.keys())
    best_volume = 0
    best_algorithm = ""

    for algorithm in algorithms:
        results = results_dict[algorithm]

        avg_volume = np.mean(results['volumes'])
        std_volume = np.std(results['volumes'])
        avg_success = np.mean(results['success_rates'])
        avg_efficiency = np.mean(results['volume_efficiencies'])
        avg_time = np.mean(results['decision_times']) * 1000

        if avg_volume > best_volume:
            best_volume = avg_volume
            best_algorithm = algorithm

        status = "BEST" if algorithm == best_algorithm else "GOOD" if avg_efficiency > 0.7 else "FAIR"

        print(f"{algorithm:<25} {avg_volume:<7.2f}±{std_volume:<5.2f} "
              f"{avg_success:<12.3f} {avg_efficiency:<12.3f} "
              f"{avg_time:<12.3f} {status:<10}")

    print("\n" + "=" * 100)
    print(f"BEST PERFORMING ALGORITHM: {best_algorithm}")
    print(f"BEST VOLUME ACHIEVED: {best_volume:.2f}")
    print("=" * 100)


def main():
    """Main comparison function"""
    print("🚀 Comprehensive 3D Palletizing Algorithm Comparison")
    print("=" * 60)

    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    # Initialize algorithms
    algorithms = {}

    # 1. Greedy Algorithm (friend's approach)
    print("1. Initializing Greedy Algorithm...")
    greedy = UpgradedGreedyAlgorithm()
    algorithms[greedy.name] = greedy

    # 2. MLP Imitation Learning (friend's approach)
    print("2. Training MLP Imitation Learning...")
    mlp = UpgradedMLPAlgorithm()
    mlp.train_on_demonstrations(n_episodes=500, epochs=150)
    algorithms[mlp.name] = mlp

    # 3. PPO (baseline RL)
    print("3. Training PPO...")

    def make_env():
        return OptimizedPalletEnv()

    vec_env = DummyVecEnv([make_env])
    ppo_model = create_ppo_model(vec_env)

    loss_callback = LossTrackingCallback()
    print("   Training PPO for 200,000 timesteps...")
    ppo_model.learn(total_timesteps=200000, callback=loss_callback, progress_bar=True)
    algorithms["PPO"] = ppo_model

    # 4. DQN (your implementation)
    print("4. Training DQN...")
    dqn_env = DummyVecEnv([make_env])
    dqn_model = create_optimized_dqn_model(dqn_env, 384)  # Use best configuration
    print("   Training DQN for 200,000 timesteps...")
    dqn_model.learn(total_timesteps=200000, progress_bar=True)
    algorithms["DQN"] = dqn_model

    # Evaluate all algorithms
    print("\n🔍 Evaluating all algorithms...")
    results = {}
    evaluation_seeds = list(range(2000, 2015))  # Use different seeds for evaluation

    for name, algorithm in algorithms.items():
        results[name] = evaluate_algorithm(algorithm, name,
                                           n_episodes=15, seeds=evaluation_seeds)

    # Create visualizations
    print("\n📊 Creating comparison visualizations...")
    create_comparison_visualization(results, save_path="algorithm_comparison.png")
    create_side_by_side_visualization(algorithms, seed=999)

    # Print summary
    print_comparison_summary(results)

    # Save results
    with open('algorithm_comparison_results.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'algorithms': list(algorithms.keys()),
            'evaluation_seeds': evaluation_seeds,
            'environment': 'OptimizedPalletEnv'
        }, f)

    print("\n✅ Analysis completed and saved to 'algorithm_comparison_results.pkl'")


if __name__ == "__main__":
    main()