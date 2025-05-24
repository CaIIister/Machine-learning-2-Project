# Quick test script to validate the DQN implementation
# Run this first to make sure everything works before running full experiments

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from palletizing_dqn import Enhanced3DBoxEnv
import time


# Import your main classes (assuming they're in the same file or imported)
# from your_main_file import Enhanced3DBoxEnv

def quick_test():
    """Quick test to validate the implementation"""
    print("Quick Test of DQN 3D Palletizing")
    print("=" * 40)

    # Test 1: Environment Creation
    print("1. Testing environment creation...")
    try:
        env = Enhanced3DBoxEnv()
        obs, _ = env.reset()
        print(f"   ✓ Environment created successfully")
        print(f"   ✓ Observation space: {env.observation_space.shape}")
        print(f"   ✓ Action space: {env.action_space.n}")
    except Exception as e:
        print(f"   ✗ Error creating environment: {e}")
        return False

    # Test 2: Random Actions
    print("\n2. Testing random actions...")
    try:
        total_reward = 0
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            if done:
                break
        print(f"   ✓ Random actions work, total reward: {total_reward:.2f}")
    except Exception as e:
        print(f"   ✗ Error with random actions: {e}")
        return False

    # Test 3: DQN Model Creation
    print("\n3. Testing DQN model creation...")
    try:
        vec_env = DummyVecEnv([lambda: Enhanced3DBoxEnv()])
        model = DQN("MlpPolicy", vec_env, verbose=0)
        print(f"   ✓ DQN model created successfully")
    except Exception as e:
        print(f"   ✗ Error creating DQN model: {e}")
        return False

    # Test 4: Short Training
    print("\n4. Testing short training (1000 steps)...")
    try:
        start_time = time.time()
        model.learn(total_timesteps=1000)
        training_time = time.time() - start_time
        print(f"   ✓ Training completed in {training_time:.2f} seconds")
    except Exception as e:
        print(f"   ✗ Error during training: {e}")
        return False

    # Test 5: Model Prediction
    print("\n5. Testing model prediction...")
    try:
        env = Enhanced3DBoxEnv()
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        print(f"   ✓ Model prediction works, action: {action}")
    except Exception as e:
        print(f"   ✗ Error with model prediction: {e}")
        return False

    # Test 6: Full Episode
    print("\n6. Testing full episode...")
    try:
        env = Enhanced3DBoxEnv()
        obs, _ = env.reset()
        total_volume = 0
        boxes_placed = 0

        for step in range(20):  # Test first 20 boxes
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

            if info['placed']:
                boxes_placed += 1
                total_volume += info['volume']

            if done:
                break

        print(f"   ✓ Full episode test: {boxes_placed} boxes placed, volume: {total_volume}")
    except Exception as e:
        print(f"   ✗ Error during full episode: {e}")
        return False

    print("\n" + "=" * 40)
    print("✓ ALL TESTS PASSED! Ready for full experiments.")
    return True


def run_single_experiment():
    """Run a single quick experiment for demonstration"""
    print("\nRunning single experiment (64 neurons, 5000 steps)...")

    # Create environment
    vec_env = DummyVecEnv([lambda: Enhanced3DBoxEnv()])

    # Create model
    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=0.001,
        buffer_size=5000,
        learning_starts=500,
        batch_size=32,
        verbose=0
    )

    # Train
    start_time = time.time()
    model.learn(total_timesteps=5000)
    training_time = time.time() - start_time

    # Evaluate
    env = Enhanced3DBoxEnv()
    obs, _ = env.reset(seed=42)

    total_volume = 0
    boxes_placed = 0

    for step in range(env.n_boxes):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        if info['placed']:
            boxes_placed += 1
            total_volume += info['volume']

        if done:
            break

    success_rate = boxes_placed / env.n_boxes

    print(f"\nSingle Experiment Results:")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Boxes placed: {boxes_placed}/{env.n_boxes}")
    print(f"  Success rate: {success_rate:.3f}")
    print(f"  Total volume: {total_volume}")

    # Show visualization
    env.render("DQN Quick Test Result")

    return model, {
        'training_time': training_time,
        'boxes_placed': boxes_placed,
        'success_rate': success_rate,
        'total_volume': total_volume
    }


if __name__ == "__main__":
    # Run quick validation
    if quick_test():
        # Run single experiment
        model, results = run_single_experiment()

        print("\n" + "=" * 50)
        print("Quick test completed successfully!")
        print("You can now run the full experiment script.")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("Tests failed. Please check the errors above.")
        print("=" * 50)