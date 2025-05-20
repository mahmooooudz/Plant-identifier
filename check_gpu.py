# check_gpu.py
# Script to verify GPU availability and optimize TensorFlow settings

import tensorflow as tf
import os
import time
import numpy as np

def check_gpu():
    """Check for GPU availability and optimize settings"""
    print("\n===== GPU Check and Optimization =====")
    
    # Check if TensorFlow can see the GPU
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ No GPU found. Training will use CPU only (much slower).")
        print("   Make sure you have CUDA and cuDNN installed correctly.")
        return False
        
    print(f"✅ Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")
    
    # Print GPU details
    gpu_details = tf.config.experimental.get_device_details(gpus[0])
    if gpu_details and 'compute_capability' in gpu_details:
        cc = gpu_details['compute_capability']
        print(f"   Compute capability: {cc[0]}.{cc[1]}")
    
    # Configure GPU memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"❌ Error setting memory growth: {e}")
    
    # Enable mixed precision
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("✅ Mixed precision enabled (float16/float32)")
    except Exception as e:
        print(f"❌ Could not enable mixed precision: {e}")
    
    # Check if TensorFlow operations run on GPU
    print("\nRunning GPU test...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([10000, 10000])
        b = tf.random.normal([10000, 10000])
        start_time = time.time()
        c = tf.matmul(a, b)
        # Force execution with .numpy()
        c.numpy()
        gpu_time = time.time() - start_time
    
    # Run the same operation on CPU for comparison
    with tf.device('/CPU:0'):
        a = tf.random.normal([10000, 10000])
        b = tf.random.normal([10000, 10000])
        start_time = time.time()
        c = tf.matmul(a, b)
        c.numpy()
        cpu_time = time.time() - start_time
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    
    print(f"\nMatrix multiplication test:")
    print(f"   GPU time: {gpu_time:.2f} seconds")
    print(f"   CPU time: {cpu_time:.2f} seconds")
    print(f"   Speedup: {speedup:.1f}x")
    
    if speedup < 5:
        print("\n⚠️ Warning: GPU acceleration seems lower than expected.")
        print("   This might indicate suboptimal configuration.")
        print("   - Make sure your NVIDIA drivers are up to date")
        print("   - Ensure you have the matching CUDA version for TensorFlow")
    else:
        print("\n🚀 GPU is working correctly and providing good acceleration!")
    
    # RTX 4060 specific optimizations
    print("\nOptimizing for RTX 4060:")
    
    # Check and set optimal batch size
    vram_mb = 8 * 1024  # 8GB VRAM
    recommended_batch_size = min(32, max(8, vram_mb // 256))
    print(f"   Recommended batch size: {recommended_batch_size}")
    
    # Calculate expected training time based on test results
    epochs = 30
    fine_tune_epochs = 15
    estimated_steps = 150  # Assumption based on your dataset size
    
    # Base estimate on matrix test speedup
    estimate_per_epoch = 60 / speedup if speedup > 0 else 60  # 1 minute per epoch on fast GPU
    total_estimate = (epochs + fine_tune_epochs) * estimate_per_epoch
    
    print(f"   Estimated training time: {total_estimate/60:.1f} hours ({total_estimate:.0f} minutes)")
    
    # Return the recommended batch size
    return {
        "working": speedup > 2,
        "recommended_batch_size": recommended_batch_size,
        "estimated_time_minutes": total_estimate
    }

if __name__ == "__main__":
    result = check_gpu()
    
    if result and result["working"]:
        print("\n===== Recommended Training Command =====")
        batch_size = result["recommended_batch_size"]
        estimated_time = result["estimated_time_minutes"]
        
        print(f"python scripts/train_layer1.py --data_dir data/train --model_dir models/layer1_is_plant \\")
        print(f"--epochs 30 --fine_tune_epochs 15 --batch_size {batch_size} --class_weight \\")
        print(f"--dropout_rate 0.3 --l2_reg 0.001 --unfreeze_layers 80")
        
        print(f"\nEstimated training time: {estimated_time/60:.1f} hours")
        print("\nTo start training with these optimized settings, run:")
        print("run_training_enhanced.bat")
    else:
        print("\n❌ GPU setup has issues. Consider using CPU training instead (will be slower):")
        print("python scripts/train_layer1.py --data_dir data/train --model_dir models/layer1_is_plant --batch_size 8")