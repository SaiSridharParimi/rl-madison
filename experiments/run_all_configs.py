"""
Run all configuration files automatically
"""

import subprocess
import sys
from pathlib import Path
from time import time
import json

# All configuration files to run
CONFIGS = [
    'config/config_simple_2000.yaml',
    'config/config_simple_5000.yaml',
    'config/config_simple_10000.yaml',
    'config/config_medium_2000.yaml',
    'config/config_medium_5000.yaml',
    'config/config_medium_10000.yaml',
    'config/config_complex_2000.yaml',
    'config/config_complex_5000.yaml',
    'config/config_complex_10000.yaml',
]

def run_config(config_path: str) -> dict:
    """Run a single configuration and return results"""
    config_name = Path(config_path).stem.replace('config_', '')
    print(f"\n{'='*60}")
    print(f"Running: {config_name.upper()}")
    print(f"{'='*60}")
    
    start_time = time()
    
    try:
        # Run training
        result = subprocess.run(
            [sys.executable, 'experiments/train_madison_rl.py', '--config', config_path],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed_time = time() - start_time
        
        print(f"✅ {config_name} completed in {elapsed_time:.2f} seconds")
        
        return {
            'config': config_name,
            'status': 'success',
            'time': elapsed_time,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    
    except subprocess.CalledProcessError as e:
        elapsed_time = time() - start_time
        print(f"❌ {config_name} failed after {elapsed_time:.2f} seconds")
        print(f"Error: {e.stderr}")
        
        return {
            'config': config_name,
            'status': 'failed',
            'time': elapsed_time,
            'error': str(e),
            'stderr': e.stderr
        }

def main():
    """Run all configurations"""
    print("="*60)
    print("MADISON RL - BATCH TRAINING")
    print("Running all configurations...")
    print("="*60)
    
    results = []
    total_start = time()
    
    for i, config_path in enumerate(CONFIGS, 1):
        print(f"\n[{i}/{len(CONFIGS)}] Processing {Path(config_path).name}...")
        
        result = run_config(config_path)
        results.append(result)
        
        # Save progress after each run
        progress_file = Path('models/training_progress.json')
        progress_file.parent.mkdir(exist_ok=True)
        with open(progress_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    total_time = time() - total_start
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"\nTotal Configs: {len(CONFIGS)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total Time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    
    if successful:
        print(f"\n✅ Successful Configurations:")
        for r in successful:
            print(f"  - {r['config']}: {r['time']:.2f}s")
    
    if failed:
        print(f"\n❌ Failed Configurations:")
        for r in failed:
            print(f"  - {r['config']}: {r.get('error', 'Unknown error')}")
    
    # Save final results
    results_file = Path('models/training_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'total_time': total_time,
            'successful': len(successful),
            'failed': len(failed),
            'results': results
        }, f, indent=2)
    
    print(f"\n📊 Results saved to: {results_file}")
    print(f"📊 Progress saved to: {Path('models/training_progress.json')}")
    print("\n" + "="*60)
    print("All configurations processed!")
    print("="*60)

if __name__ == '__main__':
    main()

