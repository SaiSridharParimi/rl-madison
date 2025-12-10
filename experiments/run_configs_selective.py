"""
Run selected configurations (interactive)
"""

import subprocess
import sys
from pathlib import Path
from time import time

# All available configurations
ALL_CONFIGS = {
    'simple_2000': 'config/config_simple_2000.yaml',
    'simple_5000': 'config/config_simple_5000.yaml',
    'simple_10000': 'config/config_simple_10000.yaml',
    'medium_2000': 'config/config_medium_2000.yaml',
    'medium_5000': 'config/config_medium_5000.yaml',
    'medium_10000': 'config/config_medium_10000.yaml',
    'complex_2000': 'config/config_complex_2000.yaml',
    'complex_5000': 'config/config_complex_5000.yaml',
    'complex_10000': 'config/config_complex_10000.yaml',
}

def print_menu():
    """Print configuration menu"""
    print("\n" + "="*60)
    print("MADISON RL - CONFIGURATION SELECTOR")
    print("="*60)
    print("\nAvailable Configurations:")
    print()
    
    configs_list = list(ALL_CONFIGS.items())
    
    # Group by complexity
    print("SIMPLE (Fast convergence):")
    for i, (name, path) in enumerate(configs_list[:3], 1):
        print(f"  {i}. {name}")
    
    print("\nMEDIUM (Balanced):")
    for i, (name, path) in enumerate(configs_list[3:6], 4):
        print(f"  {i}. {name}")
    
    print("\nCOMPLEX (Scalability demo):")
    for i, (name, path) in enumerate(configs_list[6:9], 7):
        print(f"  {i}. {name}")
    
    print("\nOptions:")
    print("  a. Run all configurations")
    print("  s. Run all simple configurations")
    print("  m. Run all medium configurations")
    print("  c. Run all complex configurations")
    print("  q. Quit")
    print()

def run_config(config_path: str):
    """Run a single configuration"""
    config_name = Path(config_path).stem.replace('config_', '')
    print(f"\n{'='*60}")
    print(f"Running: {config_name.upper()}")
    print(f"{'='*60}\n")
    
    try:
        subprocess.run(
            [sys.executable, 'experiments/train_madison_rl.py', '--config', config_path],
            cwd=Path(__file__).parent.parent,
            check=True
        )
        print(f"\n✅ {config_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {config_name} failed!")
        return False

def main():
    """Interactive configuration runner"""
    while True:
        print_menu()
        choice = input("Select option: ").strip().lower()
        
        if choice == 'q':
            print("Exiting...")
            break
        
        elif choice == 'a':
            # Run all
            configs_to_run = list(ALL_CONFIGS.values())
            print(f"\nRunning all {len(configs_to_run)} configurations...")
        
        elif choice == 's':
            # Run simple
            configs_to_run = list(ALL_CONFIGS.values())[:3]
            print(f"\nRunning {len(configs_to_run)} simple configurations...")
        
        elif choice == 'm':
            # Run medium
            configs_to_run = list(ALL_CONFIGS.values())[3:6]
            print(f"\nRunning {len(configs_to_run)} medium configurations...")
        
        elif choice == 'c':
            # Run complex
            configs_to_run = list(ALL_CONFIGS.values())[6:9]
            print(f"\nRunning {len(configs_to_run)} complex configurations...")
        
        elif choice.isdigit():
            # Run specific config
            idx = int(choice) - 1
            if 0 <= idx < len(ALL_CONFIGS):
                configs_to_run = [list(ALL_CONFIGS.values())[idx]]
            else:
                print("Invalid selection!")
                continue
        else:
            print("Invalid option!")
            continue
        
        # Run selected configs
        total_start = time()
        successful = 0
        failed = 0
        
        for i, config_path in enumerate(configs_to_run, 1):
            print(f"\n[{i}/{len(configs_to_run)}]")
            if run_config(config_path):
                successful += 1
            else:
                failed += 1
        
        total_time = time() - total_start
        
        print("\n" + "="*60)
        print("BATCH SUMMARY")
        print("="*60)
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total Time: {total_time/60:.2f} minutes")
        print("="*60)
        
        if len(configs_to_run) > 1:
            input("\nPress Enter to continue...")

if __name__ == '__main__':
    main()

