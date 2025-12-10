#!/bin/bash
# Batch script to run all configurations

echo "============================================================"
echo "MADISON RL - Running All Configurations"
echo "============================================================"
echo ""

# Array of all config files
configs=(
    "config/config_simple_2000.yaml"
    "config/config_simple_5000.yaml"
    "config/config_simple_10000.yaml"
    "config/config_medium_2000.yaml"
    "config/config_medium_5000.yaml"
    "config/config_medium_10000.yaml"
    "config/config_complex_2000.yaml"
    "config/config_complex_5000.yaml"
    "config/config_complex_10000.yaml"
)

total=${#configs[@]}
current=0
successful=0
failed=0

start_time=$(date +%s)

for config in "${configs[@]}"; do
    current=$((current + 1))
    config_name=$(basename "$config" .yaml | sed 's/config_//')
    
    echo "============================================================"
    echo "[$current/$total] Running: $config_name"
    echo "============================================================"
    
    if python3 experiments/train_madison_rl.py --config "$config"; then
        echo "✅ $config_name completed successfully"
        successful=$((successful + 1))
    else
        echo "❌ $config_name failed"
        failed=$((failed + 1))
    fi
    
    echo ""
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))

echo "============================================================"
echo "BATCH TRAINING SUMMARY"
echo "============================================================"
echo "Total Configs: $total"
echo "Successful: $successful"
echo "Failed: $failed"
echo "Total Time: ${minutes}m ${seconds}s"
echo "============================================================"

