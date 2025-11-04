#!/bin/bash
# Check available storage on all mounted volumes

echo "=== Disk Space Check ==="
echo ""
echo "All mounted filesystems:"
df -h
echo ""
echo "=== Largest Available Volumes ==="
df -h | sort -k 4 -h | tail -5
echo ""
echo "=== Current Directory Space ==="
df -h .
echo ""
echo "=== Home Directory Space ==="
df -h ~
echo ""
echo "=== Workspace Directory (if exists) ==="
if [ -d /workspace ]; then
    df -h /workspace
else
    echo "/workspace does not exist"
fi
echo ""
echo "=== Recommendations ==="
echo "LongCat-Flash-Omni needs: 1.1TB minimum"
echo "For training with checkpoints: 2TB recommended"
