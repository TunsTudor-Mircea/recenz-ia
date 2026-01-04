#!/bin/bash
set -e

# Create symlink for preprocessing module if it doesn't exist
if [ ! -L "/app/preprocessing" ]; then
    ln -s /app/app/preprocessing /app/preprocessing
    echo "Created symlink: /app/preprocessing -> /app/app/preprocessing"
fi

# Create symlink for features module if it doesn't exist
if [ ! -L "/app/features" ]; then
    ln -s /app/app/features /app/features
    echo "Created symlink: /app/features -> /app/app/features"
fi

# Execute the command passed to the container
exec "$@"
