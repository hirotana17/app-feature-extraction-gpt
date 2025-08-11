#!/usr/bin/env python3
import os
from pathlib import Path

# Base directory
base_dir = "./data"

# Domain configurations
domains = {
    "in-domain": [f"bin{i}" for i in range(10)],  # bin0 to bin9
    "out-of-domain": [
        "COMMUNICATION", "HEALTH_AND_FITNESS", "LIFESTYLE", "MAPS_AND_NAVIGATION",
        "PERSONALIZATION", "PRODUCTIVITY", "SOCIAL", "TOOLS", "TRAVEL_AND_LOCAL", "WEATHER"
    ]
}

# Subdirectories to create
subdirs = [
    "original_data",
    "formatted_original_data", 
    "fine_tuning_data",
    "feature_extracted_data",
    "evaluation_result"
]

# Create directories
for domain, categories in domains.items():
    domain_path = os.path.join(base_dir, domain)
    
    for category in categories:
        category_path = os.path.join(domain_path, category)
        
        # Create subdirectories
        for subdir in subdirs:
            subdir_path = os.path.join(category_path, subdir)
            Path(subdir_path).mkdir(parents=True, exist_ok=True)
            print(f"Created: {subdir_path}")

print("Directory creation completed!")
