import os
import json

def generate_multiclass_config(output_file):
    class_file = {
        "classes": ["blood_vessel"]
    }
    with open(output_file, "w") as f:
        json.dump(class_file, f, indent=4)

def load_multiclass_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    classes = config["classes"]
    return config, classes, len(classes)

def save_multiclass_config(config_file, config):
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)