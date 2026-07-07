import argparse
from camtrap_lab.runner import run_experiment

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run a camtrap detector experiment from a YAML config.")
    ap.add_argument("--config", required=True, help="Path to experiment YAML")
    args = ap.parse_args()
    run_experiment(args.config)
