"""
Step 1 — Run social interactions and collect data.

Usage:
    python experiments/run_interactions.py [--config config/config.yaml]
                                           [--device cpu|cuda]
                                           [--n_episodes 5]
                                           [--tasks qa debate]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.social.agents import LLMAgent, TargetAgent, make_interlocutor_configs, make_target_config
from src.social.environment import InteractionDatasetRunner
from src.social.tasks import TASK_REGISTRY, ALL_TASKS
from src.activation.extractor import ActivationDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    p.add_argument("--n_episodes", type=int, default=None)
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Task IDs to run (e.g. qa debate)")
    p.add_argument("--dry_run", action="store_true",
                   help="Load models but run only 1 episode per condition")
    p.add_argument("--target_model", default=None,
                   help="Override target model (e.g. gpt2 for quick test)")
    p.add_argument("--interlocutor_model", default=None,
                   help="Override interlocutor model (default: same as target_model or gpt2)")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or cfg["experiment"].get("device", "cpu")
    n_episodes = args.n_episodes or cfg["data_collection"]["n_episodes_per_condition"]
    if args.dry_run:
        n_episodes = 1

    task_ids = args.tasks or cfg["tasks"]
    tasks = [TASK_REGISTRY[tid] for tid in task_ids if tid in TASK_REGISTRY]
    if not tasks:
        logger.error(f"No valid tasks found in: {task_ids}")
        sys.exit(1)

    # Model overrides (useful for quick CPU tests with small gpt2)
    target_model = args.target_model or cfg["target"].get("model_name", "gpt2-xl")
    interlocutor_model = args.interlocutor_model or target_model

    # Build agents
    logger.info(f"Loading target model ({target_model}) on {device}…")
    target_cfg = make_target_config(device=device, model_name=target_model)
    target = TargetAgent(target_cfg)
    logger.info("Target loaded.")

    interlocutor_cfgs = make_interlocutor_configs(device=device, model_name=interlocutor_model)
    interlocutors = []
    for ic in interlocutor_cfgs:
        logger.info(f"Loading interlocutor: {ic.agent_id} ({ic.model_name})…")
        interlocutors.append(LLMAgent(ic))

    # Interaction dataset runner
    interactions_dir = cfg["output"]["interactions_dir"]
    runner = InteractionDatasetRunner(
        target=target,
        interlocutors=interlocutors,
        tasks=tasks,
        output_dir=interactions_dir,
        n_episodes_per_condition=n_episodes,
        seed=cfg["experiment"]["seed"],
    )

    logger.info(
        f"Starting data collection: "
        f"{len(interlocutors)} interlocutors × {len(tasks)} tasks × {n_episodes} episodes"
    )
    interactions = runner.run_all(verbose=True)
    logger.info(f"Collected {len(interactions)} interactions.")

    # Immediately convert to activation dataset
    activation_cfg = cfg["activation"]
    ds = ActivationDataset(
        output_path=activation_cfg["output_path"],
        aggregation=activation_cfg["aggregation"],
    )
    ds.add_from_interactions(interactions)
    ds.save()
    logger.info("Activation dataset saved.")


if __name__ == "__main__":
    main()
