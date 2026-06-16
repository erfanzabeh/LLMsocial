"""
Social interaction environment.

Orchestrates multi-turn dialogue between the TARGET agent and an INTERLOCUTOR
agent, collecting activations from the target at each reading step.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .agents import LLMAgent, TargetAgent
from .tasks import Task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """One turn of dialogue."""
    turn_idx: int
    speaker: str          # "target" or interlocutor agent_id
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Interaction:
    """
    A complete interaction episode between the target and one interlocutor
    on one task.
    """
    interaction_id: str
    task_id: str
    target_id: str
    interlocutor_id: str
    turns: List[Turn] = field(default_factory=list)
    # Activation snapshots: list of dicts collected at each target-reading step
    # Each dict: {"turn_idx": int, "layer_name": str -> tensor (n_tokens, dim)}
    activation_snapshots: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialisable representation (without tensors)."""
        return {
            "interaction_id": self.interaction_id,
            "task_id": self.task_id,
            "target_id": self.target_id,
            "interlocutor_id": self.interlocutor_id,
            "turns": [
                {
                    "turn_idx": t.turn_idx,
                    "speaker": t.speaker,
                    "content": t.content,
                    "timestamp": t.timestamp,
                }
                for t in self.turns
            ],
        }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SocialEnvironment:
    """
    Manages a single episode of LLM–LLM interaction.

    Flow per turn:
      1. Interlocutor generates a message.
      2. Target reads that message → activations extracted.
      3. Target generates a response → added to history.
      4. Repeat.
    """

    def __init__(
        self,
        target: TargetAgent,
        interlocutor: LLMAgent,
        task: Task,
        rng: Optional[random.Random] = None,
        max_context_turns: int = 8,
    ):
        self.target = target
        self.interlocutor = interlocutor
        self.task = task
        self.rng = rng or random.Random()
        self.max_context_turns = max_context_turns

    # ------------------------------------------------------------------

    def _truncated_history(self, history: List[Dict]) -> List[Dict]:
        """Keep only the last `max_context_turns` turns to avoid OOM."""
        return history[-self.max_context_turns:]

    # ------------------------------------------------------------------

    def run_episode(
        self,
        interaction_id: Optional[str] = None,
        verbose: bool = False,
    ) -> Interaction:
        """
        Run one full interaction episode and return an Interaction object
        that includes per-turn activations from the target.
        """
        if interaction_id is None:
            interaction_id = (
                f"{self.task.task_id}__{self.interlocutor.agent_id}"
                f"__{int(time.time())}"
            )

        interaction = Interaction(
            interaction_id=interaction_id,
            task_id=self.task.task_id,
            target_id=self.target.agent_id,
            interlocutor_id=self.interlocutor.agent_id,
        )

        history: List[Dict[str, str]] = []   # shared conversation history
        turn_idx = 0

        # --- Opening move from interlocutor ---
        starter = self.task.sample_starter(self.rng)
        history.append({"role": self.interlocutor.agent_id, "content": starter})
        interaction.turns.append(Turn(turn_idx, self.interlocutor.agent_id, starter))
        if verbose:
            logger.info(f"[{turn_idx}] {self.interlocutor.agent_id}: {starter[:80]}…")
        turn_idx += 1

        for _ in range(self.task.n_turns):
            # -- Target reads the last interlocutor message → extract activations --
            last_interlocutor_msg = history[-1]["content"]
            acts = self.target.read_and_extract(
                text=last_interlocutor_msg,
                interlocutor_id=self.interlocutor.agent_id,
            )
            snapshot = {"turn_idx": turn_idx - 1}   # the turn we just read
            snapshot.update(acts)
            interaction.activation_snapshots.append(snapshot)

            # -- Target generates a response --
            target_response, _ = self.target.generate_with_activations(
                history=self._truncated_history(history),
                task_description=self.task.description,
                interlocutor_id=self.interlocutor.agent_id,
            )
            history.append({"role": self.target.agent_id, "content": target_response})
            interaction.turns.append(Turn(turn_idx, self.target.agent_id, target_response))
            if verbose:
                logger.info(f"[{turn_idx}] TARGET: {target_response[:80]}…")
            turn_idx += 1

            # -- Interlocutor responds --
            interlocutor_response = self.interlocutor.generate(
                history=self._truncated_history(history),
                task_description=self.task.description,
            )
            history.append({"role": self.interlocutor.agent_id, "content": interlocutor_response})
            interaction.turns.append(Turn(turn_idx, self.interlocutor.agent_id, interlocutor_response))
            if verbose:
                logger.info(f"[{turn_idx}] {self.interlocutor.agent_id}: {interlocutor_response[:80]}…")
            turn_idx += 1

        return interaction


# ---------------------------------------------------------------------------
# Dataset runner — runs all combinations and saves results
# ---------------------------------------------------------------------------

class InteractionDatasetRunner:
    """
    Runs N episodes for every (interlocutor, task) combination and saves
    the results to disk.
    """

    def __init__(
        self,
        target: TargetAgent,
        interlocutors: List[LLMAgent],
        tasks: List[Task],
        output_dir: str = "data/interactions",
        n_episodes_per_condition: int = 5,
        seed: int = 42,
    ):
        self.target = target
        self.interlocutors = interlocutors
        self.tasks = tasks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_episodes = n_episodes_per_condition
        self.rng = random.Random(seed)

    def run_all(self, verbose: bool = True) -> List[Interaction]:
        all_interactions: List[Interaction] = []
        total_runs = len(self.interlocutors) * len(self.tasks) * self.n_episodes
        completed_runs = 0

        for interlocutor in self.interlocutors:
            for task in self.tasks:
                logger.info(
                    f"Running: interlocutor={interlocutor.agent_id}, task={task.task_id}"
                )
                env = SocialEnvironment(
                    target=self.target,
                    interlocutor=interlocutor,
                    task=task,
                    rng=self.rng,
                )
                for ep in range(self.n_episodes):
                    iid = f"{task.task_id}__{interlocutor.agent_id}__ep{ep:03d}"
                    if verbose:
                        print(
                            f"[{completed_runs + 1}/{total_runs}] "
                            f"persona={interlocutor.agent_id} | "
                            f"task={task.task_id} | "
                            f"episode={ep + 1}/{self.n_episodes}",
                            flush=True,
                        )
                    try:
                        interaction = env.run_episode(
                            interaction_id=iid, verbose=verbose
                        )
                        all_interactions.append(interaction)
                        self._save_interaction(interaction)
                        completed_runs += 1
                        if verbose:
                            print(
                                f"Completed {iid} "
                                f"({completed_runs}/{total_runs})",
                                flush=True,
                            )
                    except Exception as exc:
                        completed_runs += 1
                        logger.error(f"Episode {iid} failed: {exc}")
                        if verbose:
                            print(
                                f"Failed {iid} "
                                f"({completed_runs}/{total_runs})",
                                flush=True,
                            )

        return all_interactions

    def _save_interaction(self, interaction: Interaction):
        """Save the JSON transcript (activations saved separately via extractor)."""
        path = self.output_dir / f"{interaction.interaction_id}.json"
        with open(path, "w") as f:
            json.dump(interaction.to_dict(), f, indent=2)
