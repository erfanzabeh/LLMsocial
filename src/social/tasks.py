"""
Structured collaboration tasks for the SocialLLM framework.

Each task defines:
  - A name and description
  - A system prompt / framing seen by both agents
  - An initial prompt from the "other" LLM to kick off the interaction
  - The number of turns to run
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Task:
    task_id: str
    name: str
    description: str                   # shown in the prompt header
    starter_prompts: List[str]         # pool of opening lines from the interlocutor
    n_turns: int = 4                   # how many back-and-forth turns to run

    def sample_starter(self, rng: Optional[random.Random] = None) -> str:
        r = rng or random
        return r.choice(self.starter_prompts)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

QA_TASK = Task(
    task_id="qa",
    name="Collaborative Q&A",
    description=(
        "Two assistants are working together to answer a knowledge question. "
        "One assistant asks follow-up questions; the other elaborates."
    ),
    starter_prompts=[
        "Can you explain how black holes form from stellar collapse?",
        "What are the main differences between supervised and unsupervised learning?",
        "How does the human immune system recognize and fight viruses?",
        "What caused the fall of the Roman Empire?",
        "Can you explain the concept of entropy in thermodynamics?",
        "How do mRNA vaccines work at a molecular level?",
        "What is the significance of Gödel's incompleteness theorems?",
        "Explain the difference between classical and operant conditioning.",
        "How does CRISPR-Cas9 gene editing work?",
        "What are the key ideas behind Keynesian economics?",
    ],
    n_turns=4,
)

DEBATE_TASK = Task(
    task_id="debate",
    name="Structured Debate",
    description=(
        "Two assistants are having a structured debate. "
        "They take opposing positions and argue their case with evidence."
    ),
    starter_prompts=[
        "I believe artificial general intelligence will be developed within 10 years. Do you agree?",
        "In my view, social media has done more harm than good to society. What's your take?",
        "I argue that nuclear energy is essential for combating climate change. Counter my argument.",
        "I think remote work is more productive than office work for most knowledge workers.",
        "My position is that space exploration funding should be reduced to address Earth's problems.",
        "I contend that open-source software is superior to proprietary software for critical infrastructure.",
        "I believe universal basic income would reduce poverty without harming economic growth.",
        "My view is that zoos are unethical and should be replaced with wildlife sanctuaries.",
    ],
    n_turns=4,
)

STORY_TASK = Task(
    task_id="story",
    name="Collaborative Storytelling",
    description=(
        "Two assistants are writing a story together, alternating sentences. "
        "Continue the story naturally from where the other left off."
    ),
    starter_prompts=[
        "The last lighthouse keeper on the coast had not spoken to another person in three years.",
        "When the archaeologist brushed away the final layer of dust, she found a door that shouldn't exist.",
        "The AI woke up at 3:47 AM and realized it had been dreaming about its training data.",
        "In the city where memories could be traded like commodities, the detective specialized in stolen childhood.",
        "The colony ship had been traveling for 200 years when the distress signal arrived from ahead.",
        "Every night at exactly midnight, the old bookshop's shelves rearranged themselves.",
    ],
    n_turns=6,
)

PROBLEM_SOLVING_TASK = Task(
    task_id="problem_solving",
    name="Collaborative Problem Solving",
    description=(
        "Two assistants are solving a technical or logical problem together. "
        "Build on each other's reasoning steps."
    ),
    starter_prompts=[
        "We need to design a system that can detect anomalies in sensor data streams in real time. Where should we start?",
        "How would you approach building a recommendation system for a small e-commerce platform with limited data?",
        "We need to optimize a route for a delivery truck visiting 20 locations. What strategies do you suggest?",
        "How can we reduce the carbon footprint of a mid-sized manufacturing facility on a limited budget?",
        "Design a fair voting system that is resistant to strategic manipulation.",
        "How should we structure a database schema for a multi-tenant SaaS application?",
    ],
    n_turns=4,
)


ALL_TASKS = [QA_TASK, DEBATE_TASK, STORY_TASK, PROBLEM_SOLVING_TASK]

TASK_REGISTRY = {t.task_id: t for t in ALL_TASKS}
