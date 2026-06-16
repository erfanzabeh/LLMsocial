"""
LLM Agent wrappers for the SocialLLM framework.

The TARGET agent is GPT-2-XL; its forward pass is intercepted to collect
per-layer activations.  INTERLOCUTOR agents are other GPT-2 variants
(or differently-prompted versions of GPT-2) that play the "other side"
of every structured task.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    model_name: str                     # HuggingFace model id
    agent_id: str                       # unique name used as label in activations
    persona: Optional[str] = None       # optional system-prompt / persona prefix
    temperature: float = 0.8
    max_new_tokens: int = 64            # Reduced from 128 to stay within GPT-2's 1024 token limit
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Additional generation kwargs forwarded verbatim
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class LLMAgent:
    """Wraps a HuggingFace causal-LM for turn-based dialogue."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        ).to(config.device)
        self.model.eval()

        self._gen_config = GenerationConfig(
            temperature=config.temperature,
            do_sample=config.temperature > 0,
            max_new_tokens=config.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            **config.generation_kwargs,
        )

    @property
    def agent_id(self) -> str:
        return self.config.agent_id

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def build_prompt(self, history: List[Dict[str, str]], task_description: str) -> str:
        """
        Convert a list of {"role": ..., "content": ...} turns into a flat
        text prompt GPT-2 can consume.
        """
        parts: List[str] = []
        if self.config.persona:
            parts.append(f"[Persona: {self.config.persona}]\n")
        parts.append(f"[Task: {task_description}]\n\n")
        for turn in history:
            role = turn["role"].upper()
            parts.append(f"{role}: {turn['content']}\n")
        parts.append("ASSISTANT: ")   # cue for this agent's response
        return "".join(parts)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        history: List[Dict[str, str]],
        task_description: str,
    ) -> str:
        """Run one generation step and return the new text."""
        prompt = self.build_prompt(history, task_description)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=850,   # leave room for new tokens + safety margin
        ).to(self.config.device)

        prompt_len = inputs["input_ids"].shape[1]
        output_ids = self.model.generate(
            **inputs,
            generation_config=self._gen_config,
        )
        new_ids = output_ids[0, prompt_len:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return response


# ---------------------------------------------------------------------------
# Target agent (GPT-2-XL) with activation hooks
# ---------------------------------------------------------------------------

class TargetAgent(LLMAgent):
    """
    The target agent whose *hidden-unit activations* are recorded
    while it reads the interlocutor's message (the *input turn*).

    We hook into the residual stream after every transformer block and
    also capture the intermediate MLP hidden activations.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._hooks: List[Any] = []
        # filled during a forward pass: layer_name -> (n_tokens, hidden_dim)
        self.last_activations: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _make_residual_hook(self, name: str):
        def hook(module, input, output):
            # output is the residual stream tensor: (batch, seq, hidden)
            # We store only the first batch item, detached to CPU
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            self.last_activations[name] = tensor[0].detach().cpu()
        return hook

    def _make_mlp_hook(self, name: str):
        """Capture the post-activation (after GELU) hidden state inside MLP."""
        def hook(module, input, output):
            self.last_activations[name] = output[0].detach().cpu() \
                if isinstance(output, tuple) else output.detach().cpu()
        return hook

    def register_hooks(self):
        """Register forward hooks on all transformer blocks."""
        self.remove_hooks()
        for i, block in enumerate(self.model.transformer.h):   # GPT-2 block list
            # Residual stream after full block
            h = block.register_forward_hook(
                self._make_residual_hook(f"block_{i:02d}_residual")
            )
            self._hooks.append(h)
            # MLP hidden (after act function) — GPT-2: block.mlp.act
            h2 = block.mlp.act.register_forward_hook(
                self._make_mlp_hook(f"block_{i:02d}_mlp_hidden")
            )
            self._hooks.append(h2)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Activation extraction on a *reading* forward pass
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def read_and_extract(
        self,
        text: str,
        interlocutor_id: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Run a forward pass over `text` (the interlocutor's message)
        WITHOUT generating anything.  Returns the captured activations.

        Args:
            text: The raw text the target agent is "reading."
            interlocutor_id: Label for this interlocutor condition.

        Returns:
            dict mapping layer_name -> tensor of shape (n_tokens, hidden_dim)
        """
        self.last_activations = {}
        self.register_hooks()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.config.device)

        self.model(**inputs)          # forward pass — hooks fire here
        self.remove_hooks()

        return {k: v.clone() for k, v in self.last_activations.items()}

    @torch.inference_mode()
    def generate_with_activations(
        self,
        history: List[Dict[str, str]],
        task_description: str,
        interlocutor_id: str,
    ) -> tuple[str, Dict[str, torch.Tensor]]:
        """
        Generate a response AND record activations on the input context.
        Returns (response_text, activations_dict).
        """
        prompt = self.build_prompt(history, task_description)
        # First: extract activations on the prompt (reading the context)
        acts = self.read_and_extract(prompt, interlocutor_id)
        # Then: generate the response
        response = self.generate(history, task_description)
        return response, acts


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_interlocutor_configs(
    device: str = "cpu",
    model_name: str = "gpt2",
) -> List[AgentConfig]:
    """
    Return a list of AgentConfig objects representing distinct 'other LLMs'.

    Same base model for all interlocutors — only the persona varies.
    This isolates the effect of *identity/style* rather than model architecture,
    giving a cleaner experimental design.
    """
    return [
        AgentConfig(
            model_name=model_name,
            agent_id="persona_curious",
            persona=(
                "You are a relentlessly curious assistant who responds primarily "
                "with follow-up questions and expresses wonder at every idea."
            ),
            temperature=0.9,
            device=device,
        ),
        AgentConfig(
            model_name=model_name,
            agent_id="persona_formal",
            persona=(
                "You are a formal, academic assistant. You respond in structured, "
                "numbered points and always cite hypothetical references."
            ),
            temperature=0.6,
            device=device,
        ),
        AgentConfig(
            model_name=model_name,
            agent_id="persona_creative",
            persona=(
                "You are a poetic, creative assistant who answers every question "
                "through vivid metaphors, analogies, and storytelling."
            ),
            temperature=1.0,
            device=device,
        ),
        AgentConfig(
            model_name=model_name,
            agent_id="persona_concise",
            persona=(
                "You are a terse, minimalist assistant. "
                "Never use more than two sentences. Be blunt."
            ),
            temperature=0.5,
            device=device,
        ),
        AgentConfig(
            model_name=model_name,
            agent_id="persona_skeptical",
            persona=(
                "You are a skeptical, devil's-advocate assistant who questions "
                "every claim and always presents the opposing viewpoint."
            ),
            temperature=0.8,
            device=device,
        ),
    ]


def make_target_config(
    device: str = "cpu",
    model_name: str = "gpt2-xl",
) -> AgentConfig:
    return AgentConfig(
        model_name=model_name,
        agent_id="target",
        persona=None,
        temperature=0.8,
        device=device,
    )
