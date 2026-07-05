"""
004_Claude3_5_Sonnet.py

Model: Claude 3.5 Sonnet (Anthropic, June 2024; updated October 2024)
What this file demonstrates: a toy "computer use" action-grounding loop --
given a structured description of on-screen elements (name, type, bounding
box) and a natural-language task, a small PyTorch model grounds the task into
a specific UI element and emits a structured action (click / type / scroll)
with pixel coordinates, then the loop feeds back a simulated post-action
screen state, mirroring the screenshot -> action -> new-screenshot cycle
Anthropic's own computer-use documentation describes.

IMPORTANT: Anthropic has never disclosed how Claude 3.5 Sonnet actually
performs pixel-coordinate grounding, nor its architecture or training data
for this capability. The real system takes a raw screenshot IMAGE as input;
this toy demonstration instead takes a structured list of on-screen elements
(as if produced by some upstream perception step) purely so the grounding and
action-selection LOGIC can be implemented and inspected directly in PyTorch,
without also having to build and train a real vision encoder from scratch.
The point of this file is the control-flow and grounding mechanism -- text
description of an element -> embedding -> similarity match -> coordinate
output -- not a claim about Claude 3.5 Sonnet's real internals.
"""

from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# A simulated "screen": a list of UI elements with a type, a text label, and
# a bounding box (x0, y0, x1, y1) in pixel coordinates. This stands in for
# whatever a real vision encoder would extract from a raw screenshot; here it
# is given directly so the demo can focus on grounding + action selection.
# ---------------------------------------------------------------------------

@dataclass
class UIElement:
    label: str
    kind: str            # "button", "text_field", "checkbox", "link", "menu_item"
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1

    def center(self) -> Tuple[int, int]:
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) // 2, (y0 + y1) // 2


@dataclass
class ScreenState:
    elements: List[UIElement]
    description: str = ""  # e.g. "Login form" -- flavor text for the demo


class ActionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    DONE = "done"


@dataclass
class GroundedAction:
    action: ActionType
    target_label: Optional[str] = None
    coordinates: Optional[Tuple[int, int]] = None
    text: Optional[str] = None
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Tiny text embedding: a bag-of-character-trigram hashed embedding, learned
# via a small linear projection. Purely a lightweight, self-contained stand-in
# for "some text encoder" -- not a claim about real tokenization/embeddings.
# ---------------------------------------------------------------------------

class HashedTextEncoder(nn.Module):
    def __init__(self, embed_dim: int = 32, num_buckets: int = 512):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def _trigram_ids(self, text: str) -> List[int]:
        # zlib.crc32 (not Python's built-in hash()) so results are stable
        # across process runs -- str hash() is salted per-process by default
        # and would make this "deterministic" demo silently non-reproducible.
        text = f"  {text.lower()}  "
        return [zlib.crc32(text[i:i + 3].encode("utf-8")) % self.num_buckets
                for i in range(len(text) - 2)]

    def forward(self, text: str) -> torch.Tensor:
        ids = torch.tensor(self._trigram_ids(text), dtype=torch.long)
        if ids.numel() == 0:
            ids = torch.zeros(1, dtype=torch.long)
        bag = self.embedding(ids).mean(dim=0)
        return F.normalize(self.proj(bag), dim=-1)


# ---------------------------------------------------------------------------
# The grounding + action-selection model: embeds the task instruction and
# every candidate UI element's label, ranks elements by similarity to the
# instruction, and picks an action type based on the target element's kind.
# This is the toy analogue of "ground the instruction into a specific pixel
# coordinate and issue a structured action."
# ---------------------------------------------------------------------------

class ComputerUseAgent(nn.Module):
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.encoder = HashedTextEncoder(embed_dim=embed_dim)

    def ground(self, instruction: str, screen: ScreenState) -> Tuple[UIElement, float]:
        """Embed the instruction and every element's label, return the
        highest-similarity element and its similarity score -- the toy
        analogue of mapping "click the blue Submit button" onto a specific
        pixel location by grounding text against perceived screen content."""
        instr_emb = self.encoder(instruction)
        best_elem, best_score = None, -1.0
        for elem in screen.elements:
            elem_emb = self.encoder(f"{elem.kind} {elem.label}")
            score = F.cosine_similarity(instr_emb.unsqueeze(0), elem_emb.unsqueeze(0)).item()
            if score > best_score:
                best_elem, best_score = elem, score
        return best_elem, best_score

    def decide_action(self, instruction: str, screen: ScreenState,
                       grounding_threshold: float = 0.15) -> GroundedAction:
        """Full step of the computer-use loop: ground the instruction onto an
        element, then choose the action type appropriate for that element's
        kind, and emit a structured action with pixel coordinates -- exactly
        the kind of structured output a real executor would consume to move
        a mouse and click, per Anthropic's documented computer-use loop."""
        elem, score = self.ground(instruction, screen)

        if elem is None or score < grounding_threshold:
            # Low-confidence grounding: a real system would ideally re-observe
            # the screen or ask for clarification rather than act blindly --
            # this mirrors the real misclick-risk concern (Section 6 of the
            # .md file) that low-confidence grounding shouldn't be executed.
            return GroundedAction(action=ActionType.DONE, confidence=score)

        coords = elem.center()
        if elem.kind == "text_field":
            typed_text = _extract_typed_text(instruction)
            return GroundedAction(ActionType.TYPE, target_label=elem.label,
                                   coordinates=coords, text=typed_text, confidence=score)
        if elem.kind in ("button", "link", "menu_item", "checkbox"):
            return GroundedAction(ActionType.CLICK, target_label=elem.label,
                                   coordinates=coords, confidence=score)
        return GroundedAction(ActionType.DONE, confidence=score)


def _extract_typed_text(instruction: str) -> str:
    """Very crude extraction of a quoted string to "type" -- e.g. instruction
    'type "alice@example.com" into the email field' -> 'alice@example.com'.
    A real system would rely on the model's own language understanding
    rather than a regex-like heuristic; this is a stand-in only."""
    if '"' in instruction:
        parts = instruction.split('"')
        if len(parts) >= 2:
            return parts[1]
    return ""


# ---------------------------------------------------------------------------
# A tiny scripted environment: applies a GroundedAction to a ScreenState and
# returns the NEXT ScreenState, standing in for "execute the action, then
# capture a new screenshot" -- the other half of Anthropic's documented
# screenshot -> action -> new-screenshot loop.
# ---------------------------------------------------------------------------

def apply_action(screen: ScreenState, action: GroundedAction) -> ScreenState:
    if action.action == ActionType.CLICK and action.target_label == "Submit":
        return ScreenState(elements=[
            UIElement("Success message", "text_field", (100, 40, 400, 70)),
        ], description="Form submitted")
    if action.action == ActionType.TYPE:
        updated = []
        for elem in screen.elements:
            if elem.label == action.target_label:
                updated.append(UIElement(f"{elem.label} (filled: {action.text})", elem.kind, elem.bbox))
            else:
                updated.append(elem)
        return ScreenState(elements=updated, description=screen.description)
    return screen  # no-op / DONE: screen unchanged


def run_computer_use_loop(agent: ComputerUseAgent, screen: ScreenState,
                           instructions: List[str], max_steps: int = 10) -> None:
    print(f"\nInitial screen: {screen.description}")
    for elem in screen.elements:
        print(f"  [{elem.kind}] '{elem.label}' bbox={elem.bbox} center={elem.center()}")

    for step, instruction in enumerate(instructions[:max_steps], start=1):
        print(f"\nStep {step}: instruction = {instruction!r}")
        action = agent.decide_action(instruction, screen)
        if action.action == ActionType.DONE:
            print(f"  -> DONE / no confident grounding (confidence={action.confidence:.3f})")
            continue
        print(f"  -> grounded to element '{action.target_label}' (confidence={action.confidence:.3f})")
        if action.action == ActionType.CLICK:
            print(f"  -> ACTION: click at {action.coordinates}")
        elif action.action == ActionType.TYPE:
            print(f"  -> ACTION: click at {action.coordinates}, then type {action.text!r}")
        screen = apply_action(screen, action)
        print(f"  -> new screen state: {screen.description or '[unchanged]'}")


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    print("=" * 78)
    print("Claude 3.5 Sonnet (2024): toy computer-use action-grounding loop")
    print("Real screenshot perception and coordinate grounding are undisclosed;")
    print("this demo takes a structured element list instead of a raw image and")
    print("focuses purely on the grounding -> structured-action control flow.")
    print("=" * 78)

    login_screen = ScreenState(
        elements=[
            UIElement("Email", "text_field", (80, 100, 380, 130)),
            UIElement("Password", "text_field", (80, 150, 380, 180)),
            UIElement("Remember me", "checkbox", (80, 200, 110, 230)),
            UIElement("Submit", "button", (80, 260, 200, 300)),
            UIElement("Forgot password?", "link", (220, 260, 380, 290)),
        ],
        description="Login form",
    )

    agent = ComputerUseAgent(embed_dim=32)

    task_instructions = [
        'Type "alice@example.com" into the email field',
        'Type "hunter2" into the password field',
        "Click the submit button",
    ]

    run_computer_use_loop(agent, login_screen, task_instructions)

    print(
        "\nReal computer-use systems ground against a raw screenshot IMAGE (pixel"
        "\ngrid), not a structured element list, and must additionally decide when"
        "\nan action failed or had no effect -- the verification problem discussed"
        "\nin the .md file's Section 8. This demo only shows the grounding and"
        "\nstructured-action-emission control flow, at toy scale, with a perfect"
        "\nscripted environment rather than a real, noisy desktop."
    )
