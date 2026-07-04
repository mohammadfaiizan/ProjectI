# Search-Based Planning and MCTS for Agents

## Table of Contents

1. From Reasoning About Text to Planning Actions
2. Greedy Action Selection and Its Limits
3. Framing Agent Planning as Search
4. Exploration vs. Exploitation in an Agent Context
5. Monte Carlo Tree Search: The Core Idea
6. MCTS for LLM Agents: A Worked Sketch
7. Cheaper Approximations to Full MCTS
8. When Search-Based Planning Is Worth the Cost
9. Failure Modes of Search-Based Planning
10. Production Guidance and Interview Framing

---

## 1. From Reasoning About Text to Planning Actions

The previous two chapters were about improving the quality of a single answer — a piece of text the model produces and, optionally, critiques and revises. This chapter shifts to a different but related problem: an agent that takes a sequence of actions in an environment — calling tools, querying APIs, clicking through a UI, moving a robot, editing files in a codebase — where each action changes the state of the world and constrains what actions make sense next. The core question is the same one that motivated tree-of-thought (should you commit to the first plausible option or explore several before committing), but it's now asked about real actions with real side effects rather than about candidate sentences, which changes the cost-benefit calculation in an important way: undoing a bad reasoning step costs nothing but tokens, while undoing a bad real-world action — a sent email, a modified file, a placed order — can be expensive, slow, or impossible.

Search-based planning is the general term for treating "what should the agent do next" as a search problem over a space of possible action sequences, rather than committing greedily to whatever the model's single best guess for the next action happens to be. Monte Carlo Tree Search (MCTS) is the specific, well-known search algorithm — famous from its role in AlphaGo and AlphaZero — for doing this efficiently when the space of possible action sequences is too large to search exhaustively. This chapter builds up from why greedy action selection is often insufficient, through the general concept of search-based planning, into MCTS specifically, and closes on the practical question that matters most in a production system: given that search-based planning is expensive, when does it actually pay for itself compared to just executing the agent's first idea?

## 2. Greedy Action Selection and Its Limits

Most deployed LLM agents today use what is, underneath the tool-calling scaffolding, a greedy policy: at each step, the model is given the current state (conversation history, tool results so far, the goal) and asked to produce the single next action, that action is executed, the result is appended to context, and the loop repeats. This is the ReAct-style loop that underlies the majority of production agent frameworks, and it works well for a large class of tasks for a specific reason: many agent tasks have a "locally obvious" best next step at each point — if the goal is to answer a question that requires looking something up, the obviously correct next action is to look it up, and there's no meaningful alternative branch worth considering. In these tasks, the model's first idea for the next action usually is the right one, and any search machinery wrapped around it would just add latency without changing the outcome.

Greedy selection starts to break down on tasks where the immediate best-looking action is not the one that leads to the best overall outcome — the sequential-decision-making equivalent of a greedy algorithm getting stuck in a local optimum. A concrete example: an agent tasked with debugging a failing test might greedily pick "add a print statement and rerun" as the locally reasonable next step, when a different first action — reading the git blame history for the failing function — would have revealed the actual regression in one step and saved several rounds of trial and error. The greedy agent isn't wrong to consider "add a print statement" a reasonable action; the problem is it never compared that action against the alternative before committing, because greedy selection by construction only ever generates and executes one candidate action per step. A second failure pattern specific to agents (distinct from a single-shot reasoning task) is that actions in an environment can have side effects that are hard or impossible to undo, so a wrong greedy choice doesn't just waste a reasoning step, it can leave the environment in a state that makes recovery harder — a partially applied file edit, a duplicate record created via an API call, a shopping cart with the wrong item added and payment already initiated on a subsequent step.

## 3. Framing Agent Planning as Search

To move beyond greedy selection, you need to formalize the agent's decision process as something a search algorithm can operate over: a state, a set of actions available from that state, a transition function describing what state results from taking an action, and some notion of value or reward that tells you how good a given state or completed trajectory is. This is precisely the framing used in reinforcement learning and classical AI planning (it's the structure of a Markov Decision Process), and casting an LLM agent's task this way is what makes algorithms like MCTS applicable at all — MCTS doesn't know anything about LLMs or tool calls, it only knows how to search over states, actions, and rewards, so the engineering work is in defining those pieces for your specific agent task.

The state, for an LLM agent, is typically the full context so far: the goal, the conversation/tool-call history, and any external environment state that's been observed (file contents, API responses, webpage content). The action space is whatever the agent's tool set allows — a discrete set of possible tool calls, or in the case of free-form actions, a set of candidate next steps the model itself proposes (much like a ToT thought-generation step, but each "thought" here is an executable action rather than a piece of reasoning text). The transition function is, practically, "execute the action against the real environment or a simulator, and observe the resulting state" — for real-world side-effecting actions this is where a genuinely important design decision comes in, because you generally cannot afford to actually execute several candidate actions against production systems just to see which one works out, which is the topic of Section 7. The reward or value signal is the hardest part to get right and is usually either a terminal signal (did the overall task succeed, at the end of a full trajectory) or an estimated intermediate value (an LLM- or heuristic-based guess at how promising a partial trajectory looks, with the same reliability caveats discussed for ToT's thought-evaluation step in the previous chapter).

Once you have these pieces, "planning" becomes "search for a high-value sequence of actions before committing to execute the first one," and the question of which search algorithm to use is a question of how large the action space and horizon are, how expensive it is to evaluate a candidate trajectory, and how much you can simulate versus how much requires real execution.

## 4. Exploration vs. Exploitation in an Agent Context

The exploration/exploitation trade-off is usually introduced in the context of multi-armed bandits or reinforcement learning, but it applies directly, and concretely, to agent action selection. At any decision point, the agent (or the search procedure acting on its behalf) has some current estimate of which action looks best — exploiting means taking that action and refining the estimate around it, while exploring means trying an action that currently looks worse but hasn't been evaluated enough to be sure, on the chance that it's actually better and the current estimate is just noisy or based on too little information. Pure exploitation risks getting stuck on a plausible-looking but suboptimal first idea, exactly the greedy-selection failure mode from Section 2. Pure exploration wastes budget trying options that are obviously weak, gathering information that doesn't change the ultimate decision.

What makes this trade-off concrete rather than abstract in an agent-planning setting is that "trying an action" here often means actually calling a tool, which has real cost (API fees, latency) and sometimes real risk (a side effect you can't cleanly undo). This means the exploration budget in agent planning is typically much smaller and much more carefully rationed than in domains like game-playing, where a simulator lets you explore millions of hypothetical trajectories for free. It also means an important design distinction: exploring by *simulating* an action's likely outcome (asking the LLM "if I did X, what would probably happen and how good would that be") is cheap and safe but only as accurate as the model's ability to predict environment behavior, while exploring by *actually executing* an action is accurate but costly and sometimes irreversible. Most practical search-based agent planning uses simulated exploration for the search/comparison phase and reserves real execution for the action the search ultimately selects, treating the simulation as an imperfect but much cheaper proxy for the real transition function. UCB1 (Upper Confidence Bound) and its variants, the formula MCTS uses to balance exploration and exploitation (introduced properly in the next section), exist specifically to make this trade-off systematic rather than a matter of hand-tuned heuristics: it gives each candidate action a score that rewards both current estimated value and how under-explored the action still is, so the algorithm naturally shifts from exploring broadly early on to exploiting the best-looking option as evidence accumulates.

## 5. Monte Carlo Tree Search: The Core Idea

MCTS is a best-first search algorithm designed for exactly the situation described above: an action space too large to explore exhaustively, a transition function you can simulate (even if imperfectly) rather than compute exactly, and a reward signal that may only be reliably available at the end of a full trajectory (a game's win/loss outcome, an agent task's ultimate success/failure) rather than cleanly at every intermediate step. It builds a search tree incrementally, one simulated trajectory ("rollout") at a time, and it does this through four repeated phases, which is the part worth being able to explain precisely rather than just naming, since "I've heard of MCTS" and "I can explain the four phases and why each exists" are very different depths of understanding in an interview.

**Selection** walks down the existing tree from the root, at each node picking the child that best balances estimated value against how little it's been explored — this is where the UCB1-style formula from Section 4 is applied, typically `value_estimate + C * sqrt(log(parent_visits) / child_visits)`, where the first term favors exploitation of known-good children and the second term grows for infrequently-visited children, favoring exploration, with `C` a tunable constant controlling how strongly exploration is weighted. Selection continues until it reaches a node that has unexplored children — a state from which not every possible action has been tried yet.

**Expansion** adds one or more of those unexplored children to the tree — concretely, for an LLM agent, this usually means generating one or a few candidate next actions from the current state via an LLM call, since the "full" action space at a given state is often not enumerable in advance the way it is in a board game.

**Simulation (rollout)** estimates the value of the newly expanded node by playing forward from it — in classical MCTS (as in AlphaGo's predecessor systems) this meant playing random or heuristic-guided moves to the end of the game and recording the outcome; for an LLM agent it more often means asking the model to simulate what would plausibly happen if this action sequence were continued to completion, or executing a small number of further real or simulated steps and using a value-estimation LLM call to score the resulting state, since full rollouts to a real task's completion are usually too expensive to do many times over.

**Backpropagation** takes the value obtained from the simulation and propagates it back up the path that was selected in phase one, updating the visit count and average value estimate of every node along that path — this is what makes the tree's value estimates improve over successive iterations rather than resetting each time, and it's the mechanism by which early, noisy value estimates get progressively refined as more rollouts pass through a given node.

These four phases repeat for a fixed budget (a number of iterations, or a time limit), and at the end, the action taken from the root is typically the child with the most visits (not necessarily the highest raw estimated value — visit count is a more robust choice because a node visited many times has a value estimate that's had many chances to be corrected, whereas a node visited only once might have an artificially high estimate purely from a lucky rollout).

## 6. MCTS for LLM Agents: A Worked Sketch

The following sketch shows the four MCTS phases wired to an LLM agent's action space, with an LLM used both to expand (propose candidate next actions) and to simulate (estimate the value of a partial trajectory without fully executing it against the real environment). This is a conceptual, single-threaded implementation meant to make the algorithm's structure legible, not a production-ready planner.

```python
import math
import random

class MCTSNode:
    def __init__(self, state, parent=None, action_taken=None):
        self.state = state                # accumulated context / trajectory so far
        self.parent = parent
        self.action_taken = action_taken  # the action that produced this state from parent
        self.children: list["MCTSNode"] = []
        self.untried_actions: list[str] | None = None  # lazily populated
        self.visits = 0
        self.total_value = 0.0

    @property
    def value_estimate(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0

    def is_fully_expanded(self) -> bool:
        return self.untried_actions is not None and len(self.untried_actions) == 0


class LLM_MCTS_Planner:
    def __init__(self, llm, executor, goal, iterations=30, exploration_c=1.4,
                 rollout_depth=2):
        self.llm = llm
        self.executor = executor      # actually executes an action against the env
        self.goal = goal
        self.iterations = iterations
        self.c = exploration_c
        self.rollout_depth = rollout_depth

    def plan_next_action(self, current_state: str) -> str:
        root = MCTSNode(state=current_state)

        for _ in range(self.iterations):
            node = self._select(root)
            if not node.is_fully_expanded():
                node = self._expand(node)
            value = self._simulate(node)
            self._backpropagate(node, value)

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action_taken

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.untried_actions is not None and node.is_fully_expanded() and node.children:
            node = max(node.children, key=self._ucb1)
        return node

    def _ucb1(self, node: MCTSNode) -> float:
        if node.visits == 0:
            return float("inf")  # force at least one visit before comparing on value
        exploit = node.value_estimate
        explore = self.c * math.sqrt(math.log(node.parent.visits) / node.visits)
        return exploit + explore

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if node.untried_actions is None:
            node.untried_actions = self._propose_actions(node.state)

        action = node.untried_actions.pop()
        next_state = self._simulate_transition(node.state, action)
        child = MCTSNode(state=next_state, parent=node, action_taken=action)
        node.children.append(child)
        return child

    def _propose_actions(self, state: str, n: int = 3) -> list[str]:
        response = self.llm.generate(f"""
        Goal: {self.goal}
        Current state / history: {state}

        Propose {n} distinct, concrete next actions (tool calls) worth
        considering. Do not evaluate them, just propose diverse options.
        Return a JSON list of short action descriptions.
        """)
        import json
        return json.loads(response)

    def _simulate_transition(self, state: str, action: str) -> str:
        # Cheap, imperfect model-predicted transition, NOT real execution --
        # real execution is reserved for the action the search finally picks.
        predicted = self.llm.generate(f"""
        Current state: {state}
        Action taken: {action}

        Predict, briefly and concretely, the most likely resulting state
        or tool output after this action.
        """)
        return f"{state}\n[Action: {action}] -> [Predicted result: {predicted}]"

    def _simulate(self, node: MCTSNode) -> float:
        state = node.state
        for _ in range(self.rollout_depth):
            action = self._propose_actions(state, n=1)[0]
            state = self._simulate_transition(state, action)

        score = self.llm.generate(f"""
        Goal: {self.goal}
        Simulated trajectory: {state}

        Score 0.0-1.0 how close this trajectory gets to achieving the goal.
        Return only the number.
        """)
        try:
            return float(score.strip())
        except ValueError:
            return 0.0

    def _backpropagate(self, node: MCTSNode, value: float):
        current = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent
```

Two things about this sketch deserve emphasis because they're exactly where interview conversations about "have you actually implemented this" tend to focus. First, notice that `_simulate_transition` never calls `self.executor` — the search phase operates entirely on the model's *predicted* outcomes, and only after `plan_next_action` returns its final choice would the calling code invoke `self.executor` for real. This separation is what makes MCTS affordable for real agents: you pay for many rounds of cheap, simulated exploration and only one round of expensive, possibly irreversible real execution. Second, the `float("inf")` for unvisited nodes in `_ucb1` is not a stylistic flourish — it guarantees every child gets tried at least once before the algorithm starts trusting value estimates to compare them, which matters because a value estimate based on zero rollouts is not a value estimate at all, it's an absence of information, and treating it as "presumed bad" rather than "presumed worth checking" would bias the search away from ever discovering that an initially-unpromising-looking action is actually the best one.

## 7. Cheaper Approximations to Full MCTS

Full MCTS with real environment execution at every rollout is rarely affordable for LLM agents outside of research settings, because each simulated action typically costs an LLM call (or several), and the iteration counts that make MCTS effective in board games (thousands to millions of rollouts) are completely impractical at LLM-call prices and latencies. Production and near-production systems that use search-flavored planning almost always use a cut-down version, and it's worth knowing the common simplifications because they're what you'd actually reach for.

The most common simplification is a shallow beam search instead of a full tree search: generate the top-k candidate next actions, simulate one step ahead for each (rather than a full rollout to the goal), score the resulting one-step-ahead states, keep the best one or few, and repeat. This captures the core benefit of not committing greedily to the very first idea, and of comparing a small number of alternatives before acting, without paying for MCTS's iterative tree refinement, which only pays off when you can afford enough iterations for the visit-count statistics to actually stabilize.

A second common simplification replaces rollout-based value estimation with a learned or heuristic value function that scores a state directly without needing to simulate forward at all — for instance, a smaller fine-tuned model trained specifically to predict "how likely is this partial trajectory to lead to task success," which is much cheaper per call than asking a large general-purpose LLM to reason its way to a score, and can be run over many candidate branches in parallel cheaply. This mirrors the "use a smaller/cheaper evaluator model" pattern from tree-of-thought and self-critique — the theme across all of these techniques is that the model generating candidates and the model (or function) scoring them don't need to be the same size or even the same architecture, and decoupling them is usually where the real cost savings live.

A third simplification, useful specifically when actions have real side effects, is executing tentatively against a sandbox or dry-run mode rather than either "just simulate via LLM prediction" or "execute for real" — for example, running a proposed code change against a test suite in an isolated container, or calling an API's dry-run/preview endpoint if one exists, before committing to the real side-effecting call. This trades the LLM's imperfect prediction of an action's outcome for the environment's own ground truth about that outcome, at the cost of needing a safely reversible or isolated execution path to exist in the first place, which is not always available (there's no dry-run mode for "send this email").

## 8. When Search-Based Planning Is Worth the Cost

The decision of whether to wrap an agent's action selection in any of this machinery — full MCTS, beam search, or even a much simpler "generate three candidate next actions and pick the best with one extra LLM call" — should be driven by a concrete assessment of the task, not a general belief that more sophistication is better. Search-based planning earns its cost when at least two conditions hold together: the action space at each step has multiple genuinely plausible options whose downstream consequences differ meaningfully (if there's really only one sensible next action, there's nothing to search over), and mistakes are expensive to recover from, either because of real-world side effects, wasted downstream work, or a long horizon before the mistake's consequences become visible and correctable. A single-turn tool call with a cheap, obviously-correct next step — fetching a value from a well-known API, formatting a response — has neither property, and wrapping it in search is pure overhead. A multi-step task like "refactor this module in a way that keeps all tests passing," where an early structural decision (which abstraction to introduce) constrains everything downstream and is expensive to unwind once a dozen files have been edited around it, has both properties strongly, and is exactly the kind of task where spending extra inference upfront comparing two or three structural approaches before committing pays for itself many times over in avoided rework.

A useful diagnostic question, mirroring the one from the previous chapter on tree-of-thought, is: if you let the greedy agent run and it turns out to have picked badly, how far into the trajectory do you find out, and how much of the prior work is salvageable? If bad choices reveal themselves immediately and cheaply (a tool call fails fast with a clear error, easily retried with a different approach), greedy execution with reactive error handling is usually cheaper and just as effective as upfront search — you're paying the "exploration cost" only on the branches that actually turn out to be needed, rather than upfront on branches that might not matter. If bad choices only reveal themselves many steps later, after significant compounding work has been built on top of the mistake, upfront search that compares options before committing becomes worth its cost precisely because it avoids that compounding.

| Signal | Favors Greedy Execution | Favors Search-Based Planning |
|---|---|---|
| Number of plausible next actions | Usually one clear best option | Several meaningfully different options |
| Cost of a wrong choice | Cheap, fails fast, easily retried | Expensive, hard to undo, or discovered late |
| Task horizon | Short (few steps to completion) | Long (many dependent steps downstream) |
| Reversibility of actions | Reversible / idempotent | Side-effecting, irreversible, or costly to redo |
| Latency tolerance | Low (interactive, user waiting) | Higher (batch, background, or high-stakes enough to justify delay) |

## 9. Failure Modes of Search-Based Planning

Search-based planning inherits every reliability caveat from tree-of-thought's evaluation step, because the value/scoring function at the heart of MCTS's simulation phase is usually an LLM call, subject to the same self-evaluation unreliability discussed in the self-critique chapter — a search algorithm is only as good as the signal it's searching over, and "search harder" cannot compensate for "the value estimates are systematically biased." If the LLM used for rollout scoring has a blind spot that causes it to consistently overrate a certain style of action (for instance, overrating actions that produce verbose, confident-sounding intermediate output regardless of whether that output is actually correct), MCTS will search very efficiently toward a confidently wrong conclusion, which is arguably worse than a greedy agent's mistake because the search process lends the outcome an unearned appearance of having been carefully vetted.

A second failure mode specific to using LLM-predicted transitions (Section 6's `_simulate_transition`) rather than real execution during search is that the model's prediction of what a tool call will return can be systematically optimistic or simply wrong in ways that don't match the real environment's actual behavior — models are generally better at predicting "typical" or "expected" tool outputs than genuinely unusual ones, so search conducted purely over simulated transitions can systematically underweight the branches that would actually encounter edge cases, exactly the trajectories where careful planning would have been most valuable. This is a strong argument for hybridizing real execution into the search wherever it's cheap and safe enough to do so (dry runs, sandboxes, read-only exploratory calls) rather than relying purely on LLM-predicted rollouts.

A third failure mode is budget mismanagement: because each MCTS iteration costs real LLM calls, teams that adopt it without instrumenting cost per decision can find that a planner meant to improve a handful of high-stakes decisions is quietly running its full iteration budget on every single agent step, including the many steps where Section 8's diagnostic questions would have said greedy execution was fine. This is why, in practice, search-based planning is almost always gated behind an explicit difficulty or stakes classifier, the same escalation pattern recommended for tree-of-thought and self-critique — cheap greedy execution as the default, with search invoked selectively rather than universally.

## 10. Production Guidance and Interview Framing

If asked to design a search-based planning component for an agent, the strongest answers walk through the same sequence covered in this chapter rather than jumping straight to "I'd implement MCTS": first, establish that the task actually has the two properties from Section 8 that make search worth its cost — multiple genuinely different plausible actions, and expensive-to-undo mistakes; second, define the state/action/transition/reward structure concretely for the specific agent, since this mapping is where most of the real engineering work and most of the design decisions live, not in the search algorithm itself, which is largely off-the-shelf; third, decide explicitly whether transitions during search will be simulated (cheap, imperfect, LLM-predicted) or partially real (dry runs, sandboxes, safe read-only probes), because this choice determines both cost and how trustworthy the search's conclusions actually are; fourth, decide on the granularity of search — full MCTS with iterative refinement, or a cheaper shallow beam search — based on how many iterations you can actually afford at the latency and cost budget the product requires; and fifth, put a difficulty/stakes gate in front of the whole thing so that the default path for the many simple, low-stakes decisions remains cheap greedy execution, with search reserved for the subset of decisions where it earns its keep. Being able to articulate that MCTS itself is the least novel part of this design — the real work is state/action/reward modeling, transition fidelity, and cost-aware gating — is what separates an answer that demonstrates production experience from one that demonstrates having read the AlphaGo paper.
