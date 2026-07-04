# Chain, Tree, and Graph of Thought

## Table of Contents

1. Why Reasoning Structure Matters
2. Chain-of-Thought: Linear Reasoning
3. The Failure Modes of a Single Chain
4. Tree of Thoughts: Branching and Backtracking
5. Building a Tree-of-Thought Search
6. Graph of Thoughts: Merging, Looping, and Reuse
7. Building a Graph-of-Thought Structure
8. Choosing the Right Structure for the Job
9. Cost, Latency, and the Economics of Deliberation
10. Production Notes and Interview Framing

---

## 1. Why Reasoning Structure Matters

When you ask a large language model to answer a hard question, the model is doing next-token prediction, not deliberate search. Left to itself, it commits to a line of reasoning the moment it starts generating and never looks back — there's no built-in mechanism for the model to notice three sentences in that its approach is wrong and try something else. This matters because many real problems, from multi-step arithmetic to itinerary planning to debugging a stack trace, are not solved correctly on the first attempt by an expert either. A senior engineer facing a hard bug doesn't stare at the code and produce the fix in one linear pass of thought; they form a hypothesis, test it, and if it's wrong, they discard that branch of reasoning and try another one. Structuring an LLM's reasoning process — deciding whether it reasons in a straight line, a branching tree, or an arbitrary graph — is really a decision about how much of that hypothesis-test-discard loop you are willing to externalize and control programmatically, versus leaving implicit inside a single forward pass.

The three techniques in this chapter — chain-of-thought, tree-of-thought, and graph-of-thought — form a progression in exactly this dimension. Chain-of-thought asks the model to externalize its reasoning as a sequence of steps within one generation, which helps because it gives the model more intermediate computation to work with and gives you, the engineer, a trace to inspect. Tree-of-thought goes further by generating multiple candidate next steps at each point, evaluating them, and pursuing only the promising ones, which requires wrapping the model in an explicit search procedure outside the single generation call. Graph-of-thought relaxes tree-of-thought's requirement that every idea have exactly one parent, allowing separate lines of reasoning to be merged, refined, and revisited, which is closer to how humans actually synthesize solutions from multiple partial insights. Each step up this ladder buys you more robustness against the model's tendency to lock into a wrong path early, at the cost of more inference calls, more orchestration code, and more places for things to go subtly wrong. Understanding when the extra cost is worth paying — and when it's pure overhead — is the practical skill this chapter is building toward.

## 2. Chain-of-Thought: Linear Reasoning

Chain-of-thought (CoT) prompting is the simplest and cheapest of the three techniques, and it remains the default reasoning mode for the overwhelming majority of production LLM calls. The core idea is almost embarrassingly simple: instead of asking the model to jump straight to a final answer, you ask it to "think step by step," producing a sequence of intermediate reasoning steps before the answer. The mechanism behind why this works is worth understanding precisely, because it explains both CoT's power and its limits. A transformer decoder produces one token at a time, and each token it emits becomes part of the context available for producing the next token. When you force the model to write out "first, we know X; therefore Y; combining X and Y gives Z" before writing the final answer, you are giving the model extra forward passes' worth of computation to work with and, critically, you are giving it a written record of intermediate facts it derived, which it can then condition on when producing later tokens. Without this, a model asked "what is 47 * 38, and is the result prime?" would have to do multiplication and primality checking implicitly in a fixed number of layers with no scratch space; with CoT, the multiplication result becomes text in the context, which the model can then reason over as if it were given data rather than having to have derived it silently.

This is why chain-of-thought helps disproportionately on tasks with multiple dependent sub-steps — arithmetic, multi-hop question answering, logic puzzles — and helps much less on tasks that are more like single lookups or matters of taste, where there isn't really a chain of intermediate facts to derive. It's also why the specific way you elicit CoT matters less than actually getting the model to produce intermediate steps at all: zero-shot prompts like "let's think step by step," few-shot exemplars showing worked reasoning, and models that have been fine-tuned or RL-trained to reason at length (the "reasoning models" that emerged after 2024) are all instances of the same underlying mechanism, just with different amounts of the behavior baked in versus elicited at inference time.

```python
class CoT_Reasoner:
    """Minimal chain-of-thought wrapper: one linear reasoning path, no branching."""

    def __init__(self, llm):
        self.llm = llm

    def reason(self, question: str, context: str | None = None) -> dict:
        prompt = f"""
        Question: {question}
        {'Context: ' + context if context else ''}

        Think through this step by step before answering:
        1. State the relevant facts or constraints.
        2. Work through the logic needed to connect them.
        3. Arrive at a final answer.

        End your response with a line starting exactly with "Final Answer:".
        """
        response = self.llm.generate(prompt, temperature=0.2)
        final_answer = self._extract_final_answer(response)
        return {"reasoning": response, "answer": final_answer}

    def _extract_final_answer(self, response: str) -> str:
        for line in response.splitlines():
            if line.strip().lower().startswith("final answer:"):
                return line.split(":", 1)[1].strip()
        return response.strip().splitlines()[-1]
```

A cheap but effective variant of plain CoT is self-consistency: instead of trusting a single reasoning chain, you sample several chains at a higher temperature and take a majority vote (or a similarity-weighted vote) over their final answers. This doesn't add structure to any individual chain — it's still linear reasoning — but it hedges against the fact that a single sample can go down a wrong path due to an unlucky token choice early on. Self-consistency is worth mentioning here because it's the natural stepping stone toward tree-of-thought: once you're already generating multiple independent chains and comparing them, the next logical move is to let those chains share partial progress and prune early rather than running each one to completion independently.

```python
from collections import Counter

class Self_Consistency_Reasoner:
    def __init__(self, llm, num_samples: int = 5):
        self.llm = llm
        self.num_samples = num_samples

    def reason(self, question: str) -> dict:
        answers = []
        for _ in range(self.num_samples):
            response = self.llm.generate(
                f"Think step by step and answer: {question}",
                temperature=0.7,  # diversity matters more than precision here
            )
            answers.append(self._extract_final_answer(response))

        counts = Counter(answers)
        best_answer, votes = counts.most_common(1)[0]
        return {"answer": best_answer, "confidence": votes / self.num_samples}

    def _extract_final_answer(self, response: str) -> str:
        return response.strip().splitlines()[-1]
```

## 3. The Failure Modes of a Single Chain

To understand why anyone would pay the extra cost of tree- or graph-structured reasoning, you need to be concrete about how a single chain fails. The most important failure mode is early commitment: once the model has written a few tokens down a particular line of reasoning, everything it generates afterward is conditioned on those tokens being true, and the model has no mechanism to "undo" them. If the second sentence of a CoT trace contains a subtly wrong assumption — a misremembered formula, a misread constraint, an arithmetic slip — the model will, in the overwhelming majority of cases, continue reasoning as if that sentence were correct, because from the model's point of view its own prior output is just more context to condition on, indistinguishable in status from the original prompt. This is different from how a careful human reasons; a human working through a hard proof will often reread their own working and catch an error a few lines back. A single autoregressive generation pass has no equivalent of "rereading and catching" built in — reflection, if it happens at all, has to be explicitly re-invoked (see the next chapter).

The second failure mode is that a single chain, by construction, explores exactly one hypothesis. For problems where the first plausible-sounding approach is not the correct one — combinatorial puzzles, planning problems with dead ends, creative tasks with multiple viable strategies — a linear chain will pursue that first approach to its conclusion even if a different approach would have been dramatically easier or more correct. Self-consistency partially addresses this by sampling multiple independent chains, but it does so blindly: it doesn't know which chains are promising until they're fully finished, so it pays the full cost of every chain regardless of how early a chain went off the rails. This is the specific inefficiency that tree-of-thought is designed to fix: evaluate partial progress and abandon bad branches early, rather than running every branch to completion before comparing.

The third failure mode, more subtle, is that a chain cannot express reasoning structures where two independent lines of thought need to be combined. If solving a problem genuinely requires deriving fact A via one line of reasoning and fact B via a different, unrelated line of reasoning, and then combining A and B, a single linear chain has to serialize this — reason about A, then reason about B, then combine — which works fine when A and B really are independent, but breaks down when exploring the A-branch and the B-branch each individually would benefit from their own branching search. This is the gap graph-of-thought fills, letting the reasoning process look like an actual dependency graph rather than a straight line or a tree rooted at one origin.

## 4. Tree of Thoughts: Branching and Backtracking

Tree-of-thought (ToT) reasoning treats problem solving as a search problem over a space of "thoughts," where a thought is a coherent intermediate step — a partial solution, a sub-conclusion, a proposed next move — rather than a single token. The tree has the original problem at the root; at each node, the model is prompted to generate several distinct candidate next thoughts (branching), each candidate is scored for how promising it looks, and the search continues by expanding only the highest-scoring branches while abandoning (pruning) the rest. Critically, this also gives you backtracking almost for free: if every child of a promising-looking node turns out to score poorly once expanded, the search can retreat to a sibling node that was scored lower initially but never got the chance to be explored, something a single linear chain structurally cannot do because it has already discarded the alternative continuations it didn't generate.

The three design choices that define a concrete ToT implementation are how thoughts are generated, how they're evaluated, and how the tree is searched. Thought generation is usually just an LLM call asking for N distinct next steps, with an explicit instruction to make them different from each other — otherwise the model tends to produce near-duplicates that don't actually diversify the search. Evaluation can be done by asking the model to self-score a thought's promise on some scale (cheap but noisy, since it's the same kind of model making the same kind of judgment it's not reliably good at — see the next chapter's discussion of self-critique), by using a separate, often smaller and cheaper, verifier model, or by a rule-based check when the domain allows it (e.g., "does this partial arithmetic expression parse and produce a plausible intermediate value"). Search strategy is typically breadth-first with pruning (keep the top-k thoughts at each level) or depth-first with backtracking (dive into the most promising branch and back out on failure); breadth-first with pruning is more robust when the evaluator is noisy since it hedges across several promising branches at once, while depth-first is cheaper when the evaluator is trustworthy and branches rarely need reviving.

The original ToT paper (Yao et al., 2023) demonstrated large gains on tasks like the "Game of 24" (combine four numbers with arithmetic operators to reach 24) and creative writing tasks with structural constraints, precisely because these tasks have exactly the properties described in the previous section: many plausible first moves, most of which turn out to be dead ends only a few steps later, and no way to know which without exploring several. On tasks that don't have this structure — a single lookup, a well-specified transformation with one correct approach — ToT adds branching that has nothing meaningful to branch over, and you pay several times the cost of CoT for no quality gain. Interviewers probing this topic are often listening for exactly this discrimination: not "can you describe ToT" but "can you tell me the property of a task that makes ToT worth its cost" (multiple plausible partial paths whose quality is only revealed by extending them, combined with a search space small enough that pruning is tractable within a reasonable token budget).

## 5. Building a Tree-of-Thought Search

The following sketch makes the search process concrete: it generates a fixed branching factor of thoughts at each node, scores each candidate thought independently of the model's own confidence about its full solution, prunes anything below a threshold, and recurses only into the branches worth exploring. Note that unlike a naive recursive implementation, a production version should track a global node budget (rather than only depth and branching factor) so that pathological cases — every branch scoring just above the pruning threshold — don't blow up the total cost.

```python
import json
from dataclasses import dataclass, field

@dataclass
class ThoughtNode:
    content: str
    depth: int
    score: float = 0.0
    children: list["ThoughtNode"] = field(default_factory=list)
    parent: "ThoughtNode | None" = None


class TreeOfThought:
    def __init__(self, llm, branching_factor=3, max_depth=3,
                 prune_below=0.4, node_budget=40):
        self.llm = llm
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.prune_below = prune_below
        self.node_budget = node_budget
        self.nodes_expanded = 0

    def solve(self, problem: str) -> list[str]:
        root = ThoughtNode(content=f"PROBLEM: {problem}", depth=0, score=1.0)
        self._expand(root, problem)
        best_leaf = self._best_leaf(root)
        return self._path_to_root(best_leaf)

    def _expand(self, node: ThoughtNode, problem: str):
        if node.depth >= self.max_depth or self.nodes_expanded >= self.node_budget:
            return

        candidates = self._generate_thoughts(problem, node, self.branching_factor)
        scored = [(c, self._evaluate_thought(problem, node, c)) for c in candidates]

        # Keep only promising branches; sort so the best gets explored first
        # under a shared node budget.
        promising = sorted(
            [(c, s) for c, s in scored if s >= self.prune_below],
            key=lambda pair: -pair[1],
        )

        for content, score in promising:
            if self.nodes_expanded >= self.node_budget:
                break
            child = ThoughtNode(content=content, depth=node.depth + 1,
                                 score=score, parent=node)
            node.children.append(child)
            self.nodes_expanded += 1
            self._expand(child, problem)

    def _generate_thoughts(self, problem, node, n) -> list[str]:
        history = self._path_to_root(node)
        response = self.llm.generate(f"""
        Problem: {problem}
        Reasoning so far: {' -> '.join(history)}

        Propose {n} distinct, non-overlapping next steps in the reasoning.
        Each must make a concrete claim or move, not a vague restatement.
        Return a JSON list of strings.
        """)
        return json.loads(response)

    def _evaluate_thought(self, problem, node, thought) -> float:
        history = self._path_to_root(node)
        response = self.llm.generate(f"""
        Problem: {problem}
        Path so far: {' -> '.join(history)}
        Candidate next step: {thought}

        Score 0.0-1.0: is this step logically valid given the path so far,
        and does it plausibly move closer to a full solution?
        Return only the number.
        """)
        try:
            return float(response.strip())
        except ValueError:
            return 0.0

    def _best_leaf(self, node: ThoughtNode) -> ThoughtNode:
        if not node.children:
            return node
        best_child = max(
            (self._best_leaf(c) for c in node.children),
            key=lambda leaf: leaf.score,
        )
        return best_child

    def _path_to_root(self, node: ThoughtNode) -> list[str]:
        path, current = [], node
        while current is not None:
            path.append(current.content)
            current = current.parent
        return list(reversed(path))
```

Two details in this sketch matter more than they might look. First, `_best_leaf` walks the whole tree at the end rather than assuming the deepest node reached along the "obvious" path is best — because pruning happens per-node based on a local score, the globally best leaf might be shallower than max depth if that branch reached a satisfying conclusion early. Second, the node budget is shared across the whole search rather than being a fixed per-level fan-out, which is what actually keeps the cost predictable in production; without it, a wide, shallow-looking config (branching_factor=5, max_depth=4) can spawn 5^4 = 625 evaluation calls, and interview conversations about ToT often stall right here because candidates describe the branching logic but not the budget control that makes it deployable.

## 6. Graph of Thoughts: Merging, Looping, and Reuse

Tree-of-thought's structural constraint — every thought has exactly one parent — is a simplification that holds only for problems that genuinely decompose into a single hierarchy of sub-decisions. Many real reasoning tasks don't have that shape. Consider drafting a technical design document: you might generate three candidate approaches to the data model in parallel, and separately three candidate approaches to the API surface, and the strongest final design might combine the best data-model idea with the best API idea — a synthesis that isn't a child of either parent alone, but a merge of two independent branches. A strict tree cannot represent this without either forcing an artificial ordering (design the API only after fully committing to a data model, losing the ability to explore both independently) or duplicating work (regenerating an API proposal from scratch inside every data-model branch). Graph-of-thought (GoT) generalizes the reasoning structure to an arbitrary directed graph: nodes are still thoughts, but edges can converge (multiple parents feeding one merged child) as well as diverge, and a thought can be refined in place and looped back on without necessarily creating a whole new subtree.

The operations that make graph-of-thought distinct from tree-of-thought are aggregation and refinement. Aggregation takes several existing thoughts — possibly from unrelated branches — and asks the model to synthesize them into one stronger thought, which is the mechanism that captures the "combine the best data model with the best API design" example above. Refinement takes a single thought and asks the model to improve it in place, optionally looping this multiple times, which captures the human pattern of iterating on a draft rather than always branching into fresh alternatives. Because both of these operations can point back at nodes anywhere in the graph, GoT needs an explicit graph data structure with cycle awareness (a refinement loop that never terminates needs a stopping condition), whereas a tree gets acyclicity for free from its structure.

The added expressiveness comes at real cost beyond token spend: a graph of thoughts is materially harder to reason about, debug, and score than a tree. In a tree, "the best path" has one clean definition — pick the leaf with the highest score and trace to the root. In a graph, a node might have contributions from several ancestors through different aggregation steps, so "the best path" is really "the best sub-DAG," which is a more expensive and less intuitively definable thing to search for. This is why GoT is reserved, in practice, for a narrower set of problems than ToT: research synthesis, technical writing that needs multiple independent threads combined, or optimization problems (the original GoT paper demonstrated it on sorting and set-operations tasks) where the ability to merge partial solutions materially reduces the total work versus having to reason about everything as one linear or tree-structured sequence. For the median production agent task, GoT's overhead in engineering complexity outweighs its benefit, and this is exactly the trade-off an interviewer wants you to be able to articulate rather than reciting the technique.

## 7. Building a Graph-of-Thought Structure

The sketch below shows the two operations that differentiate GoT from ToT — merging and refining — layered on a simple graph representation. It deliberately keeps scoring and traversal simple to keep the focus on the structural difference: a node can now have multiple parents (`Merge_Thoughts`), and a node can be improved without necessarily creating a new branch context (`Refine_Thought`, which does create a new node here but explicitly designates it as a linear successor of one node, not a branch).

```python
import json
from dataclasses import dataclass, field

@dataclass
class GoTNode:
    id: str
    content: str
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    score: float | None = None


class GraphOfThought:
    def __init__(self, llm):
        self.llm = llm
        self.nodes: dict[str, GoTNode] = {}

    def add_thought(self, node_id: str, content: str, parent_ids: list[str] | None = None):
        node = GoTNode(id=node_id, content=content, parents=parent_ids or [])
        self.nodes[node_id] = node
        for pid in node.parents:
            self.nodes[pid].children.append(node_id)
        return node

    def merge_thoughts(self, source_ids: list[str], new_id: str) -> GoTNode:
        """Aggregate several independent branches into one stronger thought."""
        sources = [self.nodes[sid].content for sid in source_ids]
        merged = self.llm.generate(f"""
        You have {len(sources)} independent partial solutions:
        {json.dumps(sources, indent=2)}

        Synthesize them into a single stronger solution that keeps the best
        elements of each and resolves any contradictions between them.
        """)
        return self.add_thought(new_id, merged, parent_ids=source_ids)

    def refine_thought(self, node_id: str, max_loops: int = 2) -> str:
        """Iteratively improve one thought in place, looping back on itself."""
        current_id = node_id
        for i in range(max_loops):
            current = self.nodes[current_id]
            improved = self.llm.generate(f"""
            Current solution: {current.content}

            Identify one concrete weakness and fix it. If there is nothing
            meaningful left to improve, repeat the solution unchanged and
            say so explicitly.
            """)
            if improved.strip() == current.content.strip():
                break
            new_id = f"{node_id}_refine{i}"
            self.add_thought(new_id, improved, parent_ids=[current_id])
            current_id = new_id
        return current_id

    def score_node(self, node_id: str) -> float:
        node = self.nodes[node_id]
        if node.score is not None:
            return node.score
        response = self.llm.generate(
            f"Score this solution 0.0-1.0 for correctness and completeness:\n{node.content}"
        )
        node.score = float(response.strip())
        return node.score

    def best_terminal(self) -> str:
        """Among nodes with no children (nothing built on top of them yet),
        return the id of the highest scoring one."""
        terminals = [nid for nid, n in self.nodes.items() if not n.children]
        return max(terminals, key=self.score_node)
```

A realistic pipeline using this structure would branch into two or three independent lines of reasoning with plain `add_thought` calls (functioning like ToT so far), refine each line once or twice with `refine_thought`, then call `merge_thoughts` on the best refined node from each line, and finally score the merged result against the original refined branches to check the merge actually produced something better rather than just different. That last check matters in practice — aggregation prompts can produce a bland compromise that's worse than the stronger of its two inputs, so a GoT pipeline that skips comparing the merge against its sources will silently regress quality some fraction of the time.

## 8. Choosing the Right Structure for the Job

The decision of which structure to use is best made by asking a small number of concrete questions about the task rather than defaulting to "more structure is more sophisticated, so use the fancier one." The first question is whether the task has more than one plausible correct approach that only reveals its (in)correctness a few steps in. If there's essentially one reasonable way to solve it — extracting a date from text, summarizing a document, classifying a support ticket — a single chain-of-thought pass is not just cheaper but also more reliable, because branching search is only valuable when there's real uncertainty about which branch to pursue; if there's nothing to branch over, ToT's evaluator is scoring near-duplicate candidates and adding pure noise plus cost.

The second question is whether early partial progress is a reliable predictor of final quality — that is, can a partial reasoning path a few steps deep be scored meaningfully, or does quality only become apparent once the full chain is complete? Tree-of-thought's entire value proposition rests on being able to prune bad branches early; if evaluating a half-finished thought is no more informative than a coin flip, pruning will discard some fraction of eventual winners along with the losers, and you'd be better off with self-consistency (run several full chains, no pruning, just vote at the end) or with plain CoT and a larger sample budget.

The third question is whether the problem genuinely requires combining independent lines of reasoning, versus just needing the single best line among several candidates. If the answer is "pick the best of N," a tree suffices — you never need to merge sibling branches, only select among leaves. If the answer is "the best solution draws from multiple angles that were developed separately," that's the graph-of-thought signature, and it justifies the added complexity of tracking multi-parent nodes and writing merge prompts.

Table form for a quick reference, though the paragraphs above are the part worth internalizing for an interview:

| Structure | Branches? | Backtrack? | Merge? | Relative Cost | Use When |
|---|---|---|---|---|---|
| Chain-of-Thought | No | No | No | 1x | Single plausible approach; sequential dependent facts |
| Self-Consistency | Parallel, independent | No | Implicit (vote) | Nx (N samples) | Want variance reduction, no need for early pruning |
| Tree-of-Thought | Yes | Yes | No | Depth × Branching, pruned | Multiple plausible approaches; early partial quality is informative |
| Graph-of-Thought | Yes | Yes | Yes | Highest, hard to bound | Best answer requires synthesizing independently-developed threads |

## 9. Cost, Latency, and the Economics of Deliberation

Every one of these techniques trades inference cost and latency for a chance at higher answer quality, and it's worth being explicit about the shape of that trade-off because it's usually the deciding factor in production, more than raw capability. A plain CoT call is one generation with a longer output — cost scales roughly linearly with the length of the reasoning trace, and latency is one round trip. Self-consistency multiplies both cost and (if run serially) latency by the sample count, though the calls are embarrassingly parallel, so with enough concurrent capacity the wall-clock latency hit can be much smaller than the cost hit. Tree-of-thought's cost is a function of branching factor, depth, and how aggressively you prune — in the worst case, unpruned, it's branching_factor^depth generation-plus-evaluation calls, but well-tuned pruning (a beam-search-style "keep top-k per level" policy) turns this into a controlled, roughly linear-in-depth cost with a much larger constant factor than plain CoT. Graph-of-thought's cost is the hardest to bound up front because the number of merge and refinement operations is often decided dynamically based on intermediate scores, which is exactly why production GoT systems need hard caps on total node count and total LLM calls rather than only depth limits.

The other dimension that's easy to overlook is latency variance rather than just mean cost. A ToT search with early pruning based on a noisy evaluator will sometimes prune a genuinely good branch by bad luck and have to fall back to a worse one, which is a correctness risk, not just a cost one — this is why production ToT implementations often keep more branches alive than the "obviously correct" minimum (e.g., top-3 instead of top-1 per level) purely as a hedge against evaluator noise, accepting extra cost as insurance against a low-probability but high-impact pruning mistake. When you're asked in an interview to reason about deploying ToT or GoT at scale, this is the kind of second-order concern — not "what's the average cost" but "what's the failure mode when the cheap evaluator you're using to prune is wrong" — that distinguishes someone who has actually built one of these systems from someone reciting the paper.

## 10. Production Notes and Interview Framing

A few practical patterns recur across real deployments of branching reasoning. First, almost nobody runs ToT or GoT unconditionally on every request; instead, a fast, cheap router (often a small classifier or a single LLM call) first estimates whether the incoming task looks like it has the "multiple plausible approaches, ambiguous which is best" signature described in Section 8, and only escalates to tree- or graph-structured search for that subset. This keeps average latency and cost low while still buying quality on the hard tail of requests, and it's the same escalation pattern you'll see recommended for self-critique and search-based planning in the next two chapters — cheap default, expensive fallback triggered by a signal of difficulty.

Second, the evaluator used to score thoughts is frequently a smaller or cheaper model than the one generating them, precisely because scoring "is this partial reasoning step plausible" is a much easier task than generating a full high-quality continuation, so it doesn't need the same capability tier — this is both a cost optimization and, done carefully, a way to get an evaluation signal that's somewhat independent of the generator's own blind spots (though not fully independent, since a related model trained similarly can share failure modes; see the next chapter's discussion of why self-evaluation is unreliable even across model sizes).

Third, almost every real implementation caps total nodes or total tool/LLM calls rather than only capping depth and branching factor, because depth × branching_factor bounds look controlled on paper but interact badly with retries, parallel branch evaluation, and refinement loops in ways that are easy to miss until a bill arrives. When discussing this in an interview, naming the node/call budget as a first-class design parameter — not an afterthought — signals production experience more than describing the search algorithm itself, which most candidates can do from having read the papers.

Finally, it's worth being able to state plainly, without hedging, that these structures are reasoning-quality techniques, not reasoning-correctness guarantees. A tree-of-thought search with a bad evaluator can confidently prune the correct branch and confidently return a wrong answer; a graph-of-thought merge can synthesize two flawed partial solutions into a more articulate but still flawed final one. None of chain, tree, or graph structuring substitutes for actual verification against ground truth — a unit test passing, a calculation checked by a calculator tool, a citation checked against a retrieved source. They improve the odds that the reasoning process explores the right part of the solution space; they do not, by themselves, confirm that it landed there. That distinction — between search quality and verified correctness — is exactly the bridge into the next chapter on self-critique and external verification.
