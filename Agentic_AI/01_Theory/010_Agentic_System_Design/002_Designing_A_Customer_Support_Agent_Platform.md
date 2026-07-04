# Designing a Customer Support Agent Platform

## Framing the Problem

Where the coding agent chapter was about depth — one hard task, executed carefully over many steps —
a customer support agent platform is about breadth and concurrency: thousands of simultaneous
conversations, each one needing to feel instant, accurate, and safe, running on a budget that has to
survive being multiplied by call volume. This is the system-design question that tests whether you
can think like a platform engineer as much as an AI engineer: the LLM call is almost the easy part;
the hard part is everything around it — session management, retrieval freshness, escalation logic,
and cost control — at production scale.

The scenario to hold in your head throughout: a company (say, a mid-size e-commerce or SaaS
business) wants to deploy an agent that handles chat-based support across their website and app. It
needs to answer policy and product questions grounded in the company's actual documentation, look up
order or account details through internal APIs, handle multi-turn conversations where context from
three messages ago still matters, and hand off to a human agent smoothly when it's stuck, when the
customer is upset, or when the action requested is too sensitive to automate (refunds above a
threshold, account deletion, anything touching payment details).

## Requirements

Stating requirements up front focuses the rest of the design:

- **Latency**: users expect chat-like responsiveness. A reasonable target is first-token latency under 1.5-2 seconds and, for turns requiring tool calls (an order lookup, say), a full response within 4-6 seconds. Anything slower and users start double-messaging or abandoning the chat.
- **Concurrency**: production deployments for a mid-size business can mean thousands of concurrent conversations at peak (think a flash sale or an outage-driven support spike), each idle most of the time but bursty when active.
- **Grounding and accuracy**: answers about policy, pricing, or product behavior must be grounded in current, authoritative documentation — not the model's parametric knowledge, which may be stale or simply wrong for this specific company. Hallucinated policy answers are a direct source of business and legal risk (a support bot inventing a refund policy is a liability, not a minor bug).
- **Escalation to humans**: the system must recognize when it is out of its depth — low retrieval confidence, repeated failure to resolve, explicit user request for a human, detected frustration — and hand off smoothly, without losing conversation context.
- **Multi-turn state**: conversations span many turns, sometimes across sessions (a user leaves and comes back an hour later), and the agent needs to remember relevant facts (order number already provided, issue already described) without re-asking and without blowing the context budget by replaying the entire history every turn.
- **Safety and compliance**: PII handling (redaction/masking in logs), auditability (every action the agent took, especially ones with side effects, must be traceable), and content safety (the agent must not be tricked into leaking internal data, other customers' information, or performing unauthorized actions).

## High-Level Architecture

```
 Client (web/app chat widget)
        |
        v
 +--------------+       +------------------+
 |  API Gateway  |------>|  Session Manager |----> Session Store (Redis/DB)
 |  (auth, rate  |       |  (conv. state,   |
 |   limiting)   |       |   user context)  |
 +--------------+       +--------+---------+
                                  |
                                  v
                        +------------------+
                        |   Orchestrator    |
                        |  (turn handling,  |
                        |   routing logic)  |
                        +--------+---------+
                 +----------------+-----------------+
                 |                |                  |
                 v                v                  v
        +----------------+ +--------------+  +------------------+
        | Retrieval (RAG)| | Tool Layer   |  | Guardrails /      |
        | - doc index    | | (CRM, order  |  | Safety Classifier |
        | - freshness    | |  lookup API, |  | (input + output)  |
        |   pipeline     | |  refund API) |  +------------------+
        +----------------+ +--------------+
                 |                |
                 +--------+-------+
                          v
                +-------------------+        +--------------------+
                |   LLM (response    |------->| Escalation Service |
                |   generation)      |        | (human handoff,    |
                +-------------------+        |  context transfer)  |
                                              +--------------------+
                                                        |
                                                        v
                                              Human Agent Console
                                                        |
                                              Observability / Analytics
                                              (logs, metrics, feedback loop)
```

A single user turn flows like this: the gateway authenticates the request and applies rate limits;
the session manager loads (or creates) conversation state; the orchestrator decides what the turn
needs — pure Q&A, an account-specific lookup, or an action — and fans out to retrieval and/or tools
as needed; a guardrail layer screens both the incoming message (for injection attempts, abuse) and
the outgoing response (for policy violations, leaked PII, or fabricated claims) before it reaches
the user; and every step is logged for observability and later analysis. Let's go through the pieces
that are specific to support agents rather than generic to any LLM app.

## Session and State Management

Support conversations are stateful in a way that a one-shot Q&A tool isn't. A user might say "what's
the status of my order" in turn one, get an answer, then say "actually can I change the delivery
address" in turn four — the agent needs to still know which order is being discussed without the
user repeating the order number.

The practical design keeps two tiers of state. **Short-term conversational state** — the last N
turns verbatim — lives in a fast store (Redis is the typical choice) keyed by session ID, with a TTL
so abandoned sessions don't accumulate forever. **Extracted structured state** — the things worth
remembering beyond raw text, like "order #48213, issue: late delivery, customer_id: X" — is pulled
out explicitly (either by a lightweight extraction call or by having tools return structured results
that get pinned into a "facts" section of the working context) and kept separately from the raw
transcript, because structured facts survive summarization and truncation far better than trying to
re-derive them from old chat text every time.

As a conversation grows, replaying the full transcript on every turn becomes both slow and
expensive. The standard mitigation is a **sliding window plus running summary**: keep the last 6-10
turns verbatim for conversational coherence, and maintain a continuously updated one-paragraph
summary of everything older, regenerated incrementally rather than from scratch each time. This
keeps the per-turn token cost roughly constant regardless of how long the conversation has run,
which matters a lot at the concurrency levels this system needs to support.

```python
class ConversationState:
    def __init__(self, session_id, store, window_size=8):
        self.session_id = session_id
        self.store = store
        self.window_size = window_size

    def get_context(self) -> dict:
        raw = self.store.get(self.session_id) or {"turns": [], "summary": "", "facts": {}}
        recent_turns = raw["turns"][-self.window_size:]
        return {
            "summary": raw["summary"],       # compact history of older turns
            "facts": raw["facts"],           # structured, e.g. order_id, issue_type
            "recent_turns": recent_turns,    # verbatim recent exchange
        }

    def append_turn(self, user_msg, agent_msg, new_facts=None):
        raw = self.store.get(self.session_id) or {"turns": [], "summary": "", "facts": {}}
        raw["turns"].append({"user": user_msg, "agent": agent_msg})
        if new_facts:
            raw["facts"].update(new_facts)

        if len(raw["turns"]) > self.window_size * 2:
            to_summarize = raw["turns"][: -self.window_size]
            raw["summary"] = summarize_incremental(raw["summary"], to_summarize)
            raw["turns"] = raw["turns"][-self.window_size :]

        self.store.set(self.session_id, raw, ttl_seconds=3600 * 24)
```

## Grounding via RAG

Because policy and product content changes independently of the model, the platform needs a
retrieval layer that stays current without retraining anything. The knowledge base (help articles,
policy documents, product specs) is chunked and embedded the same way as in any RAG system, but
support platforms have a few domain-specific wrinkles worth calling out.

**Freshness matters more than in most RAG applications.** A pricing change or a policy update needs
to be reflected in the agent's answers within, realistically, minutes to hours — not the next batch
reindex cycle. This pushes toward an incremental indexing pipeline (index on document save/publish,
not on a nightly cron) and toward including document version/last-updated metadata in the retrieved
context, so the model can be instructed to prefer the most recent version if conflicting content is
retrieved.

**Retrieval confidence should gate the response, not just inform it.** If the top retrieved chunks
have low similarity scores, that is a direct, cheap signal that the knowledge base doesn't actually
cover this question — and the right behavior is not to let the model "do its best" from weak context
(that's exactly when hallucination happens), but to either ask a clarifying question or escalate.
This is one of the cleanest, cheapest escalation signals available, and it's worth wiring in before
more elaborate ones.

**Answers should be attributable.** Because a wrong policy statement is a liability, production
support agents typically constrain generation to be traceable to specific retrieved passages (citing
which help article an answer came from), which both improves trustworthiness for the user and gives
you an audit trail when someone disputes what the bot told them.

## The Tool Layer: Actions, Not Just Answers

A support agent that can only answer questions from documentation is a glorified FAQ search. The
value multiplies once it can take actions: look up an order, check a shipment status, initiate a
return, apply a small credit. Each of these is a tool call against an internal API, and the design
consideration specific to this domain is that **tools should be tiered by risk and reversibility**,
matching the human-in-the-loop patterns discussed generally in a later chapter but worth previewing
here: read-only lookups (order status, account details) can execute freely since they have no side
effects; low-risk, easily reversible actions (resending a confirmation email, applying a small,
policy-bounded discount) can execute autonomously with logging; and high-risk or hard-to-reverse
actions (refunds above a threshold, cancelling an order, changing account ownership) should require
either explicit user confirmation captured in the conversation ("just to confirm, you'd like me to
cancel order #48213 — is that right?") or routing to a human for approval, depending on the amount
at stake.

## Escalation to Humans

Escalation is the feature that determines whether users trust the system at all, and it deserves
explicit design rather than being an afterthought once the "happy path" is built. There are
essentially three triggers worth building, and production systems use them together.

**Confidence-based escalation** fires on internal signals: low RAG retrieval scores, the model's own
expressed uncertainty (some platforms have the model emit a confidence field alongside its answer,
though this is only moderately reliable on its own), or the tool layer returning an error the agent
doesn't know how to recover from. **Behavioral escalation** fires on conversation-shape signals: the
same issue being restated after two or three unsuccessful agent responses, or the user explicitly
asking for a human. **Sentiment-based escalation** fires on a lightweight classifier (can be a
small, fast model or even a simple heuristic) detecting frustration or anger, since an agent that
keeps cheerfully offering documentation links to an already-frustrated customer actively makes
things worse.

The mechanics of a good handoff matter as much as the trigger: the human agent's console should
receive the full conversation transcript, the extracted structured facts (order ID, issue type,
everything already established), and a short auto-generated summary of what's been tried — so the
customer never has to repeat themselves from scratch. Losing this context on handoff is one of the
most common and most damaging failures in real deployments, because it converts what should be a
seamless escalation into a visibly broken experience.

```python
def evaluate_escalation(turn_result, conversation_state) -> "EscalationDecision":
    if turn_result.retrieval_top_score < RETRIEVAL_CONFIDENCE_FLOOR:
        return EscalationDecision(escalate=True, reason="low_retrieval_confidence")

    if conversation_state.same_issue_turn_count() >= 3:
        return EscalationDecision(escalate=True, reason="unresolved_after_retries")

    if turn_result.user_requested_human:
        return EscalationDecision(escalate=True, reason="explicit_request")

    if sentiment_classifier.score(turn_result.user_message) > FRUSTRATION_THRESHOLD:
        return EscalationDecision(escalate=True, reason="detected_frustration")

    if turn_result.action_risk_tier == "high" and not turn_result.user_confirmed:
        return EscalationDecision(escalate=True, reason="high_risk_action_needs_review")

    return EscalationDecision(escalate=False)
```

## Trade-off: Cost vs. Accuracy at Scale

At the concurrency levels this platform targets, model choice is a cost lever with real magnitude,
not a rounding error. Running every single turn — including "hi, what are your store hours" — on the
largest frontier model, at thousands of concurrent sessions, gets expensive fast, while running
everything on the cheapest small model produces enough bad answers to erode trust in the whole
product. The standard resolution is **tiered model routing**: use a fast, cheap model (or even a
non-LLM classifier) to route the incoming message by intent and complexity, handle simple,
well-covered FAQ-style questions with a smaller model plus strong RAG grounding (most of the volume
in a typical support workload falls here), and reserve the larger model for multi-step reasoning,
ambiguous requests, or anything already flagged as higher-risk. This routing decision itself needs
to be fast and cheap, since it runs on every turn — usually a small classifier model or even
embedding-similarity against a set of known intent clusters, not another full LLM call with a long
prompt.

It's also worth noting that grounding quality substitutes for model size more than people initially
expect: a smaller model with excellent, tightly-relevant retrieved context frequently outperforms a
larger model with none, because the hard part of most support questions is *knowing the specific,
current answer*, not general reasoning ability. This means the RAG pipeline is often a better
investment of engineering time than chasing the newest, largest model.

## Trade-off: Latency vs. Thoroughness

Every additional step in the pipeline — a guardrail check, a retrieval call, a tool call — adds
latency, and support chat is one of the least forgiving domains for slow responses because users
compare it, consciously or not, to instant messaging with another human. Two techniques recur in
production designs. **Streaming** the response token-by-token as it's generated (rather than waiting
for the full answer) means the perceived latency is dominated by time-to-first-token rather than
total generation time, which is usually the single highest-leverage latency fix available.
**Parallelizing independent steps** — running the input guardrail check and the retrieval call
concurrently rather than sequentially, since neither depends on the other's output — trims real
wall-clock time off every turn. Where steps genuinely can't be parallelized (you can't screen an
output you haven't generated yet), the fallback is to make each one as fast as possible
individually: small, fast classifier models for guardrails rather than another full LLM call, and
aggressive caching of retrieval results for common questions (discussed further in the scalability
chapter).

## Trade-off: Safety at Scale

A guardrail that a human reviews for every conversation doesn't scale to thousands of concurrent
sessions; safety has to be largely automated, which means accepting a different risk profile than a
fully human-supervised system. The practical approach layers cheap, fast, automated checks on both
ends of every turn — input screening for prompt injection and abuse, output screening for policy
violations, PII leakage, and unsupported claims — backed by a much smaller volume of actual human
review, targeted at a random audit sample plus anything the automated layer flags as borderline
rather than clearly safe or clearly unsafe. This is the same confidence-tiering idea applied to
safety instead of just to escalation: certain-safe traffic flows straight through, certain-unsafe
traffic is blocked automatically, and the ambiguous middle is what gets the expensive human
attention, keeping the review burden proportional to actual risk rather than to total volume.

## Scale Numbers to Reason With

A useful set of figures to have ready: a mid-size deployment might see 5,000-20,000 concurrent
sessions at peak, with average conversations of 6-10 turns; at, say, 10,000 concurrent sessions with
a 2-second average response time and reasonable idle-time between messages, the actual concurrent
LLM-call load might be a few hundred requests in flight at once, which is the number that actually
sizes your inference capacity and rate limits with model providers — concurrent sessions and
concurrent inference calls are very different numbers, and conflating them leads to over- or
under-provisioning. On cost, assume an average of 2-4K input tokens per turn (system prompt,
retrieved context, recent conversation window) and a few hundred output tokens; across an 8-turn
conversation, that's roughly 20-30K tokens total, which at blended small/large model routing pricing
lands in the range of a few cents per conversation — multiplied across tens of thousands of
conversations a day, model routing decisions easily swing the monthly bill by an order of magnitude,
which is exactly why the tiered-routing trade-off above is worth defending with numbers in an
interview rather than asserting as a best practice.

