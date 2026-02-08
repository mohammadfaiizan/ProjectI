# Expected Value and Strategy

## Problem 1: When to Stop Rolling Dice

**Statement:** Roll a die repeatedly. After each roll, you can stop and take the current value, or continue. What's the optimal strategy and expected value?

**Solution:** Optimal strategy: Stop if roll $\geq 4$. Expected value: $4.25$.

**Key insight:** Use backward induction.

**Value function:** $V(n)$ = expected value starting from state $n$.

**Optimality:** Stop if current roll $\geq$ continuation value.

**Computation:**
- $V(6) = 6$ (always stop)
- $V(5) = 5$ (stop, since $E[\text{continue}] = 3.5 < 5$)
- $V(4) = 4$ (stop)
- $V(3) = \max(3, E[\text{continue}]) = \max(3, 4.25) = 4.25$ (continue)
- $V(2) = 4.25$ (continue)
- $V(1) = 4.25$ (continue)

**Starting value:** $E[V(\text{first roll})] = \frac{1}{6}(4 + 4 + 4.25 + 4.25 + 4.25 + 4.25) = 4.25$.

**Strategy:** Stop if roll $\geq 4$, otherwise continue.

## Problem 2: Card Game - When to Stop

**Statement:** Draw cards from deck. After each card, decide: stop and take face value (A=1, J/Q/K=10), or continue. Cards not replaced. What's optimal strategy?

**Solution:** Dynamic programming with state = remaining cards.

**Value function:** $V(\text{remaining deck})$.

**Optimality:** Stop if current card value $\geq$ expected continuation value.

**Computation:** Complex due to state space, but solvable via DP.

**Approximation:** For large deck, threshold around 7-8 (similar to dice problem).

## Problem 3: Envelope Paradox

**Statement:** Two envelopes contain money. One has twice the other. You pick one, see \$X. Should you switch?

**Solution:** No advantage to switching (paradox resolved by proper prior).

**Incorrect reasoning:** Other envelope has $X/2$ or $2X$ with prob 1/2 each, so $E[\text{switch}] = 1.25X > X$.

**Problem:** Assumes uniform prior on $X$, which is impossible (can't be uniform over all positive reals).

**Correct analysis:** With proper prior (e.g., $X$ has distribution $f$), switching provides no advantage in expectation.

## Problem 4: St. Petersburg Paradox

**Statement:** Coin flip game: Win \$2 if H, \$4 if TH, \$8 if TTH, etc. (win \$2^n if $n$ tails then heads). Expected value is infinite. How much to pay?

**Solution:** Use utility function. With log utility: $E[U] = \sum_{n=1}^{\infty} \frac{\ln(2^n)}{2^n} = \ln 2 \sum_{n=1}^{\infty} \frac{n}{2^n} = 2\ln 2$.

So would pay up to $e^{2\ln 2} = 4$ dollars.

**Key insight:** Diminishing marginal utility resolves paradox.

## Problem 5: Secretary Problem

**Statement:** Interview $n$ candidates sequentially. After each, decide immediately. Can't go back. Maximize probability of selecting best candidate.

**Solution:** Reject first $k \approx n/e$ candidates, then select first better than all previous. Success probability $\approx 1/e$.

**Optimal $k$:** Maximizes $P(\text{success}) = \frac{k}{n} \sum_{i=k+1}^n \frac{1}{i-1}$.

For large $n$: $k^* \approx n/e$, $P^* \approx 1/e \approx 0.368$.

**Strategy:**
1. Observe and reject first $\lfloor n/e \rfloor$ candidates
2. From next candidate onward, accept first one better than best seen so far
3. If none found, take last candidate

## Problem 6: Optimal Stopping - House Selling

**Statement:** Receive offers sequentially. Each offer: accept (stop) or reject (can't go back). Maximize expected offer.

**Solution:** If offers uniform on $[0,1]$: Accept first offer $\geq 1/e \approx 0.368$.

**Key insight:** Similar to secretary problem. Threshold = $1/e$ for uniform distribution.

**General:** For distribution $F$, threshold $t$ satisfies: $t = E[\max(t, X)]$ where $X \sim F$.

## Problem 7: Auction Theory - First-Price Sealed Bid

**Statement:** $n$ bidders, each has private value $v_i \sim \text{Uniform}[0,1]$. Highest bidder wins, pays their bid. What's optimal bid?

**Solution:** For symmetric equilibrium: $b^*(v) = \frac{n-1}{n}v$.

**Key insight:** Bid below value to account for winner's curse.

**Expected payment:** $E[\text{payment} | \text{win}] = \frac{n-1}{n+1}v$ (when $v$ is highest).

## Problem 8: Auction Theory - Second-Price Sealed Bid

**Statement:** Same setup, but winner pays second-highest bid.

**Solution:** Optimal strategy: Bid your true value $b^*(v) = v$.

**Key insight:** Second-price auction is strategy-proof (truthful bidding is dominant strategy).

**Expected payment:** $E[\text{payment}] = E[\text{second highest} | \text{you win}] = \frac{n-1}{n+1}v$.

## Problem 9: Game Theory - Prisoner's Dilemma

**Statement:** Two players. If both cooperate: (3,3). If both defect: (1,1). If one defects: (5,0) for defector, (0,5) for cooperator.

**Solution:** Dominant strategy: Both defect. Nash equilibrium: (defect, defect) with payoff (1,1).

**Key insight:** Individual rationality leads to suboptimal outcome. (Cooperate, cooperate) is Pareto optimal but not Nash.

**Repeated game:** Can sustain cooperation with trigger strategies if discount factor high enough.

## Problem 10: Nash Equilibrium

**Statement:** Find Nash equilibria in various games.

**Definition:** Strategy profile where no player can improve by unilaterally deviating.

**Example - Matching Pennies:**
- Player 1: H or T
- Player 2: H or T
- Payoffs: (1,-1) if match, (-1,1) if don't match
- Nash: Mixed strategy: both play H with prob 1/2.

**Mixed strategy equilibrium:** Each player indifferent between pure strategies.

## Problem 11: Kelly Criterion

**Statement:** Repeated bets with edge $p$ (win prob) and odds $b$ (win $b$ per unit bet if win). What fraction of wealth to bet?

**Solution:** Optimal fraction: $f^* = \frac{pb - (1-p)}{b} = p - \frac{1-p}{b}$.

**For even-money bet ($b=1$):** $f^* = 2p - 1$ (if $p > 1/2$).

**Key insight:** Maximizes long-term growth rate: $G = p\ln(1+bf) + (1-p)\ln(1-f)$.

**Example:** $p = 0.6$, $b = 1$: $f^* = 2 \times 0.6 - 1 = 0.2$ (bet 20% of wealth).

**Properties:**
- Never risk ruin ($f^* < 1$)
- Maximizes $E[\ln(\text{wealth})]$
- Conservative (half-Kelly common in practice)

## Problem 12: Optimal Betting

**Statement:** You have edge: win prob $p > 1/2$, even-money bet. How much to bet?

**Solution:** Kelly criterion: $f^* = 2p - 1$.

**Example:** $p = 0.55$: $f^* = 0.1$ (bet 10%).

**Full-Kelly vs Half-Kelly:**
- Full-Kelly: Maximum growth, high volatility
- Half-Kelly: 75% of growth, much lower volatility
- Often prefer half-Kelly in practice

## Problem 13: Gambler's Ruin - Optimal Betting

**Statement:** Start with \$100, target \$200. Each bet: win \$1 with prob $p$, lose \$1 with prob $1-p$. What's optimal bet size?

**Solution:** For fair game ($p=1/2$): Any bet size gives same probability (1/2), but smaller bets reduce risk.

**For $p > 1/2$:** Larger bets increase win probability but also risk.

**Kelly-optimal:** Bet fraction $f = 2p - 1$ of current wealth (if allowed).

**Fixed bet:** If must bet fixed amount, larger bets increase win probability but also ruin risk.

## Problem 14: Two-Armed Bandit

**Statement:** Two slot machines with unknown win probabilities $p_1$, $p_2$. Maximize total winnings over $T$ plays.

**Solution:** Exploration vs exploitation trade-off.

**Upper Confidence Bound (UCB):** 
$$UCB_i = \hat{p}_i + c\sqrt{\frac{\ln t}{n_i}}$$

Play arm with highest UCB.

**Thompson Sampling:** Sample from posterior, play arm with highest sample.

**Optimal:** Gittins index (complex, requires solving Bellman equation).

## Problem 15: Multi-Armed Bandit

**Statement:** $k$ arms, unknown rewards. Maximize cumulative reward.

**Epsilon-greedy:** Explore with prob $\epsilon$, exploit best arm otherwise.

**UCB1:** $UCB_i = \bar{r}_i + \sqrt{\frac{2\ln t}{n_i}}$.

**Regret:** $O(\sqrt{kT \ln T})$ for UCB1.

## Problem 16: Optimal Stopping - Job Search

**Statement:** Receive job offers sequentially. Each offer: accept (stop) or continue searching. Offers i.i.d. from known distribution.

**Solution:** Threshold strategy: Accept first offer $\geq$ threshold $t^*$.

**Threshold:** $t^* = E[\max(t^*, X)]$ where $X$ is offer distribution.

**For uniform $[0,1]$:** $t^* = 1/e \approx 0.368$.

**Expected number of offers:** $1/P(X \geq t^*) = 1/(1-t^*) = e$.

## Problem 17: Search Problem - House Hunting

**Statement:** View houses sequentially. After each, decide: buy (stop) or continue. Maximize expected house value (or utility).

**Solution:** Similar to secretary problem. Threshold strategy optimal.

**With recall:** Can go back to previous houses → different strategy (more complex).

## Problem 18: Investment Timing

**Statement:** Stock price follows random walk. When to buy/sell to maximize expected profit?

**Solution:** Optimal stopping problem. Threshold strategies common.

**Buy-and-hold vs timing:** For random walk, timing doesn't help (martingale property).

**With drift:** Optimal to buy immediately if positive drift, never if negative drift.

## Problem 19: Resource Allocation

**Statement:** Allocate limited resource across opportunities with uncertain returns. Maximize expected total return.

**Solution:** Greedy algorithm if returns are independent and submodular.

**Dynamic programming:** If returns depend on allocation amounts.

**Example:** Allocate \$1000 across 3 investments with expected returns $r_1(x)$, $r_2(x)$, $r_3(x)$.

Solve: $\max \sum_i r_i(x_i)$ s.t. $\sum_i x_i \leq 1000$.

**Optimality:** $r_i'(x_i^*) = \lambda$ (equal marginal returns) if interior solution.

## Problem 20: Portfolio Choice Under Uncertainty

**Statement:** Choose portfolio to maximize expected utility with uncertain returns.

**Mean-variance:** $\max \boldsymbol{w}^T \boldsymbol{\mu} - \frac{\lambda}{2} \boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w}$.

**Kelly criterion:** For repeated investments, maximize $E[\ln(\text{wealth})]$.

**Optimal:** $\boldsymbol{w}^* = \frac{1}{\lambda} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}$ (if unconstrained).

**With constraints:** Use optimization methods (quadratic programming).

**Key insight:** Risk aversion parameter $\lambda$ determines risk-return trade-off.
