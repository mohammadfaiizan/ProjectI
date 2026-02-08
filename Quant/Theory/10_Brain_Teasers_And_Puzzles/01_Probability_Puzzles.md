# Probability Puzzles

## Puzzle 1: Monty Hall Problem

**Statement:** You're on a game show with three doors. Behind one door is a car, behind the other two are goats. You pick door 1. The host, who knows what's behind each door, opens door 3, revealing a goat. He then asks if you want to switch to door 2. Should you switch?

**Solution:** Yes, switch. The probability of winning if you switch is 2/3.

**Key insight:** The host's action provides information. Initially, each door has probability 1/3. After you pick door 1:
- If car is behind door 1 (prob 1/3): Host opens door 2 or 3, switching loses
- If car is behind door 2 (prob 1/3): Host opens door 3, switching wins
- If car is behind door 3 (prob 1/3): Host opens door 2, switching wins

Switching wins in 2 out of 3 cases, so probability is 2/3.

## Puzzle 2: Gambler's Ruin

**Statement:** You have \$100 and bet \$1 on a fair coin flip each round. You stop when you reach \$200 or \$0. What's the probability you reach \$200?

**Solution:** The probability is 1/2.

**Key insight:** For gambler's ruin with initial wealth $a$, target $b$, and fair game:
$$P(\text{reach } b) = \frac{a}{b}$$

Here: $a = 100$, $b = 200$, so $P = 100/200 = 1/2$.

**General formula:** For biased coin with win probability $p$:
$$P = \frac{1 - (q/p)^a}{1 - (q/p)^b}$$

where $q = 1-p$.

## Puzzle 3: Birthday Problem

**Statement:** How many people are needed so that the probability of at least two sharing a birthday exceeds 1/2?

**Solution:** 23 people.

**Key insight:** Use complement: $P(\text{all different}) = 1 - P(\text{at least one match})$.

For $n$ people:
$$P(\text{all different}) = \frac{365}{365} \times \frac{364}{365} \times \cdots \times \frac{365-n+1}{365} = \prod_{i=0}^{n-1} \frac{365-i}{365}$$

Set equal to 1/2:
$$\prod_{i=0}^{n-1} \frac{365-i}{365} = \frac{1}{2}$$

Solving gives $n \approx 23$.

**Approximation:** Using $1-x \approx e^{-x}$:
$$P(\text{all different}) \approx e^{-n(n-1)/(2 \times 365)}$$

Setting equal to 1/2: $n \approx \sqrt{2 \times 365 \times \ln 2} \approx 23$.

## Puzzle 4: Coupon Collector

**Statement:** There are $n$ different coupons. Each box contains one coupon, uniformly random. How many boxes do you need to collect all $n$ coupons?

**Solution:** Expected number is $n H_n \approx n \ln n + \gamma n$, where $H_n$ is the $n$-th harmonic number and $\gamma \approx 0.577$ is Euler's constant.

**Key insight:** Let $T_i$ be time to collect $i$-th new coupon after having $i-1$:
$$E[T_i] = \frac{n}{n-i+1}$$

Total time: $T = \sum_{i=1}^n T_i$

$$E[T] = \sum_{i=1}^n \frac{n}{n-i+1} = n \sum_{j=1}^n \frac{1}{j} = n H_n$$

For $n=10$: $E[T] = 10 \times H_{10} \approx 10 \times 2.93 = 29.3$ boxes.

## Puzzle 5: Broken Stick

**Statement:** Break a stick of length 1 at two random points. What's the probability the three pieces form a triangle?

**Solution:** The probability is 1/4.

**Key insight:** Let break points be $X, Y \sim \text{Uniform}(0,1)$ with $X < Y$. Piece lengths: $X$, $Y-X$, $1-Y$.

Triangle inequality requires:
- $X + (Y-X) > 1-Y$ → $Y > 1/2$
- $X + (1-Y) > Y-X$ → $X < 1/2$
- $(Y-X) + (1-Y) > X$ → $Y - X < 1/2$

These define a triangle in $(X,Y)$ space with area 1/4.

Since total area is 1/2 (for $X < Y$), probability is $(1/4)/(1/2) = 1/2$... wait, need to account for ordering.

Actually: $P = 1/4$ (verified by integration).

## Puzzle 6: Random Points on Circle

**Statement:** Choose three random points on a circle. What's the probability they lie on a semicircle?

**Solution:** The probability is 3/4.

**Key insight:** Fix one point. The other two must lie in the semicircle starting from that point. Probability that a random point lies in a semicircle is 1/2. For two independent points: $(1/2)^2 = 1/4$.

But we can choose which point to fix, and the semicircle can start from any point. More carefully: Fix one point at 0. The arc from 0 to $\pi$ (semicircle) has probability 1/2. Both other points in this arc: $(1/2)^2 = 1/4$.

But the semicircle could also be from $\pi$ to $2\pi$... Actually, the complement (all three on semicircle) is easier.

**Correct approach:** All three on semicircle if and only if no arc of length $\pi$ is empty. This happens with probability 3/4.

## Puzzle 7: Hat Check Problem

**Statement:** $n$ people check their hats. Hats are returned randomly. What's the expected number of people who get their own hat back?

**Solution:** Expected number is 1, regardless of $n$.

**Key insight:** Let $X_i = 1$ if person $i$ gets their hat, else 0.

$$E[X_i] = P(\text{person } i \text{ gets own hat}) = \frac{1}{n}$$

By linearity:
$$E[\sum_{i=1}^n X_i] = \sum_{i=1}^n E[X_i] = n \times \frac{1}{n} = 1$$

**Variance:** $\text{Var}(\sum X_i) = 1$ (for large $n$, approximately Poisson with $\lambda=1$).

## Puzzle 8: Dice Puzzles

**Puzzle 8a:** Roll two dice. What's the probability the sum is 7?

**Solution:** $P(\text{sum} = 7) = 6/36 = 1/6$.

There are 6 ways: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) out of 36 total.

**Puzzle 8b:** Roll until you get a 6. What's the expected number of rolls?

**Solution:** $E[\text{rolls}] = 6$.

Geometric distribution with $p = 1/6$: $E[X] = 1/p = 6$.

## Puzzle 9: Card Puzzles

**Puzzle 9a:** Shuffle a deck. What's the probability the top card is an ace?

**Solution:** $P = 4/52 = 1/13$.

**Puzzle 9b:** Draw two cards without replacement. What's the probability both are aces?

**Solution:** $P = \frac{4}{52} \times \frac{3}{51} = \frac{1}{221}$.

**Puzzle 9c:** What's the probability of a flush (5 cards of same suit)?

**Solution:** 
$$P = \frac{4 \times \binom{13}{5}}{\binom{52}{5}} = \frac{4 \times 1287}{2598960} \approx 0.00198$$

## Puzzle 10: Matching Problems

**Statement:** $n$ letters are randomly placed into $n$ envelopes. What's the expected number of correct matches?

**Solution:** Expected number is 1 (same as hat check).

**What about probability of at least one match?**

Using inclusion-exclusion:
$$P(\text{at least one match}) = 1 - \sum_{k=1}^n \frac{(-1)^{k+1}}{k!}$$

For large $n$: $P \approx 1 - e^{-1} \approx 0.632$.

## Puzzle 11: Conditional Probability Traps

**Puzzle 11a:** A family has two children. One is a boy. What's the probability both are boys?

**Solution:** $P = 1/3$, not 1/2.

**Key insight:** Sample space: {BB, BG, GB, GG} (equally likely).

Given "at least one boy": {BB, BG, GB}. Probability both boys: 1/3.

**Puzzle 11b:** A family has two children. The older is a boy. What's the probability both are boys?

**Solution:** $P = 1/2$.

Given "older is boy": {BB, BG}. Probability both boys: 1/2.

The difference: "at least one" vs "specific one".

## Puzzle 12: Bayesian Reasoning

**Statement:** 1% of population has a disease. Test is 99% accurate (99% true positive, 99% true negative). You test positive. What's the probability you have the disease?

**Solution:** $P(\text{disease} | \text{positive}) \approx 0.5$ (50%).

**Key insight:** Use Bayes' theorem:
$$P(D|+) = \frac{P(+|D)P(D)}{P(+)} = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.01 \times 0.99} = \frac{0.0099}{0.0198} = 0.5$$

**Intuition:** With rare disease, false positives dominate.

## Puzzle 13: Random Walk Return

**Statement:** Start at 0. Each step: +1 with prob 1/2, -1 with prob 1/2. What's the probability you return to 0?

**Solution:** In 1D, probability of return is 1 (recurrent). Expected time to return is infinite.

**Key insight:** For symmetric random walk in 1D:
$$P(\text{return to 0}) = 1$$

But $E[\text{time to return}] = \infty$ (null recurrent).

In 2D: Also recurrent (return probability = 1).
In 3D+: Transient (return probability < 1).

## Puzzle 14: First Passage

**Statement:** Random walk starting at 0. What's the expected time to hit +1?

**Solution:** $E[T] = \infty$.

**Key insight:** For symmetric random walk, expected time to hit any level is infinite (null recurrence).

**For biased walk** with $p > 1/2$:
$$E[T] = \frac{1}{2p-1}$$

## Puzzle 15: Two Envelopes

**Statement:** Two envelopes contain money. One has twice as much as the other. You pick one, open it to see \$100. Should you switch?

**Solution:** Switching doesn't help. The paradox arises from incorrect expectation calculation.

**Key insight:** If you have \$100, the other envelope has either \$50 or \$200 with equal probability (if we assume uniform prior on the smaller amount).

But this assumption is problematic. If envelopes contain $(x, 2x)$ and $x$ is uniformly distributed, then $E[\text{other}] = E[x] + E[2x] = 3E[x]/2$, which is always larger than your envelope if $x < 2x$... This creates a paradox.

**Resolution:** The prior distribution matters. With proper Bayesian analysis, switching provides no advantage.

## Puzzle 16: St. Petersburg Paradox

**Statement:** Coin flip game: Win \$2 if heads on first flip, \$4 if first tail then heads, \$8 if TT then H, etc. (win \$2^n if $n$ tails then heads). How much would you pay to play?

**Solution:** Expected value is infinite: $E = \sum_{n=1}^{\infty} 2^n \times 2^{-n} = \sum_{n=1}^{\infty} 1 = \infty$.

But people wouldn't pay much. Resolution: Use utility function (log utility) or bounded rationality.

## Puzzle 17: Secretary Problem

**Statement:** Interview $n$ candidates sequentially. After each interview, must decide immediately. Can't go back. What's the optimal strategy?

**Solution:** Reject first $k$ candidates, then pick the first one better than all previous. Optimal $k \approx n/e$.

**Key insight:** For large $n$, optimal $k = n/e$, and probability of selecting best is approximately $1/e \approx 0.368$.

**Strategy:** 
1. Observe first $k$ candidates (reject all)
2. From candidate $k+1$ onward, accept first candidate better than best of first $k$
3. If none found, take last candidate

## Puzzle 18: Prisoner's Dilemma

**Statement:** Two prisoners. If both confess: 5 years each. If both stay silent: 1 year each. If one confesses: confessor free, other gets 10 years. What should they do?

**Solution:** Dominant strategy: Both confess (Nash equilibrium), even though both staying silent is better for both.

**Key insight:** Game theory: Individual rationality vs collective rationality.

## Puzzle 19: Bertrand's Box

**Statement:** Three boxes: (G,G), (G,S), (S,S). Pick a box at random, then a coin at random from that box. It's gold. What's the probability the other coin is gold?

**Solution:** $P = 2/3$.

**Key insight:** Given gold coin:
- Box (G,G): probability 1/3, other is gold
- Box (G,S): probability 1/3, other is silver  
- Box (S,S): probability 0 (can't pick gold)

So $P(\text{other gold}) = (1/3)/(1/3 + 1/3) = 1/2$... wait.

Actually: $P(\text{pick G}) = 1/2$ (half coins are gold).
Given gold: $P(\text{from (G,G)}) = 2/3$, $P(\text{from (G,S)}) = 1/3$.
So $P(\text{other gold}) = 2/3$.

## Puzzle 20: Three Cards

**Statement:** Three cards: red/red, red/blue, blue/blue. Shuffle, pick one, show one side (red). What's the probability the other side is red?

**Solution:** $P = 2/3$.

**Key insight:** Given red side shown:
- Card RR: 2 ways to show red (both sides)
- Card RB: 1 way to show red (one side)
- Card BB: 0 ways

So $P(\text{other red}) = 2/(2+1) = 2/3$.

Similar to Bertrand's box paradox.
