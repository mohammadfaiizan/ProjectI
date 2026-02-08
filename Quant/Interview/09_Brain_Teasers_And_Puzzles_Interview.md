# Brain Teasers And Puzzles Interview

## Q1: Two dice are rolled. What is the probability that the sum equals 7?

**A1:** There are 36 possible outcomes when rolling two dice (6 × 6). For the sum to equal 7, the favorable outcomes are: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1). That's 6 favorable outcomes. Therefore, P(sum = 7) = 6/36 = 1/6 ≈ 0.1667. This is the most likely sum for two dice because 7 can be achieved in the most ways. The probability distribution is symmetric around 7, with sums of 2 and 12 being least likely (each with probability 1/36) and sums closer to 7 being more likely.

---

## Q2: You draw cards from a standard deck. What is the probability that you get the first ace before the first king?

**A2:** Ignore all cards that are neither ace nor king—they don't affect the outcome. Among the 8 relevant cards (4 aces, 4 kings), each is equally likely to appear first. Since there are 4 aces and 4 kings, and we want an ace before a king, P(first ace before first king) = 4/(4+4) = 1/2 = 0.5. More formally, by symmetry, each of the 8 cards has equal probability 1/8 of being drawn first. The probability an ace is first is 4 × (1/8) = 1/2. This symmetry argument works because aces and kings are symmetric in the problem statement—there's no reason to favor one over the other.

---

## Q3: What is the expected number of coin flips to get two heads in a row?

**A3:** Let E be the expected number of flips. Consider the first flip: if it's T (tails), we've used 1 flip and are back to the start, so expected additional flips = E. If it's H (heads), we've used 1 flip. Now consider the second flip: if it's H, we're done (used 2 flips total). If it's T, we've used 2 flips and are back to start, so expected additional flips = E. Therefore: E = (1/2)(1 + E) + (1/2)[(1/2)(2) + (1/2)(2 + E)] = (1/2)(1 + E) + (1/4)(2) + (1/4)(2 + E) = (1/2)(1 + E) + 1/2 + (1/4)(2 + E) = 1/2 + E/2 + 1/2 + 1/2 + E/4 = 3/2 + 3E/4. Solving: E - 3E/4 = 3/2, so E/4 = 3/2, thus E = 6. Alternatively, using states: state 0 (no heads), state 1 (one head), state 2 (two heads, absorbing). E₀ = 1 + (1/2)E₀ + (1/2)E₁, E₁ = 1 + (1/2)(0) + (1/2)E₀. Solving gives E₀ = 6.

---

## Q4: Three ants are placed on the vertices of an equilateral triangle. Each ant moves to a randomly chosen adjacent vertex. What is the probability that no ants collide?

**A4:** Each ant has 2 choices (clockwise or counterclockwise to the next vertex), so there are 2³ = 8 total outcomes. For no collision, all ants must move in the same direction (all clockwise or all counterclockwise). That's 2 favorable outcomes. Therefore, P(no collision) = 2/8 = 1/4 = 0.25. If they all move clockwise, they rotate positions without meeting. If they all move counterclockwise, same result. If they move in mixed directions, at least two will meet at an edge. The probability of collision is 6/8 = 3/4. This problem generalizes to n ants on an n-gon: P(no collision) = 2/2ⁿ = 1/2ⁿ⁻¹, since all must choose the same direction.

---

## Q5: There are 100 doors, all closed. You pass by 100 times. On the i-th pass, you toggle every i-th door (open if closed, close if open). Which doors are open at the end?

**A5:** A door is toggled once for each of its divisors. Door n is toggled on passes that are divisors of n. For example, door 12 is toggled on passes 1, 2, 3, 4, 6, 12. A door ends open if it's toggled an odd number of times, i.e., if it has an odd number of divisors. Only perfect squares have an odd number of divisors (because divisors come in pairs except for the square root). Therefore, doors that are perfect squares end open: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100. That's 10 doors. All other doors (with an even number of divisors) end closed. This is a classic problem demonstrating number theory: the number of divisors function d(n) is odd if and only if n is a perfect square.

---

## Q6: Three people each randomly get a hat that is either red or blue. They can see each other's hats but not their own. They must guess their own hat color simultaneously. What strategy maximizes the probability that at least one person guesses correctly?

**A6:** Strategy: the first person guesses the color that makes the number of red hats even (or odd, depending on agreement). If the first two hats have the same color, guess that color; if different, guess red (or blue, by prior agreement). More precisely: person 1 sees hats 2 and 3. If they're the same color, person 1 guesses the opposite color. If different, person 1 guesses red (say). Person 2 sees hat 3 and knows person 1's strategy. Person 2 can deduce their own hat. Person 3 uses similar logic. This guarantees at least one correct guess. Actually, a simpler strategy: person 1 guesses the color that makes the total number of red hats even (based on what they see). Person 2 and 3 can then deduce their hats. This ensures at least one is correct. The maximum probability is 3/4. Without coordination, each guesses randomly with P(correct) = 1/2, so P(at least one correct) = 1 - (1/2)³ = 7/8. But the coordinated strategy can do better in the worst case, guaranteeing at least one correct.

---

## Q7: In Russian roulette with one bullet in a 6-chamber revolver, should you go first or second?

**A7:** If you go first: P(death) = 1/6. If you go second: P(death) = P(first person survives) × P(you die | first survived) = (5/6) × (1/5) = 1/6. Wait, that's the same! But let's reconsider: if the cylinder is spun after each turn (randomized), then P(death | go first) = 1/6, and P(death | go second) = (5/6) × (1/6) = 5/36, since if the first person survives, the cylinder is respun, so you still have 1/6 chance. So going second is better: 5/36 < 1/6 = 6/36. If the cylinder is NOT respun (deterministic rotation), then going first: P(death) = 1/6. Going second: if first survives (5/6 chance), the bullet is now in position 1, and you're in position 2, so P(death) = 0. So P(death | go second) = (5/6) × 0 = 0. Going second is much better! In the typical interpretation (no respin), going second is safer. The answer depends on whether the cylinder is randomized between turns.

---

## Q8: There are 100 prisoners and 100 light switches. Each day, one prisoner is chosen at random to enter a room with the switches. They can flip any switches they want. When a prisoner declares that all have been in the room, if correct, all go free; if wrong, all die. What strategy guarantees freedom?

**A8:** Designate one prisoner as the "counter." All other prisoners follow: on their first visit, if switch 1 is off, turn it on; otherwise, do nothing. On subsequent visits, do nothing. The counter follows: on each visit, if switch 1 is on, turn it off and increment a count. Otherwise, do nothing. When the counter's count reaches 99, declare that all prisoners have visited. Why this works: each non-counter prisoner turns switch 1 on exactly once (on their first visit). The counter counts these events by turning the switch off. After 99 "on" events, all non-counters have visited at least once. The counter may have visited multiple times, but that's fine. This strategy guarantees success, though it may take a very long time (expected time is large, but finite with probability 1). The key insight is using one switch as a communication mechanism and one prisoner as a coordinator.

---

## Q9: You have 12 balls that look identical, but one has a different weight (heavier or lighter, unknown). You have a balance scale. What is the minimum number of weighings to find the odd ball and determine if it's heavier or lighter?

**A9:** Three weighings suffice. Strategy: First weighing: put 4 balls on each side. Cases: (1) They balance: the odd ball is among the remaining 4. Second weighing: put 3 of the remaining 4 against 3 known good balls. If they balance, the 4th is odd; third weighing determines if heavier/lighter. If they don't balance, you know which group of 3 contains the odd ball and whether it's heavier or lighter; third weighing identifies it among those 3. (2) They don't balance: the odd ball is among these 8. Label the heavy side H₁,H₂,H₃,H₄ and light side L₁,L₂,L₃,L₄. Second weighing: put H₁,H₂,L₁ on left vs H₃,L₂,good on right. Cases: (a) Balance: odd is H₄ (heavy) or L₃ or L₄ (light); third weighing: L₃ vs L₄ identifies it. (b) Left heavy: odd is H₁ or H₂ (heavy) or L₂ (light); third weighing: H₁ vs H₂ identifies it. (c) Right heavy: odd is H₃ (heavy) or L₁ (light); third weighing determines. This systematic approach always identifies the odd ball in 3 weighings. It's impossible in 2 weighings (3² = 9 < 12 possible outcomes needed).

---

## Q10: A random walk starts at 0. At each step, it moves +1 with probability p or -1 with probability (1-p). What is the expected number of steps to return to 0?

**A10:** For a symmetric random walk (p = 1/2), the expected return time is infinite! This is a classic result: while the walk returns to 0 with probability 1 (recurrent), the expected return time is infinite. For p ≠ 1/2 (asymmetric), the walk is transient (positive drift if p > 1/2, negative if p < 1/2), so it may never return, and the expected return time (conditioned on return) can be computed but is often infinite or undefined. For the symmetric case, using generating functions or renewal theory, one can show that while P(return to 0) = 1, E[return time] = ∞. This seems paradoxical but is true: the walk returns infinitely often, but the time between returns has a heavy-tailed distribution. For a one-dimensional symmetric random walk, the probability of return at time 2n is approximately 1/√(πn), and summing n × (1/√(πn)) diverges. This demonstrates the difference between almost sure events and events with finite expectation.

---

## Q11: What is the expected value of the maximum of N independent uniform random variables on [0,1]?

**A11:** Let X₁, X₂, ..., Xₙ be i.i.d. Uniform(0,1). Let M = max(X₁, ..., Xₙ). The CDF is F_M(x) = P(M ≤ x) = P(all Xᵢ ≤ x) = xⁿ for x ∈ [0,1]. The PDF is f_M(x) = nxⁿ⁻¹. The expected value is E[M] = ∫₀¹ x · nxⁿ⁻¹ dx = n∫₀¹ xⁿ dx = n · [xⁿ⁺¹/(n+1)]₀¹ = n/(n+1). For example, E[max of 2 uniforms] = 2/3, E[max of 10] = 10/11. As n → ∞, E[M] → 1, which makes sense: with many draws, the maximum approaches 1. More generally, for order statistics, E[X_(k)] = k/(n+1) for Uniform(0,1), where X_(k) is the k-th smallest. The minimum has E[min] = 1/(n+1). This result is useful in auction theory, extreme value theory, and optimization problems.

---

## Q12: You break a stick at two random points. What is the probability that the three pieces form a triangle?

**A12:** Let the stick have length 1. Choose two break points uniformly: X, Y ~ Uniform(0,1), with X < Y (order them). The pieces have lengths: X, Y-X, 1-Y. For a triangle, the triangle inequality must hold for all three sides: X + (Y-X) > 1-Y, X + (1-Y) > Y-X, and (Y-X) + (1-Y) > X. Simplifying: Y > 1-Y (i.e., Y > 1/2), X < Y-X (i.e., X < Y/2), and Y-X < X + 1-Y (i.e., Y < X + 1/2). In the (X,Y) space with 0 < X < Y < 1, the feasible region is: X < Y/2, Y > 1/2, and Y < X + 1/2. The area of the feasible region is 1/8. The total area of the triangle (0 < X < Y < 1) is 1/2. Therefore, P(triangle) = (1/8)/(1/2) = 1/4 = 0.25. Alternatively, by symmetry arguments or geometric probability, one can derive this result. The key insight is that the longest piece must be less than 1/2 for a triangle to be possible.

---

## Q13: 100 people board an airplane with 100 seats. The first person sits randomly. Each subsequent person sits in their assigned seat if available, otherwise randomly. What is the probability the last person gets their assigned seat?

**A13:** The answer is 1/2. Key insight: the last person gets their seat if and only if seat 1 is taken before seat 100. Consider the process: person 1 sits randomly. If they take seat 1, everyone else sits correctly, and person 100 gets seat 100. If person 1 takes seat k (k ≠ 1, 100), then person k is displaced and must sit randomly among remaining seats. This continues until someone takes seat 1 (then all remaining sit correctly) or seat 100 (then person 100 is displaced). By symmetry, seat 1 and seat 100 are equally likely to be taken first among the "problematic" seats. More formally, at each step, the set of available seats includes both seat 1 and seat 100 (until one is taken). Each is equally likely to be chosen. Therefore, P(person 100 gets seat 100) = P(seat 1 taken before seat 100) = 1/2. This elegant solution avoids complex recursive calculations and uses symmetry.

---

## Q14: A drunkard starts at position 0. Each step, they move +1 with probability p or -1 with probability (1-p). What is the probability they reach +N before -M (where N, M > 0)?

**A14:** For the symmetric case (p = 1/2), by symmetry and the optional stopping theorem, P(reach +N before -M) = M/(M+N). This is because the walk is a martingale, and using the fact that E[position] = 0 = N × P(reach +N) + (-M) × P(reach -M), with P(reach +N) + P(reach -M) = 1. Solving gives P(reach +N) = M/(M+N). For the asymmetric case (p ≠ 1/2), the probability is: P(reach +N before -M) = [(1-p)/p]^M - 1 / {[(1-p)/p]^(M+N) - 1} if p ≠ 1/2. This comes from solving the difference equation: u(x) = p·u(x+1) + (1-p)·u(x-1) with boundary conditions u(N) = 1, u(-M) = 0, where u(x) is the probability of reaching +N before -M starting from x. The solution uses the fact that the process has exponential martingales. As a special case, if p > 1/2, the probability of reaching +N is higher, and if p < 1/2, it's lower, which matches intuition.

---

## Q15: Two people agree to meet between 12:00 and 1:00 PM. Each arrives uniformly at random in that hour and waits 15 minutes. What is the probability they meet?

**A15:** Let person A arrive at time X ~ Uniform(0,60) minutes past 12:00, and person B at Y ~ Uniform(0,60). They meet if |X - Y| ≤ 15 (within 15 minutes of each other). In the (X,Y) square [0,60] × [0,60], the meeting region is: |X - Y| ≤ 15, which forms two right triangles. The area of the non-meeting region (two triangles) is 2 × (1/2) × 45 × 45 = 2025. The total area is 3600. Therefore, P(meet) = 1 - 2025/3600 = 1575/3600 = 7/16 = 0.4375. Alternatively, compute the area of the meeting region directly: it's the square minus two triangles, giving area = 3600 - 2025 = 1575, so P = 1575/3600 = 7/16. More generally, if waiting time is w and meeting window is T, P(meet) = 1 - [(T-w)/T]² = 2w/T - (w/T)² for w ≤ T. For w = T, P = 1 (they always meet if they wait the full hour).

---

## Q16: Explain the secretary problem (optimal stopping problem).

**A16:** You interview N candidates sequentially. After each interview, you must decide to hire or continue. You can't go back. You want to maximize the probability of hiring the best candidate. Optimal strategy: reject the first k candidates (where k ≈ N/e), then hire the first candidate better than all previous ones. The probability of success is approximately 1/e ≈ 0.368. Derivation: if you reject the first k and the best overall is in position i (i > k), you succeed if the best of the first (i-1) is in positions 1..k (so you haven't hired yet). P(success | best is position i) = k/(i-1) for i > k. P(success) = Σᵢ₌ₖ₊₁ⁿ (1/n) × (k/(i-1)) ≈ (k/n) × ln(n/k). Maximizing gives k/n ≈ 1/e, so k ≈ N/e. This is the "37% rule": look at 37% of options, then pick the next best one. Applications: hiring, house hunting, dating, and any sequential decision with no recall. The strategy balances exploration (seeing enough options) and exploitation (not waiting too long).

---

## Q17: Explain the two envelope paradox.

**A17:** You're given two envelopes. One contains twice as much as the other. You pick one, open it, see $X. You're offered to switch. Should you? Paradox: The other envelope has $2X with probability 1/2 and $X/2 with probability 1/2. Expected value of switching = (1/2)($2X) + (1/2)($X/2) = $X + $X/4 = $1.25X > X. So you should switch! But by symmetry, the same argument applies after switching, suggesting you should switch back, creating a paradox. Resolution: The error is assuming the other envelope is equally likely to be 2X or X/2 regardless of X. But if you see $100, the pair might be ($50, $100) or ($100, $200). The prior distribution matters. If envelopes are filled by: choose a random amount Y, put Y in one and 2Y in the other, then if you see X, P(other is 2X | you see X) = P(Y = X) / [P(Y = X) + P(Y = X/2)]. Without a prior on Y, the calculation is invalid. With a proper prior (e.g., uniform on [0, M] with M → ∞, or a proper distribution), the paradox resolves, and switching doesn't help on average. The paradox highlights the importance of proper probability modeling.

---

## Q18: What is Bertrand's random chord problem?

**A18:** Bertrand's paradox: "What is the probability that a random chord of a circle is longer than the side of an inscribed equilateral triangle?" The answer depends on how "random" is defined, leading to different answers: (1) Random endpoints: choose two random points on the circumference. P(long chord) = 1/3. (2) Random radius and point: choose a random radius and a random point on it. P(long chord) = 1/2. (3) Random midpoint: choose a random point in the circle as the chord's midpoint. P(long chord) = 1/4. All three methods seem "random" but give different answers! This is a paradox showing that "random" needs specification. The problem is that there's no unique uniform distribution over chords—different parameterizations give different measures. Resolution: specify the sampling method explicitly. In practice, the choice depends on the physical process generating chords. This paradox illustrates that probability problems require careful definition of the sample space and probability measure. It's a cautionary tale about assuming "uniform" without specifying the measure.

---

## Q19: What is the expected number of trials to collect all N types of coupons?

**A19:** This is the coupon collector problem. Let T be the number of trials to collect all N types. We can write T = T₁ + T₂ + ... + Tₙ, where Tᵢ is the number of trials to get the i-th new coupon after having (i-1) distinct coupons. T₁ = 1 (first coupon is always new). For Tᵢ (i ≥ 2), after having (i-1) coupons, each trial has probability p = (N - i + 1)/N of getting a new coupon. Tᵢ follows a geometric distribution with success probability p, so E[Tᵢ] = 1/p = N/(N - i + 1). Therefore, E[T] = Σᵢ₌₁ⁿ E[Tᵢ] = Σᵢ₌₁ⁿ N/(N - i + 1) = N × Σⱼ₌₁ⁿ (1/j) = N × Hₙ, where Hₙ is the n-th harmonic number. Hₙ ≈ ln(n) + γ (Euler-Mascheroni constant ≈ 0.577), so E[T] ≈ N × ln(N) for large N. For example, E[collect all 100 coupons] ≈ 100 × ln(100) ≈ 460 trials. The variance is also known: Var(T) = N² × Σⱼ₌₁ⁿ (1/j²) - N × Hₙ. This problem appears in many contexts: collecting trading cards, covering all nodes in a random walk, or testing all code paths.

---

## Q20: Five pirates must divide 100 gold coins. They vote, and if ≥50% approve, the division stands; otherwise, the proposer is thrown overboard and the next senior pirate proposes. Pirates are rational, greedy, and bloodthirsty. What division does the senior pirate propose?

**A20:** Work backwards from the end. If only pirate 5 remains: gets all 100. If pirates 4 and 5 remain: pirate 4 proposes (100, 0); pirate 4 votes yes, so it passes. If pirates 3, 4, 5 remain: pirate 3 needs one more vote. Pirate 4 gets 0 if proposal fails, so pirate 3 offers (99, 0, 1) to pirates 3, 4, 5. Pirates 3 and 5 vote yes. If pirates 2, 3, 4, 5 remain: pirate 2 needs two votes. Pirates 3 and 4 get 0 and 0 if proposal fails, so pirate 2 offers (98, 0, 1, 1) to pirates 2, 3, 4, 5. Pirates 2, 3, 4 vote yes. If all five remain: pirate 1 needs two votes. Pirates 2, 3, 4 would get 0, 1, 1 if proposal fails. So pirate 1 offers (97, 0, 1, 2, 0) or (97, 0, 1, 0, 2) to pirates 1, 2, 3, 4, 5. Pirates 1, 3, and one of {4,5} vote yes. Actually, careful: if proposal fails, pirates 2-5 get (98, 0, 1, 1). So pirate 1 offers (97, 0, 1, 2, 0) giving pirates 1, 3, 4 a total of 100. Pirates 1, 3, 4 vote yes (3/5 majority). The senior pirate keeps 97, gives 1 to the 3rd, 2 to the 4th, and 0 to the 2nd and 5th. This demonstrates backward induction and game theory.

---
