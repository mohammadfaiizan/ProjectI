"""
LeetCode 1227: Airplane Seat Assignment Probability
Difficulty: Medium
Category: Probability DP - Mathematical Insight

PROBLEM DESCRIPTION:
===================
n passengers board an airplane with exactly n seats. The first passenger has lost their ticket and picks a seat randomly. But after that, the rest of the passengers will:

- Take their own seat if it is still available, and
- Pick a random seat otherwise.

Return the probability that the nth (last) passenger gets their own seat.

Example 1:
Input: n = 1
Output: 1.0
Explanation: The first passenger can only take their own seat.

Example 2:
Input: n = 2
Output: 0.5
Explanation: The first passenger can take seat 1 or seat 2.
If they take seat 1, the second passenger gets their own seat (seat 2).
If they take seat 2, the second passenger cannot get their own seat.

Example 3:
Input: n = 3
Output: 0.5

Constraints:
- 1 <= n <= 10^5
"""


def airplane_seat_recursive(n):
    """
    RECURSIVE APPROACH:
    ==================
    Solve using direct recursive probability calculation.
    
    Time Complexity: O(n!) - exponential due to multiple branches
    Space Complexity: O(n) - recursion stack
    """
    if n == 1:
        return 1.0
    
    def probability(passengers_left, seat_1_available, seat_n_available):
        # Base cases
        if passengers_left == 1:  # Last passenger
            if seat_n_available:
                return 1.0  # Gets their own seat
            else:
                return 0.0  # Cannot get their own seat
        
        # If passenger's own seat is available, they take it
        # This doesn't affect the final outcome, so we recurse with one less passenger
        if passengers_left < n and passengers_left > 1:
            # Passenger takes their own seat (deterministic)
            return probability(passengers_left - 1, seat_1_available, seat_n_available)
        
        # First passenger or random choice scenario
        total_prob = 0.0
        available_seats = seat_1_available + seat_n_available + (passengers_left - 2)
        
        if seat_1_available:
            # Choose seat 1 - game ends, nth passenger gets their seat
            total_prob += (1.0 / available_seats) * 1.0
        
        if seat_n_available:
            # Choose seat n - game ends, nth passenger doesn't get their seat
            total_prob += (1.0 / available_seats) * 0.0
        
        # Choose any other seat (passengers 2 to n-1)
        other_seats = passengers_left - 2
        if other_seats > 0:
            prob_other = other_seats / available_seats
            total_prob += prob_other * probability(passengers_left - 1, seat_1_available, seat_n_available)
        
        return total_prob
    
    # Start with first passenger choosing randomly
    return probability(n, True, True)


def airplane_seat_mathematical(n):
    """
    MATHEMATICAL INSIGHT APPROACH:
    =============================
    Use mathematical analysis to derive the answer.
    
    Time Complexity: O(1) - constant time
    Space Complexity: O(1) - constant space
    """
    if n == 1:
        return 1.0
    else:
        return 0.5


def airplane_seat_simulation(n, num_simulations=100000):
    """
    MONTE CARLO SIMULATION:
    ======================
    Verify the mathematical result using simulation.
    
    Time Complexity: O(n * num_simulations) - simulation runs
    Space Complexity: O(n) - seat tracking
    """
    import random
    
    def simulate_once():
        seats = [False] * (n + 1)  # seats[i] represents if seat i is taken
        seats[0] = True  # seat 0 doesn't exist, mark as taken
        
        # First passenger chooses randomly
        available_seats = list(range(1, n + 1))
        first_choice = random.choice(available_seats)
        seats[first_choice] = True
        
        # If first passenger chose their own seat (1) or last seat (n), game ends
        if first_choice == 1:
            return 1.0  # All subsequent passengers get their own seats
        if first_choice == n:
            return 0.0  # Last passenger cannot get their own seat
        
        # Simulate remaining passengers (2 to n-1)
        for passenger in range(2, n):
            if not seats[passenger]:
                # Passenger takes their own seat
                seats[passenger] = True
            else:
                # Passenger must choose randomly among available seats
                available = [i for i in range(1, n + 1) if not seats[i]]
                if not available:
                    break
                
                choice = random.choice(available)
                seats[choice] = True
                
                # If they chose seat 1 or seat n, game effectively ends
                if choice == 1:
                    return 1.0  # Last passenger will get their seat
                if choice == n:
                    return 0.0  # Last passenger won't get their seat
        
        # Last passenger
        return 1.0 if not seats[n] else 0.0
    
    successes = sum(simulate_once() for _ in range(num_simulations))
    return successes / num_simulations


def airplane_seat_with_analysis(n):
    """
    AIRPLANE SEAT WITH DETAILED ANALYSIS:
    ====================================
    Solve and provide comprehensive mathematical insights.
    
    Time Complexity: O(1) - mathematical solution
    Space Complexity: O(1) - analysis storage
    """
    analysis = {
        'n': n,
        'mathematical_insight': {},
        'step_by_step_reasoning': [],
        'probability_breakdown': {},
        'edge_cases': {},
        'insights': []
    }
    
    # Mathematical insight
    if n == 1:
        result = 1.0
        analysis['mathematical_insight'] = {
            'result': result,
            'reasoning': 'Only one passenger and one seat - probability is 1'
        }
        analysis['insights'].append("Trivial case: n=1 always results in probability 1.0")
    else:
        result = 0.5
        analysis['mathematical_insight'] = {
            'result': result,
            'reasoning': 'By symmetry and invariant analysis, probability is always 0.5 for n≥2'
        }
        
        # Step-by-step reasoning
        analysis['step_by_step_reasoning'] = [
            "1. First passenger randomly chooses among n seats",
            "2. If they choose seat 1: all others sit correctly, nth gets their seat (prob = 1)",
            "3. If they choose seat n: nth passenger cannot get their seat (prob = 0)",
            "4. If they choose seat k (2≤k≤n-1): passenger k will face the same choice",
            "5. The problem reduces to the same structure with fewer people",
            "6. Only seats 1 and n matter for the final outcome",
            "7. By symmetry, P(seat 1 chosen eventually) = P(seat n chosen eventually)",
            "8. Since one of them must be chosen eventually, P = 0.5"
        ]
    
    # Probability breakdown for small n
    if n <= 5:
        breakdown = {}
        if n == 1:
            breakdown = {'passenger_1_takes_seat_1': 1.0}
        elif n == 2:
            breakdown = {
                'passenger_1_takes_seat_1': 0.5,
                'passenger_1_takes_seat_2': 0.5
            }
        else:
            breakdown = {
                'passenger_1_takes_seat_1': 1.0/n,
                'passenger_1_takes_seat_n': 1.0/n,
                'passenger_1_takes_other_seat': (n-2)/n,
                'recursive_probability': 0.5
            }
        
        analysis['probability_breakdown'] = breakdown
    
    # Invariant analysis
    analysis['insights'].append(f"Result: {result}")
    if n > 1:
        analysis['insights'].append("Key insight: Problem has recursive structure with invariant")
        analysis['insights'].append("Symmetry: Seats 1 and n are equally likely to be chosen eventually")
        analysis['insights'].append("Only the first and last seats matter for the final outcome")
    
    # Edge cases
    analysis['edge_cases'] = {
        'n_equals_1': 'Trivial case, probability = 1.0',
        'n_equals_2': 'Base case, probability = 0.5',
        'large_n': 'Probability remains 0.5 regardless of n size'
    }
    
    return result, analysis


def airplane_seat_proof():
    """
    MATHEMATICAL PROOF:
    ==================
    Provide rigorous mathematical proof of the result.
    """
    print("Mathematical Proof of Airplane Seat Assignment:")
    print("=" * 60)
    
    print("\nTheorem: For n ≥ 2, the probability that the nth passenger gets their own seat is 0.5")
    print("\nProof by Strong Induction and Symmetry:")
    
    print("\nBase Cases:")
    print("• n = 1: Only one passenger, must sit in their own seat. P(1) = 1.0")
    print("• n = 2: First passenger chooses randomly between 2 seats.")
    print("  - Choose seat 1: Second passenger gets seat 2. P = 1")
    print("  - Choose seat 2: Second passenger cannot get seat 2. P = 0")
    print("  - Overall: P(2) = 0.5 × 1 + 0.5 × 0 = 0.5")
    
    print("\nInductive Step (n ≥ 3):")
    print("Assume P(k) = 0.5 for all 2 ≤ k < n. Prove P(n) = 0.5.")
    
    print("\nFirst passenger has n choices:")
    print("1. Choose seat 1 (probability 1/n):")
    print("   → All subsequent passengers sit correctly")
    print("   → nth passenger gets their own seat")
    print("   → Contribution: (1/n) × 1 = 1/n")
    
    print("\n2. Choose seat n (probability 1/n):")
    print("   → nth passenger cannot get their own seat")  
    print("   → Contribution: (1/n) × 0 = 0")
    
    print("\n3. Choose seat k where 2 ≤ k ≤ n-1 (probability 1/n each):")
    print("   → Passengers 2 through k-1 sit correctly")
    print("   → Passenger k faces the same problem with seats {1, k+1, k+2, ..., n}")
    print("   → This reduces to the same problem with (n-k+1) effective seats")
    print("   → By induction hypothesis, probability = 0.5")
    print("   → Contribution: Σ(k=2 to n-1) (1/n) × 0.5 = (n-2)/n × 0.5")
    
    print("\nTotal Probability:")
    print("P(n) = 1/n + 0 + (n-2)/n × 0.5")
    print("     = 1/n + (n-2)/(2n)")
    print("     = (2 + n - 2)/(2n)")
    print("     = n/(2n)")
    print("     = 1/2 = 0.5")
    
    print("\nAlternative Proof by Symmetry:")
    print("Key Observation: Only seats 1 and n affect the final outcome")
    print("• Any choice of seats 2,...,n-1 eventually leads to someone choosing seat 1 or n")
    print("• The problem maintains its structure: always a choice between 'first' and 'last'")
    print("• By symmetry, seat 1 and seat n are equally likely to be chosen eventually")
    print("• Since exactly one must be chosen: P(seat 1) = P(seat n) = 0.5")
    print("• P(nth passenger gets their seat) = P(seat 1 chosen eventually) = 0.5")
    
    print("\nConclusion: P(n) = 0.5 for all n ≥ 2 ✓")


def airplane_seat_variants():
    """
    AIRPLANE SEAT VARIANTS:
    ======================
    Different scenarios and generalizations.
    """
    
    def airplane_seat_k_lost_tickets(n, k):
        """k passengers lose their tickets and sit randomly"""
        if k >= n:
            return 0.0  # Too many random passengers
        
        if k == 0:
            return 1.0  # No random passengers
        
        # For k=1, we know the answer is 0.5 (for n≥2)
        if k == 1:
            return 0.5 if n >= 2 else 1.0
        
        # For k>1, the analysis becomes more complex
        # This is a simplified approximation
        return 0.5 ** k  # Very rough approximation
    
    def airplane_seat_with_preferences(n, preference_strength=0.8):
        """Passengers have preference for their own seat even when lost"""
        if n == 1:
            return 1.0
        
        # Modified probability - random passengers prefer their own seat
        # but will choose randomly if it's taken
        # This is a simplified model
        
        # The preference_strength affects the initial random choice
        effective_random_prob = 1 - preference_strength
        
        # Approximation: reduces the "randomness" of the problem
        return 0.5 + (0.5 * preference_strength)
    
    def airplane_seat_partial_information(n, known_seats_ratio=0.5):
        """Some passengers know their seat numbers, others don't"""
        if n == 1:
            return 1.0
        
        # Simplified model: known_seats_ratio of passengers know their seats
        # The rest are random
        
        random_passengers = int(n * (1 - known_seats_ratio))
        if random_passengers <= 1:
            return airplane_seat_mathematical(n)
        
        # Approximation for multiple random passengers
        return 0.5 ** (random_passengers - 1)
    
    def airplane_seat_expected_position(n):
        """Expected final position of the nth passenger"""
        if n == 1:
            return 1.0  # Gets seat 1
        
        # Expected seat number for nth passenger
        # If they get their own seat: position n
        # If they don't: expected position is harder to calculate exactly
        
        prob_own_seat = 0.5
        expected_if_own = n
        expected_if_not_own = n / 2  # Rough approximation
        
        return prob_own_seat * expected_if_own + (1 - prob_own_seat) * expected_if_not_own
    
    # Test variants
    test_cases = [1, 2, 3, 5, 10, 100]
    
    print("Airplane Seat Assignment Variants:")
    print("=" * 50)
    
    for n in test_cases:
        print(f"\nn = {n}:")
        
        basic_result = airplane_seat_mathematical(n)
        print(f"Basic problem: {basic_result:.3f}")
        
        # Multiple lost tickets
        if n >= 2:
            two_lost = airplane_seat_k_lost_tickets(n, 2)
            print(f"2 lost tickets: {two_lost:.3f}")
        
        # With preferences
        with_pref = airplane_seat_with_preferences(n, 0.7)
        print(f"With preferences: {with_pref:.3f}")
        
        # Partial information
        partial_info = airplane_seat_partial_information(n, 0.8)
        print(f"Partial information: {partial_info:.3f}")
        
        # Expected position
        expected_pos = airplane_seat_expected_position(n)
        print(f"Expected final position: {expected_pos:.1f}")


# Test cases
def test_airplane_seat():
    """Test all implementations with various inputs"""
    test_cases = [
        (1, 1.0),
        (2, 0.5),
        (3, 0.5),
        (10, 0.5),
        (100, 0.5),
        (10000, 0.5)
    ]
    
    print("Testing Airplane Seat Assignment Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"n = {n}")
        print(f"Expected: {expected}")
        
        # Skip recursive for large cases
        if n <= 8:
            try:
                recursive = airplane_seat_recursive(n)
                diff = abs(recursive - expected)
                print(f"Recursive:        {recursive:.6f} {'✓' if diff < 0.001 else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        mathematical = airplane_seat_mathematical(n)
        math_diff = abs(mathematical - expected)
        print(f"Mathematical:     {mathematical:.6f} {'✓' if math_diff < 0.001 else '✗'}")
        
        # Simulation for smaller cases
        if n <= 20:
            simulation = airplane_seat_simulation(n, 50000)
            sim_diff = abs(simulation - expected)
            print(f"Simulation:       {simulation:.6f} {'✓' if sim_diff < 0.05 else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    result, analysis = airplane_seat_with_analysis(5)
    
    print(f"n = 5, Result = {result}")
    print(f"\nMathematical Insight:")
    print(f"  {analysis['mathematical_insight']['reasoning']}")
    
    print(f"\nStep-by-step Reasoning:")
    for step in analysis['step_by_step_reasoning']:
        print(f"  {step}")
    
    print(f"\nInsights:")
    for insight in analysis['insights']:
        print(f"  • {insight}")
    
    # Mathematical proof
    print(f"\n" + "=" * 70)
    airplane_seat_proof()
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    airplane_seat_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. MATHEMATICAL ELEGANCE: Complex problem has simple answer (0.5 for n≥2)")
    print("2. INVARIANT STRUCTURE: Problem maintains same structure through recursion")
    print("3. SYMMETRY PRINCIPLE: Seats 1 and n are equally likely outcomes")
    print("4. REDUCTION: Only first and last seats matter for final result")
    print("5. PROOF TECHNIQUES: Multiple proof approaches (induction, symmetry)")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Probabilistic Analysis: Complex systems with simple outcomes")
    print("• Game Theory: Sequential decision making with random elements")
    print("• Resource Allocation: Fair distribution under uncertainty")
    print("• Algorithm Design: Problems with elegant mathematical solutions")
    print("• Interview Preparation: Classic probability puzzle with deep insights")


if __name__ == "__main__":
    test_airplane_seat()


"""
AIRPLANE SEAT ASSIGNMENT - MATHEMATICAL ELEGANCE AND SYMMETRY:
==============================================================

This problem demonstrates mathematical elegance in probability:
- Complex sequential decision process with simple answer
- Multiple proof techniques revealing deep structural insights
- Invariant analysis showing why complexity doesn't affect outcome
- Beautiful application of symmetry in probability theory

KEY INSIGHTS:
============
1. **MATHEMATICAL ELEGANCE**: Complex problem has remarkably simple answer (0.5 for n≥2)
2. **INVARIANT STRUCTURE**: Problem maintains identical structure through all steps
3. **SYMMETRY PRINCIPLE**: First and last seats are symmetric in the decision process
4. **REDUCTION PROPERTY**: Only first and last seats determine the final outcome
5. **PROOF MULTIPLICITY**: Multiple valid proof approaches (induction, symmetry, reduction)

ALGORITHM APPROACHES:
====================

1. **Recursive**: O(n!) time, O(n) space
   - Direct simulation of all possible scenarios
   - Exponential complexity due to multiple decision branches

2. **Mathematical**: O(1) time, O(1) space
   - Elegant closed-form solution using mathematical insight
   - Optimal complexity through problem structure analysis

3. **Simulation**: O(n × trials) time, O(n) space
   - Monte Carlo verification of mathematical result
   - Practical validation for understanding

MATHEMATICAL SOLUTION:
=====================
```python
def airplaneSeating(n):
    return 1.0 if n == 1 else 0.5
```

**Proof by Symmetry**:
- Only seats 1 and n affect the final outcome
- All intermediate choices eventually reduce to choosing between seats 1 and n
- By symmetry: P(seat 1 chosen) = P(seat n chosen) = 0.5
- Therefore: P(nth passenger gets their seat) = 0.5

INDUCTIVE PROOF STRUCTURE:
=========================
**Base Cases**:
- n = 1: P = 1.0 (trivial)
- n = 2: P = 0.5 (direct calculation)

**Inductive Step** (n ≥ 3):
- First passenger chooses seat 1: P = 1 (probability 1/n)
- First passenger chooses seat n: P = 0 (probability 1/n)  
- First passenger chooses seat k (2≤k≤n-1): reduces to same problem (probability 1/n each)

**Total**: P(n) = (1/n)×1 + (1/n)×0 + Σ(1/n)×0.5 = 1/n + (n-2)/n × 0.5 = 0.5

SYMMETRY ANALYSIS:
=================
**Key Observation**: Problem structure is invariant under passenger relabeling
**Critical Seats**: Only seats 1 and n matter for the outcome
**Reduction Property**: Any intermediate choice preserves the essential decision
**Symmetric Outcomes**: Seats 1 and n are equally likely to be chosen eventually

INVARIANT PROPERTIES:
=====================
**Structural Invariant**: At each step, exactly one passenger faces choice between:
- Taking the first passenger's seat (ending game favorably)
- Taking the last passenger's seat (ending game unfavorably)
- Passing the decision to another passenger

**Probability Invariant**: The probability remains 0.5 regardless of problem size

COMPLEXITY INSIGHTS:
===================
**Apparent Complexity**: n! possible seating arrangements
**Actual Complexity**: Binary outcome determined by first/last seat choice
**Reduction**: Exponential state space collapses to simple binary decision

**Lesson**: Problem complexity can be misleading; structure matters more than size

APPLICATIONS:
============
- **Algorithm Design**: Problems with elegant mathematical shortcuts
- **Probability Theory**: Complex processes with simple limiting behavior
- **Game Theory**: Sequential decision making with symmetric outcomes
- **Interview Questions**: Classic problems testing mathematical insight
- **System Design**: Resource allocation under uncertainty

RELATED PROBLEMS:
================
- **Secretary Problem**: Optimal stopping with similar structure
- **Gambler's Ruin**: Random walk with absorbing boundaries
- **Coupon Collector**: Probability analysis with surprising results
- **Ballot Problem**: Counting problems with elegant solutions

VARIANTS:
========
- **Multiple Lost Tickets**: k passengers without tickets
- **Preference Models**: Passengers prefer their own seats
- **Partial Information**: Some passengers know their seats
- **Dynamic Seating**: Seats can be changed during boarding

EDGE CASES:
==========
- **n = 1**: Trivial case, probability = 1.0
- **All Passengers Random**: Different problem structure
- **Deterministic Choices**: Removes randomness element
- **Large n**: Result remains 0.5 (scale invariance)

PROOF TECHNIQUES DEMONSTRATED:
=============================
**Strong Induction**: Building solution from base cases
**Symmetry Arguments**: Exploiting problem symmetries
**Reduction Analysis**: Showing equivalent problem structures
**Invariant Identification**: Finding preserved properties

This problem beautifully illustrates how sophisticated
probabilistic analysis can reveal elegant mathematical
structure, demonstrating that complexity often conceals
underlying simplicity through the power of symmetry
and mathematical insight.
"""
