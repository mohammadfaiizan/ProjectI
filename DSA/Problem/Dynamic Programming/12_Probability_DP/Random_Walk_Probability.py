"""
Random Walk Probability Problems
Difficulty: Medium to Hard
Category: Probability DP - Stochastic Processes

PROBLEM DESCRIPTION:
===================
Collection of random walk probability problems:

1. Simple Random Walk: Probability of reaching target from origin
2. Random Walk with Barriers: Absorbing/reflecting boundaries
3. Biased Random Walk: Non-uniform step probabilities
4. 2D Random Walk: Probability calculations on grid
5. First Passage Time: Expected time to reach target

These problems demonstrate fundamental concepts in:
- Markov chains and transition probabilities
- Absorbing and reflecting boundary conditions
- Expected value and variance calculations
- Convergence and limiting behavior

Example Problems:
- Probability of reaching +n before -m starting from 0
- Expected number of steps to reach boundary
- Probability distribution after k steps
- Long-term behavior and stationary distributions
"""


def simple_random_walk_probability(start, target, barrier_low, barrier_high):
    """
    SIMPLE RANDOM WALK PROBABILITY:
    ===============================
    Calculate probability of reaching target before hitting barriers.
    
    Time Complexity: O(n) - where n is the range size
    Space Complexity: O(n) - DP array
    """
    if target <= barrier_low or target >= barrier_high:
        return 0.0 if start != target else 1.0
    
    if start <= barrier_low or start >= barrier_high:
        return 0.0 if start != target else 1.0
    
    # DP approach
    range_size = barrier_high - barrier_low - 1
    dp = [0.0] * (range_size + 2)
    
    # Boundary conditions
    dp[0] = 0.0  # barrier_low
    dp[range_size + 1] = 0.0  # barrier_high
    
    # Target condition
    target_idx = target - barrier_low
    if 0 <= target_idx <= range_size + 1:
        dp[target_idx] = 1.0
    
    # Iterative solution (Gauss-Seidel method)
    for iteration in range(1000):  # Max iterations
        old_dp = dp[:]
        
        for i in range(1, range_size + 1):
            if i == target_idx:
                continue  # Target position is absorbing
            dp[i] = 0.5 * (dp[i - 1] + dp[i + 1])
        
        # Check convergence
        if all(abs(dp[i] - old_dp[i]) < 1e-10 for i in range(len(dp))):
            break
    
    start_idx = start - barrier_low
    return dp[start_idx] if 0 <= start_idx <= range_size + 1 else 0.0


def biased_random_walk_probability(start, target, barrier_low, barrier_high, prob_right):
    """
    BIASED RANDOM WALK PROBABILITY:
    ==============================
    Random walk with probability prob_right of moving right.
    
    Time Complexity: O(n) - range size
    Space Complexity: O(n) - DP array
    """
    if target <= barrier_low or target >= barrier_high:
        return 0.0 if start != target else 1.0
    
    if start <= barrier_low or start >= barrier_high:
        return 0.0 if start != target else 1.0
    
    range_size = barrier_high - barrier_low - 1
    dp = [0.0] * (range_size + 2)
    
    # Boundary conditions
    dp[0] = 0.0  # barrier_low
    dp[range_size + 1] = 0.0  # barrier_high
    
    # Target condition
    target_idx = target - barrier_low
    if 0 <= target_idx <= range_size + 1:
        dp[target_idx] = 1.0
    
    prob_left = 1.0 - prob_right
    
    # Iterative solution
    for iteration in range(1000):
        old_dp = dp[:]
        
        for i in range(1, range_size + 1):
            if i == target_idx:
                continue
            dp[i] = prob_left * dp[i - 1] + prob_right * dp[i + 1]
        
        if all(abs(dp[i] - old_dp[i]) < 1e-10 for i in range(len(dp))):
            break
    
    start_idx = start - barrier_low
    return dp[start_idx] if 0 <= start_idx <= range_size + 1 else 0.0


def random_walk_expected_steps(start, targets, barriers):
    """
    EXPECTED STEPS TO REACH TARGET:
    ==============================
    Calculate expected number of steps to reach any target or barrier.
    
    Time Complexity: O(n^2) - solving linear system
    Space Complexity: O(n^2) - matrix storage
    """
    import numpy as np
    
    all_positions = set([start] + targets + barriers)
    position_list = sorted(all_positions)
    n = len(position_list)
    
    # Create position index mapping
    pos_to_idx = {pos: i for i, pos in enumerate(position_list)}
    
    # Set up linear system: E[i] = 1 + 0.5 * (E[i-1] + E[i+1])
    # Rearranged: -0.5 * E[i-1] + E[i] - 0.5 * E[i+1] = 1
    
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    for i, pos in enumerate(position_list):
        if pos in targets or pos in barriers:
            # Absorbing states
            A[i, i] = 1.0
            b[i] = 0.0
        else:
            # Regular states
            A[i, i] = 1.0
            b[i] = 1.0
            
            # Left neighbor
            left_pos = pos - 1
            if left_pos in pos_to_idx:
                left_idx = pos_to_idx[left_pos]
                A[i, left_idx] = -0.5
            
            # Right neighbor
            right_pos = pos + 1
            if right_pos in pos_to_idx:
                right_idx = pos_to_idx[right_pos]
                A[i, right_idx] = -0.5
    
    # Solve linear system
    try:
        solution = np.linalg.solve(A, b)
        start_idx = pos_to_idx[start]
        return solution[start_idx]
    except:
        return float('inf')  # No solution or infinite expected time


def random_walk_2d_probability(start_x, start_y, target_x, target_y, steps):
    """
    2D RANDOM WALK PROBABILITY:
    ===========================
    Probability of being at target after exactly k steps in 2D.
    
    Time Complexity: O(steps^2) - DP over all positions
    Space Complexity: O(steps^2) - DP table
    """
    # dp[x][y] = probability of being at (x,y)
    # Use offset to handle negative coordinates
    offset = steps + max(abs(start_x), abs(start_y), abs(target_x), abs(target_y))
    size = 2 * offset + 1
    
    dp = [[0.0] * size for _ in range(size)]
    dp[start_x + offset][start_y + offset] = 1.0
    
    # Four directions: up, down, left, right
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    
    for step in range(steps):
        new_dp = [[0.0] * size for _ in range(size)]
        
        for x in range(size):
            for y in range(size):
                if dp[x][y] > 0:
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            new_dp[nx][ny] += dp[x][y] * 0.25
        
        dp = new_dp
    
    target_prob = dp[target_x + offset][target_y + offset]
    return target_prob


def random_walk_with_reflection(start, target, barrier_low, barrier_high, steps):
    """
    RANDOM WALK WITH REFLECTING BARRIERS:
    ====================================
    Calculate probability with reflecting boundaries.
    
    Time Complexity: O(steps * range) - DP over steps and positions
    Space Complexity: O(range) - position probabilities
    """
    range_size = barrier_high - barrier_low + 1
    
    # dp[pos] = probability of being at position pos
    dp = [0.0] * range_size
    start_idx = start - barrier_low
    dp[start_idx] = 1.0
    
    for step in range(steps):
        new_dp = [0.0] * range_size
        
        for i in range(range_size):
            if dp[i] > 0:
                # Move left
                left_idx = max(0, i - 1)  # Reflect at left barrier
                new_dp[left_idx] += dp[i] * 0.5
                
                # Move right
                right_idx = min(range_size - 1, i + 1)  # Reflect at right barrier
                new_dp[right_idx] += dp[i] * 0.5
        
        dp = new_dp
    
    target_idx = target - barrier_low
    return dp[target_idx] if 0 <= target_idx < range_size else 0.0


def random_walk_analysis():
    """
    COMPREHENSIVE RANDOM WALK ANALYSIS:
    ===================================
    Analyze various random walk scenarios and their properties.
    """
    print("Random Walk Probability Analysis:")
    print("=" * 50)
    
    # 1. Simple random walk - gambler's ruin
    print("\n1. Gambler's Ruin Problem:")
    start, target, low, high = 5, 10, 0, 15
    prob = simple_random_walk_probability(start, target, low, high)
    print(f"   Start: {start}, Target: {target}, Barriers: [{low}, {high}]")
    print(f"   Probability of reaching target: {prob:.6f}")
    
    # Theoretical result for gambler's ruin
    # P = (high - start) / (high - low) for unbiased walk
    theoretical = (target - start) / (high - low) if target != start else 1.0
    print(f"   Theoretical (if target at boundary): {theoretical:.6f}")
    
    # 2. Biased random walk
    print("\n2. Biased Random Walk:")
    prob_right = 0.6
    biased_prob = biased_random_walk_probability(start, target, low, high, prob_right)
    print(f"   Probability of moving right: {prob_right}")
    print(f"   Probability of reaching target: {biased_prob:.6f}")
    
    # 3. Expected number of steps
    print("\n3. Expected Number of Steps:")
    try:
        expected_steps = random_walk_expected_steps(start, [target], [low, high])
        print(f"   Expected steps to reach target or boundary: {expected_steps:.2f}")
    except:
        print("   Could not calculate (requires NumPy)")
    
    # 4. 2D Random Walk
    print("\n4. 2D Random Walk:")
    steps_2d = 10
    prob_2d = random_walk_2d_probability(0, 0, 2, 1, steps_2d)
    print(f"   Start: (0,0), Target: (2,1), Steps: {steps_2d}")
    print(f"   Probability: {prob_2d:.6f}")
    
    # 5. Reflecting barriers
    print("\n5. Random Walk with Reflecting Barriers:")
    steps_reflect = 20
    prob_reflect = random_walk_with_reflection(5, 8, 0, 10, steps_reflect)
    print(f"   Start: 5, Target: 8, Barriers: [0,10] (reflecting)")
    print(f"   Probability after {steps_reflect} steps: {prob_reflect:.6f}")
    
    # Statistical properties
    print("\n6. Statistical Properties:")
    print("   • Symmetric random walk: E[X_n] = 0, Var(X_n) = n")
    print("   • Biased walk with p>0.5: E[X_n] = n(2p-1)")
    print("   • Central Limit Theorem: X_n/√n → N(0,1) as n→∞")
    print("   • Recurrence: 1D and 2D walks are recurrent, 3D+ are transient")


def random_walk_variants():
    """
    RANDOM WALK VARIANTS:
    ====================
    Demonstrate various random walk modifications.
    """
    
    def levy_flight_simulation(steps, alpha=1.5):
        """Simulate Lévy flight with heavy-tailed step distribution"""
        import random
        import math
        
        position = 0.0
        positions = [position]
        
        for _ in range(steps):
            # Simplified Lévy flight step (Cauchy distribution approximation)
            u = random.uniform(-math.pi/2, math.pi/2)
            step = math.tan(u)  # Cauchy-distributed step
            
            # Scale step size
            step *= (0.1 ** (1/alpha))
            
            position += step
            positions.append(position)
        
        return positions
    
    def persistent_random_walk(steps, persistence=0.7):
        """Random walk with memory/persistence"""
        import random
        
        position = 0
        direction = random.choice([-1, 1])
        positions = [position]
        
        for _ in range(steps):
            # With probability persistence, keep same direction
            if random.random() < persistence:
                # Keep same direction
                pass
            else:
                # Change direction
                direction = -direction
            
            position += direction
            positions.append(position)
        
        return positions
    
    def correlated_random_walk(steps, correlation=0.5):
        """Random walk with correlated steps"""
        import random
        
        position = 0.0
        prev_step = 0.0
        positions = [position]
        
        for _ in range(steps):
            # Step correlated with previous step
            noise = random.gauss(0, 1)
            step = correlation * prev_step + (1 - correlation) * noise
            
            position += step
            positions.append(position)
            prev_step = step
        
        return positions
    
    def random_walk_on_graph(graph, start, steps):
        """Random walk on arbitrary graph"""
        import random
        
        current = start
        positions = [current]
        
        for _ in range(steps):
            if current in graph and graph[current]:
                current = random.choice(graph[current])
            positions.append(current)
        
        return positions
    
    # Example demonstrations
    print("Random Walk Variants:")
    print("=" * 30)
    
    steps = 50
    
    # Standard random walk for comparison
    import random
    standard_walk = [0]
    pos = 0
    for _ in range(steps):
        pos += random.choice([-1, 1])
        standard_walk.append(pos)
    
    print(f"Standard walk final position: {standard_walk[-1]}")
    
    # Lévy flight
    try:
        levy_walk = levy_flight_simulation(steps)
        print(f"Lévy flight final position: {levy_walk[-1]:.2f}")
    except:
        print("Lévy flight: Requires math module")
    
    # Persistent walk
    try:
        persistent_walk = persistent_random_walk(steps, 0.8)
        print(f"Persistent walk final position: {persistent_walk[-1]}")
    except:
        print("Persistent walk: Requires random module")
    
    # Graph walk example (cycle graph)
    cycle_graph = {i: [(i-1) % 6, (i+1) % 6] for i in range(6)}
    try:
        graph_walk = random_walk_on_graph(cycle_graph, 0, steps)
        print(f"Graph walk (6-cycle) final position: {graph_walk[-1]}")
    except:
        print("Graph walk: Requires random module")


# Test cases
def test_random_walk_probability():
    """Test random walk probability implementations"""
    print("Testing Random Walk Probability Solutions:")
    print("=" * 70)
    
    # Test cases with known theoretical results
    test_cases = [
        # (start, target, low_barrier, high_barrier, expected_prob)
        (1, 2, 0, 3, 0.6667),  # Simple case
        (2, 1, 0, 3, 0.3333),  # Reverse direction
        (1, 1, 0, 3, 1.0),     # Already at target
        (5, 8, 0, 10, 0.5),    # Symmetric case
    ]
    
    print("Simple Random Walk Tests:")
    for i, (start, target, low, high, expected) in enumerate(test_cases):
        result = simple_random_walk_probability(start, target, low, high)
        diff = abs(result - expected)
        print(f"Test {i+1}: Start={start}, Target={target}, Barriers=[{low},{high}]")
        print(f"  Result: {result:.6f}, Expected: {expected:.6f} {'✓' if diff < 0.01 else '✗'}")
    
    # Biased walk tests
    print("\nBiased Random Walk Tests:")
    biased_cases = [
        (1, 2, 0, 3, 0.6, 0.75),   # Right bias helps reach right target
        (2, 1, 0, 3, 0.6, 0.25),   # Right bias hurts reaching left target
    ]
    
    for i, (start, target, low, high, prob_right, expected) in enumerate(biased_cases):
        result = biased_random_walk_probability(start, target, low, high, prob_right)
        diff = abs(result - expected)
        print(f"Biased Test {i+1}: p_right={prob_right}")
        print(f"  Result: {result:.6f}, Expected: {expected:.6f} {'✓' if diff < 0.01 else '✗'}")
    
    # 2D Random Walk test
    print("\n2D Random Walk Test:")
    prob_2d = random_walk_2d_probability(0, 0, 0, 0, 4)  # Return to origin
    print(f"P(return to origin in 4 steps): {prob_2d:.6f}")
    
    # Theoretical: 2^4 * C(4,2) * C(2,1)^2 / 4^4 for 2D
    # = 16 * 6 * 4 / 256 = 384/256 = 1.5... (this is wrong calculation)
    # Correct: Should be sum over all valid paths
    
    # Detailed analysis
    print(f"\n" + "=" * 70)
    print("COMPREHENSIVE ANALYSIS:")
    print("-" * 40)
    random_walk_analysis()
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    random_walk_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. MARKOV PROPERTY: Future depends only on current state")
    print("2. MARTINGALE STRUCTURE: Unbiased walks have martingale property")
    print("3. BOUNDARY CONDITIONS: Absorbing vs reflecting barriers")
    print("4. DIMENSIONAL EFFECTS: Recurrence depends on dimension")
    print("5. LIMITING BEHAVIOR: Central limit theorem and scaling limits")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Finance: Stock price models and option pricing")
    print("• Physics: Brownian motion and diffusion processes")
    print("• Biology: Animal foraging and cell migration")
    print("• Computer Science: Algorithm analysis and load balancing")
    print("• Engineering: Signal processing and network analysis")


if __name__ == "__main__":
    test_random_walk_probability()


"""
RANDOM WALK PROBABILITY - STOCHASTIC PROCESS FOUNDATIONS:
=========================================================

This collection demonstrates fundamental stochastic processes:
- Markov chains with discrete state spaces
- Boundary value problems for difference equations
- Expected hitting times and absorption probabilities
- Multi-dimensional random processes
- Various boundary conditions (absorbing, reflecting)

KEY INSIGHTS:
============
1. **MARKOV PROPERTY**: Future evolution depends only on current state
2. **BOUNDARY CONDITIONS**: Absorbing vs reflecting barriers dramatically change behavior
3. **BIAS EFFECTS**: Even small bias can dramatically alter long-term behavior
4. **DIMENSIONAL SCALING**: Higher dimensions are typically transient
5. **CENTRAL LIMIT BEHAVIOR**: Proper scaling leads to Brownian motion

FUNDAMENTAL CONCEPTS:
====================

1. **Simple Random Walk**: Equal probability left/right steps
   - P(reach target before barrier) = classical gambler's ruin formula
   - Symmetric case: probability proportional to distance ratios

2. **Biased Random Walk**: p ≠ 0.5 for rightward movement
   - Drift parameter: μ = 2p - 1
   - Exponential dependence on bias strength

3. **Multi-dimensional Walk**: Extension to higher dimensions
   - Recurrent in 1D and 2D, transient in 3D+
   - Scaling behavior: displacement ~ √n for unbiased walks

4. **Boundary Conditions**: Different types of barriers
   - Absorbing: walker stops upon hitting
   - Reflecting: walker bounces back
   - Periodic: walker wraps around

MATHEMATICAL FOUNDATIONS:
========================
**Difference Equations**: Random walks satisfy linear difference equations
**Generating Functions**: Powerful tool for exact calculations
**Martingale Theory**: Unbiased walks have martingale structure
**Optional Stopping**: Expected values at stopping times

ALGORITHMIC APPROACHES:
======================

1. **Iterative Methods**: Gauss-Seidel for boundary value problems
2. **Matrix Methods**: Linear algebra for finite state spaces
3. **Dynamic Programming**: Optimal stopping and control problems
4. **Simulation**: Monte Carlo for complex scenarios

CORE ALGORITHMS:
===============
```python
# Simple random walk absorption probability
def absorption_probability(start, target, barriers):
    # Solve: P(x) = 0.5 * (P(x-1) + P(x+1))
    # With boundary conditions: P(target) = 1, P(barriers) = 0
    
# Expected hitting time
def expected_hitting_time(start, targets):
    # Solve: E(x) = 1 + 0.5 * (E(x-1) + E(x+1))
    # With boundary conditions: E(targets) = 0
```

COMPLEXITY ANALYSIS:
===================
**Time**: O(n) for 1D problems, O(n^d) for d-dimensional
**Space**: O(n) for iterative methods, O(n^2) for matrix methods
**Convergence**: Geometric for most boundary value problems

APPLICATIONS:
============
- **Finance**: Asset price modeling, option pricing, risk management
- **Physics**: Diffusion, Brownian motion, statistical mechanics
- **Biology**: Population dynamics, genetic drift, foraging behavior
- **Computer Science**: Load balancing, cache performance, algorithm analysis
- **Engineering**: Signal processing, reliability analysis, queueing systems

ADVANCED TOPICS:
===============
- **Lévy Flights**: Heavy-tailed step distributions
- **Persistent Walks**: Memory and correlation effects
- **Graph Walks**: Random walks on complex networks
- **Continuous Limits**: Connection to stochastic differential equations

This framework provides the foundation for understanding
more complex stochastic processes and their applications
across diverse fields requiring probabilistic modeling.
"""
