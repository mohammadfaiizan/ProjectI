"""
Matrix Exponentiation for Dynamic Programming
Difficulty: Hard
Category: Advanced DP - Matrix Exponentiation Techniques

PROBLEM DESCRIPTION:
===================
Matrix exponentiation is a powerful technique for solving linear recurrence relations
with large parameters. This file demonstrates applications to classic DP problems:

1. Fibonacci Numbers (Large N)
2. Climbing Stairs (Large N)  
3. Tribonacci Numbers (Large N)
4. Linear Recurrence Relations
5. Graph Path Counting
6. Advanced State Transitions

Key Insight: Convert O(n) linear recurrence to O(log n) using matrix multiplication.

Example Applications:
- Fibonacci(10^18) in O(log n) time
- Count paths of length k in a graph
- Solve linear DP with huge parameters
"""


def matrix_multiply(A, B, mod=None):
    """
    MATRIX MULTIPLICATION:
    =====================
    Multiply two matrices with optional modulo.
    
    Time Complexity: O(n^3) - where n is matrix dimension
    Space Complexity: O(n^2) - result matrix
    """
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Matrix dimensions don't match for multiplication")
    
    result = [[0] * cols_B for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
                if mod:
                    result[i][j] %= mod
    
    return result


def matrix_power(matrix, power, mod=None):
    """
    MATRIX EXPONENTIATION:
    =====================
    Compute matrix^power using fast exponentiation.
    
    Time Complexity: O(n^3 * log(power)) - where n is matrix dimension
    Space Complexity: O(n^2) - matrix storage
    """
    n = len(matrix)
    
    # Initialize result as identity matrix
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        result[i][i] = 1
    
    base = [row[:] for row in matrix]  # Copy matrix
    
    while power > 0:
        if power % 2 == 1:
            result = matrix_multiply(result, base, mod)
        base = matrix_multiply(base, base, mod)
        power //= 2
    
    return result


def fibonacci_matrix_exponentiation(n, mod=None):
    """
    FIBONACCI WITH MATRIX EXPONENTIATION:
    ====================================
    Calculate F(n) using matrix exponentiation.
    
    Recurrence: F(n) = F(n-1) + F(n-2)
    Matrix form: [F(n), F(n-1)] = [F(n-1), F(n-2)] * [[1,1],[1,0]]
    
    Time Complexity: O(log n) - matrix exponentiation
    Space Complexity: O(1) - constant matrices
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    # Transformation matrix for Fibonacci
    # [F(n), F(n-1)] = [F(n-1), F(n-2)] * [[1,1],[1,0]]
    transform_matrix = [[1, 1], [1, 0]]
    
    # Compute transform_matrix^(n-1)
    result_matrix = matrix_power(transform_matrix, n - 1, mod)
    
    # Initial state: [F(1), F(0)] = [1, 0]
    # Final state: [F(n), F(n-1)] = [1, 0] * transform_matrix^(n-1)
    fibonacci_n = result_matrix[0][0] * 1 + result_matrix[0][1] * 0
    
    if mod:
        fibonacci_n %= mod
    
    return fibonacci_n


def climbing_stairs_matrix_exponentiation(n, mod=None):
    """
    CLIMBING STAIRS WITH MATRIX EXPONENTIATION:
    ==========================================
    Calculate number of ways to climb n stairs using matrix exponentiation.
    
    Same as Fibonacci: ways(n) = ways(n-1) + ways(n-2)
    
    Time Complexity: O(log n) - matrix exponentiation
    Space Complexity: O(1) - constant matrices
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    
    # This is equivalent to Fibonacci(n+1)
    return fibonacci_matrix_exponentiation(n + 1, mod)


def tribonacci_matrix_exponentiation(n, mod=None):
    """
    TRIBONACCI WITH MATRIX EXPONENTIATION:
    =====================================
    Calculate T(n) = T(n-1) + T(n-2) + T(n-3).
    
    Matrix form: [T(n), T(n-1), T(n-2)] = [T(n-1), T(n-2), T(n-3)] * transformation_matrix
    
    Time Complexity: O(log n) - matrix exponentiation
    Space Complexity: O(1) - constant matrices
    """
    if n == 0:
        return 0
    if n <= 2:
        return 1
    
    # Transformation matrix for Tribonacci
    # [T(n), T(n-1), T(n-2)] = [T(n-1), T(n-2), T(n-3)] * [[1,1,1],[1,0,0],[0,1,0]]
    transform_matrix = [
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0]
    ]
    
    # Compute transform_matrix^(n-2)
    result_matrix = matrix_power(transform_matrix, n - 2, mod)
    
    # Initial state: [T(2), T(1), T(0)] = [1, 1, 0]
    tribonacci_n = (result_matrix[0][0] * 1 + 
                   result_matrix[0][1] * 1 + 
                   result_matrix[0][2] * 0)
    
    if mod:
        tribonacci_n %= mod
    
    return tribonacci_n


def linear_recurrence_matrix_exponentiation(coefficients, initial_values, n, mod=None):
    """
    GENERAL LINEAR RECURRENCE:
    =========================
    Solve f(n) = c1*f(n-1) + c2*f(n-2) + ... + ck*f(n-k).
    
    Time Complexity: O(k^3 * log n) - where k is number of terms
    Space Complexity: O(k^2) - matrix size
    """
    k = len(coefficients)
    
    if n < k:
        return initial_values[n] if n < len(initial_values) else 0
    
    # Build transformation matrix
    # [f(n), f(n-1), ..., f(n-k+1)] = [f(n-1), f(n-2), ..., f(n-k)] * transform_matrix
    transform_matrix = [[0] * k for _ in range(k)]
    
    # First row: coefficients of the recurrence
    for i in range(k):
        transform_matrix[0][i] = coefficients[i]
    
    # Other rows: shift previous values
    for i in range(1, k):
        transform_matrix[i][i - 1] = 1
    
    # Compute transform_matrix^(n-k+1)
    result_matrix = matrix_power(transform_matrix, n - k + 1, mod)
    
    # Apply to initial state
    result = 0
    for i in range(k):
        if i < len(initial_values):
            result += result_matrix[0][k - 1 - i] * initial_values[k - 1 - i]
            if mod:
                result %= mod
    
    return result


def graph_path_counting_matrix(adjacency_matrix, start, end, path_length, mod=None):
    """
    GRAPH PATH COUNTING:
    ===================
    Count paths of exactly k length from start to end vertex.
    
    Key insight: adjacency_matrix^k gives paths of length k.
    
    Time Complexity: O(V^3 * log k) - where V is vertices, k is path length
    Space Complexity: O(V^2) - adjacency matrix
    """
    # Compute adjacency_matrix^path_length
    result_matrix = matrix_power(adjacency_matrix, path_length, mod)
    
    # Number of paths from start to end
    return result_matrix[start][end]


def advanced_state_transitions_matrix(transition_matrix, initial_state, steps, mod=None):
    """
    ADVANCED STATE TRANSITIONS:
    ===========================
    Model complex state transitions using matrix exponentiation.
    
    Applications:
    - Markov chains
    - Finite state automata
    - Complex DP state transitions
    
    Time Complexity: O(s^3 * log steps) - where s is number of states
    Space Complexity: O(s^2) - transition matrix
    """
    # Compute transition_matrix^steps
    result_matrix = matrix_power(transition_matrix, steps, mod)
    
    # Apply to initial state
    final_state = [0] * len(initial_state)
    
    for i in range(len(initial_state)):
        for j in range(len(initial_state)):
            final_state[i] += result_matrix[i][j] * initial_state[j]
            if mod:
                final_state[i] %= mod
    
    return final_state


def matrix_exponentiation_analysis():
    """
    COMPREHENSIVE MATRIX EXPONENTIATION ANALYSIS:
    ============================================
    Demonstrate various applications and optimizations.
    """
    print("Matrix Exponentiation Analysis:")
    print("=" * 50)
    
    # 1. Fibonacci comparison
    print("\n1. Fibonacci Number Comparison:")
    n = 50
    
    # Standard O(n) approach
    def fibonacci_standard(n):
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    standard_result = fibonacci_standard(n)
    matrix_result = fibonacci_matrix_exponentiation(n)
    
    print(f"   F({n}) standard: {standard_result}")
    print(f"   F({n}) matrix:   {matrix_result}")
    print(f"   Match: {'✓' if standard_result == matrix_result else '✗'}")
    
    # 2. Large Fibonacci
    print(f"\n2. Large Fibonacci (Matrix Advantage):")
    large_n = 10**6
    large_fib = fibonacci_matrix_exponentiation(large_n, mod=10**9 + 7)
    print(f"   F({large_n:,}) mod 10^9+7: {large_fib}")
    
    # 3. Climbing stairs
    print(f"\n3. Climbing Stairs:")
    stairs = 45
    stairs_result = climbing_stairs_matrix_exponentiation(stairs)
    print(f"   Ways to climb {stairs} stairs: {stairs_result:,}")
    
    # 4. Tribonacci
    print(f"\n4. Tribonacci:")
    trib_n = 20
    trib_result = tribonacci_matrix_exponentiation(trib_n)
    print(f"   T({trib_n}): {trib_result}")
    
    # 5. General linear recurrence
    print(f"\n5. General Linear Recurrence:")
    # Example: a(n) = 2*a(n-1) + 3*a(n-2) + a(n-3)
    coeffs = [2, 3, 1]  # [c1, c2, c3]
    initial = [1, 2, 4]  # [a(0), a(1), a(2)]
    rec_n = 15
    
    rec_result = linear_recurrence_matrix_exponentiation(coeffs, initial, rec_n)
    print(f"   a({rec_n}) with recurrence 2*a(n-1)+3*a(n-2)+a(n-3): {rec_result}")
    
    # 6. Graph path counting
    print(f"\n6. Graph Path Counting:")
    # Simple graph: 0->1->2->0 (cycle)
    adj_matrix = [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ]
    
    path_len = 6
    paths = graph_path_counting_matrix(adj_matrix, 0, 0, path_len)
    print(f"   Paths of length {path_len} from vertex 0 to 0: {paths}")


def matrix_exponentiation_applications():
    """
    REAL-WORLD APPLICATIONS:
    =======================
    Demonstrate practical applications of matrix exponentiation.
    """
    
    def count_binary_strings_without_consecutive_ones(n):
        """Count binary strings of length n without consecutive 1s"""
        if n <= 0:
            return 0
        if n == 1:
            return 2  # "0", "1"
        
        # State transition:
        # ending_in_0(n) = ending_in_0(n-1) + ending_in_1(n-1)
        # ending_in_1(n) = ending_in_0(n-1)
        
        # Matrix: [[1,1],[1,0]] for [ending_in_0, ending_in_1]
        transform_matrix = [[1, 1], [1, 0]]
        result_matrix = matrix_power(transform_matrix, n - 1)
        
        # Initial state: [ending_in_0(1), ending_in_1(1)] = [1, 1]
        return result_matrix[0][0] + result_matrix[0][1]
    
    def count_domino_tilings(n):
        """Count ways to tile 2×n board with 1×2 dominoes"""
        if n <= 0:
            return 0
        if n == 1:
            return 1
        
        # This follows Fibonacci pattern
        return fibonacci_matrix_exponentiation(n + 1)
    
    def count_paths_in_dag(adj_matrix, start, end, length):
        """Count paths of specific length in DAG"""
        return graph_path_counting_matrix(adj_matrix, start, end, length)
    
    def solve_markov_chain(transition_matrix, initial_distribution, steps):
        """Solve Markov chain after k steps"""
        return advanced_state_transitions_matrix(transition_matrix, initial_distribution, steps)
    
    print("Matrix Exponentiation Applications:")
    print("=" * 40)
    
    # Binary strings without consecutive 1s
    print("\n1. Binary Strings (No Consecutive 1s):")
    for n in [5, 10, 20]:
        count = count_binary_strings_without_consecutive_ones(n)
        print(f"   Length {n:2d}: {count:,} strings")
    
    # Domino tilings
    print(f"\n2. Domino Tilings (2×n board):")
    for n in [3, 5, 8, 12]:
        tilings = count_domino_tilings(n)
        print(f"   2×{n:2d} board: {tilings:,} tilings")
    
    # DAG path counting
    print(f"\n3. DAG Path Counting:")
    dag_adj = [
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ]
    
    for length in [2, 3, 4]:
        paths = count_paths_in_dag(dag_adj, 0, 3, length)
        print(f"   Paths of length {length} from 0 to 3: {paths}")
    
    # Markov chain
    print(f"\n4. Markov Chain Evolution:")
    markov_transition = [
        [0.7, 0.3],
        [0.4, 0.6]
    ]
    initial_dist = [1.0, 0.0]
    
    for steps in [1, 5, 10, 100]:
        final_dist = solve_markov_chain(markov_transition, initial_dist, steps)
        print(f"   After {steps:3d} steps: [{final_dist[0]:.3f}, {final_dist[1]:.3f}]")


# Test cases
def test_matrix_exponentiation():
    """Test matrix exponentiation implementations"""
    print("Testing Matrix Exponentiation Solutions:")
    print("=" * 70)
    
    # Fibonacci tests
    fib_tests = [(0, 0), (1, 1), (2, 1), (5, 5), (10, 55), (20, 6765)]
    
    print("Fibonacci Tests:")
    for n, expected in fib_tests:
        result = fibonacci_matrix_exponentiation(n)
        print(f"  F({n:2d}) = {result:>8} {'✓' if result == expected else '✗'}")
    
    # Climbing stairs tests
    stairs_tests = [(1, 1), (2, 2), (3, 3), (4, 5), (5, 8), (10, 89)]
    
    print(f"\nClimbing Stairs Tests:")
    for n, expected in stairs_tests:
        result = climbing_stairs_matrix_exponentiation(n)
        print(f"  Stairs({n:2d}) = {result:>8} {'✓' if result == expected else '✗'}")
    
    # Tribonacci tests
    trib_tests = [(0, 0), (1, 1), (2, 1), (3, 2), (4, 4), (5, 7), (10, 149)]
    
    print(f"\nTribonacci Tests:")
    for n, expected in trib_tests:
        result = tribonacci_matrix_exponentiation(n)
        print(f"  T({n:2d}) = {result:>8} {'✓' if result == expected else '✗'}")
    
    # Large number tests
    print(f"\nLarge Number Tests:")
    mod = 10**9 + 7
    
    large_fib = fibonacci_matrix_exponentiation(10**6, mod)
    print(f"  F(10^6) mod 10^9+7 = {large_fib}")
    
    large_stairs = climbing_stairs_matrix_exponentiation(10**5, mod)
    print(f"  Stairs(10^5) mod 10^9+7 = {large_stairs}")
    
    # Comprehensive analysis
    print(f"\n" + "=" * 70)
    print("COMPREHENSIVE ANALYSIS:")
    print("-" * 40)
    matrix_exponentiation_analysis()
    
    # Applications
    print(f"\n" + "=" * 70)
    matrix_exponentiation_applications()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. COMPLEXITY REDUCTION: From O(n) to O(log n) for linear recurrences")
    print("2. MATRIX REPRESENTATION: Convert recurrence to matrix multiplication")
    print("3. FAST EXPONENTIATION: Binary exponentiation for efficiency")
    print("4. MODULAR ARITHMETIC: Handle large numbers with modulo operations")
    print("5. GENERAL FRAMEWORK: Applicable to many linear DP problems")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Competitive Programming: Large parameter DP problems")
    print("• Graph Theory: Path counting and connectivity")
    print("• Probability Theory: Markov chain analysis")
    print("• Number Theory: Recurrence sequence computation")
    print("• Algorithm Design: Linear transformation optimization")


if __name__ == "__main__":
    test_matrix_exponentiation()


"""
MATRIX EXPONENTIATION FOR DYNAMIC PROGRAMMING - ADVANCED OPTIMIZATION:
======================================================================

Matrix exponentiation transforms linear recurrence relations from O(n)
to O(log n) complexity, enabling solution of problems with massive parameters.

KEY INSIGHTS:
============
1. **LINEAR RECURRENCE CONVERSION**: Transform DP recurrence to matrix form
2. **FAST EXPONENTIATION**: Use binary exponentiation for matrix powers
3. **STATE VECTOR REPRESENTATION**: Encode DP states as vectors
4. **TRANSFORMATION MATRIX**: Capture state transitions as matrix multiplication
5. **MODULAR ARITHMETIC**: Handle large numbers with modulo operations

FUNDAMENTAL TECHNIQUE:
=====================
**Recurrence**: f(n) = c₁f(n-1) + c₂f(n-2) + ... + cₖf(n-k)

**Matrix Form**:
[f(n), f(n-1), ..., f(n-k+1)] = [f(n-1), f(n-2), ..., f(n-k)] × T

**Transformation Matrix T**:
```
T = [[c₁, c₂, ..., cₖ],
     [ 1,  0, ...,  0],
     [ 0,  1, ...,  0],
     ...
     [ 0,  0, ...,  0]]
```

**Solution**: f(n) = initial_state × T^(n-k+1)

CORE ALGORITHMS:
===============

**Matrix Multiplication**: O(k³) for k×k matrices
```python
def matrix_multiply(A, B, mod=None):
    result = [[0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
                if mod: result[i][j] %= mod
    return result
```

**Matrix Exponentiation**: O(k³ log n)
```python
def matrix_power(matrix, power, mod=None):
    n = len(matrix)
    result = identity_matrix(n)
    base = copy(matrix)
    
    while power > 0:
        if power % 2 == 1:
            result = matrix_multiply(result, base, mod)
        base = matrix_multiply(base, base, mod)
        power //= 2
    
    return result
```

CLASSIC APPLICATIONS:
====================

**1. Fibonacci Sequence**:
- Recurrence: F(n) = F(n-1) + F(n-2)
- Matrix: [[1,1], [1,0]]
- Complexity: O(log n) vs O(n)

**2. Climbing Stairs**:
- Same as Fibonacci with shifted indices
- ways(n) = ways(n-1) + ways(n-2)

**3. Tribonacci**:
- Recurrence: T(n) = T(n-1) + T(n-2) + T(n-3)
- Matrix: [[1,1,1], [1,0,0], [0,1,0]]

**4. Graph Path Counting**:
- adjacency_matrix^k gives paths of length k
- Enables efficient long-path counting

ADVANCED APPLICATIONS:
=====================

**Markov Chains**: Compute state distribution after k steps
**Linear DP**: Any recurrence with constant coefficients
**Automata**: Count accepted strings of specific length
**Combinatorics**: Generating function applications

OPTIMIZATION TECHNIQUES:
=======================

**Space Optimization**: 
- Only store current and previous states during exponentiation
- Use iterative rather than recursive matrix multiplication

**Modular Arithmetic**:
- Apply modulo at each multiplication to prevent overflow
- Use properties: (a+b) mod m = ((a mod m) + (b mod m)) mod m

**Numerical Stability**:
- Use integer arithmetic when possible
- Handle edge cases (n=0, n=1) separately

COMPLEXITY ANALYSIS:
===================
**Time**: O(k³ log n) where k is recurrence order, n is parameter
**Space**: O(k²) for matrix storage

**Comparison with Standard DP**:
- Standard: O(n) time, O(1) space
- Matrix: O(k³ log n) time, O(k²) space
- Advantage: Massive n values (10¹⁸+)

IMPLEMENTATION CONSIDERATIONS:
=============================

**Matrix Size**: Minimize by finding minimal recurrence order
**Modulo Operations**: Apply consistently to avoid overflow
**Base Cases**: Handle small n values separately
**Precision**: Use appropriate data types for problem requirements

REAL-WORLD APPLICATIONS:
=======================
- **Competitive Programming**: Problems with n ≤ 10¹⁸
- **Scientific Computing**: Long-term system evolution
- **Cryptography**: Sequence generation and period analysis
- **Economics**: Multi-period financial modeling
- **Physics**: Quantum state evolution

This technique represents one of the most powerful
optimizations in dynamic programming, enabling
solutions to problems previously considered intractable
due to parameter size constraints.
"""
