"""
LeetCode 264: Ugly Number II
Difficulty: Medium
Category: Fibonacci & Linear DP

PROBLEM DESCRIPTION:
===================
An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5.

Given an integer n, return the nth ugly number.

Example 1:
Input: n = 10
Output: 12
Explanation: [1, 2, 3, 4, 5, 6, 8, 9, 10, 12] is the sequence of the first 10 ugly numbers.

Example 2:
Input: n = 1
Output: 1
Explanation: 1 has no prime factors, therefore all of its prime factors are limited to 2, 3, and 5.

Constraints:
- 1 <= n <= 1690
"""

def nth_ugly_number_bruteforce(n):
    """
    BRUTE FORCE APPROACH:
    ====================
    Check each number to see if it's ugly until we find the nth one.
    
    Time Complexity: O(n * log(max_ugly)) - check each number
    Space Complexity: O(1) - constant space
    """
    def is_ugly(num):
        """Check if a number is ugly (only factors 2, 3, 5)"""
        if num <= 0:
            return False
        
        # Remove all factors of 2, 3, and 5
        for factor in [2, 3, 5]:
            while num % factor == 0:
                num //= factor
        
        return num == 1
    
    count = 0
    current = 1
    
    while count < n:
        if is_ugly(current):
            count += 1
            if count == n:
                return current
        current += 1
    
    return current - 1


def nth_ugly_number_dp_three_pointers(n):
    """
    DYNAMIC PROGRAMMING - THREE POINTERS:
    ====================================
    Generate ugly numbers in order using three pointers.
    
    Time Complexity: O(n) - generate n ugly numbers
    Space Complexity: O(n) - store all ugly numbers
    """
    if n <= 0:
        return 0
    
    ugly = [0] * n
    ugly[0] = 1
    
    # Three pointers for multiples of 2, 3, and 5
    i2 = i3 = i5 = 0
    
    # Next multiples
    next_2 = 2
    next_3 = 3
    next_5 = 5
    
    for i in range(1, n):
        # Choose the minimum of next multiples
        next_ugly = min(next_2, next_3, next_5)
        ugly[i] = next_ugly
        
        # Move pointers that generated this ugly number
        if next_ugly == next_2:
            i2 += 1
            next_2 = ugly[i2] * 2
        
        if next_ugly == next_3:
            i3 += 1
            next_3 = ugly[i3] * 3
        
        if next_ugly == next_5:
            i5 += 1
            next_5 = ugly[i5] * 5
    
    return ugly[n - 1]


def nth_ugly_number_priority_queue(n):
    """
    PRIORITY QUEUE (MIN HEAP) APPROACH:
    ==================================
    Use min heap to generate ugly numbers in sorted order.
    
    Time Complexity: O(n log n) - n insertions/extractions from heap
    Space Complexity: O(n) - heap storage
    """
    import heapq
    
    if n <= 0:
        return 0
    
    heap = [1]
    seen = {1}
    
    for i in range(n):
        ugly = heapq.heappop(heap)
        
        if i == n - 1:
            return ugly
        
        # Generate next ugly numbers
        for factor in [2, 3, 5]:
            new_ugly = ugly * factor
            if new_ugly not in seen:
                seen.add(new_ugly)
                heapq.heappush(heap, new_ugly)
    
    return ugly


def nth_ugly_number_optimized_dp(n):
    """
    OPTIMIZED DP WITH BETTER SPACE:
    ==============================
    Optimized version of three pointers approach.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(n) - store ugly numbers
    """
    if n == 1:
        return 1
    
    ugly = [1] * n
    
    # Pointers for 2, 3, 5 multiples
    p2 = p3 = p5 = 0
    
    for i in range(1, n):
        # Calculate next candidates
        candidate_2 = ugly[p2] * 2
        candidate_3 = ugly[p3] * 3
        candidate_5 = ugly[p5] * 5
        
        # Choose minimum
        next_ugly = min(candidate_2, candidate_3, candidate_5)
        ugly[i] = next_ugly
        
        # Update pointers (handle duplicates by updating all matching)
        if next_ugly == candidate_2:
            p2 += 1
        if next_ugly == candidate_3:
            p3 += 1
        if next_ugly == candidate_5:
            p5 += 1
    
    return ugly[n - 1]


def nth_ugly_number_set_based(n):
    """
    SET-BASED APPROACH:
    ==================
    Use set to generate ugly numbers and avoid duplicates.
    
    Time Complexity: O(n log n) - sorting operations
    Space Complexity: O(n) - set storage
    """
    if n == 1:
        return 1
    
    ugly_set = {1}
    ugly_list = [1]
    
    i = 0
    while len(ugly_list) < n:
        current = ugly_list[i]
        
        # Generate next ugly numbers
        for factor in [2, 3, 5]:
            new_ugly = current * factor
            if new_ugly not in ugly_set:
                ugly_set.add(new_ugly)
                ugly_list.append(new_ugly)
        
        i += 1
        
        # Keep list sorted (expensive but works)
        ugly_list.sort()
    
    return ugly_list[n - 1]


def nth_ugly_number_mathematical(n):
    """
    MATHEMATICAL APPROACH:
    =====================
    Use mathematical properties to generate ugly numbers efficiently.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(n) - store sequence
    """
    if n == 1:
        return 1
    
    # More efficient implementation of three pointers
    ugly = [1] * n
    
    # Initialize pointers and next values
    i2 = i3 = i5 = 0
    
    for i in range(1, n):
        # Calculate next ugly number
        ugly2 = ugly[i2] * 2
        ugly3 = ugly[i3] * 3
        ugly5 = ugly[i5] * 5
        
        min_ugly = min(ugly2, ugly3, ugly5)
        ugly[i] = min_ugly
        
        # Advance pointers
        if min_ugly == ugly2:
            i2 += 1
        if min_ugly == ugly3:
            i3 += 1
        if min_ugly == ugly5:
            i5 += 1
    
    return ugly[n - 1]


def nth_ugly_number_with_sequence(n):
    """
    GENERATE COMPLETE SEQUENCE:
    ==========================
    Return both nth ugly number and the complete sequence.
    
    Time Complexity: O(n) - generate sequence
    Space Complexity: O(n) - store sequence
    """
    if n <= 0:
        return 0, []
    
    ugly = [1]
    i2 = i3 = i5 = 0
    
    while len(ugly) < n:
        # Next candidates
        next_2 = ugly[i2] * 2
        next_3 = ugly[i3] * 3
        next_5 = ugly[i5] * 5
        
        # Choose minimum
        next_ugly = min(next_2, next_3, next_5)
        ugly.append(next_ugly)
        
        # Update pointers
        if next_ugly == next_2:
            i2 += 1
        if next_ugly == next_3:
            i3 += 1
        if next_ugly == next_5:
            i5 += 1
    
    return ugly[n - 1], ugly


def nth_ugly_number_space_optimized(n):
    """
    SPACE OPTIMIZED VERSION:
    ========================
    Generate ugly numbers without storing entire sequence.
    Note: This is challenging since we need previous ugly numbers.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(n) - still need to store sequence for this problem
    """
    # For this specific problem, we need the sequence
    # But we can optimize by not storing extra data structures
    
    if n == 1:
        return 1
    
    ugly = [0] * n
    ugly[0] = 1
    
    i2 = i3 = i5 = 0
    
    for i in range(1, n):
        # Direct calculation without extra variables
        ugly[i] = min(ugly[i2] * 2, ugly[i3] * 3, ugly[i5] * 5)
        
        # Update pointers
        if ugly[i] == ugly[i2] * 2:
            i2 += 1
        if ugly[i] == ugly[i3] * 3:
            i3 += 1
        if ugly[i] == ugly[i5] * 5:
            i5 += 1
    
    return ugly[n - 1]


def analyze_ugly_number_pattern(max_n):
    """
    PATTERN ANALYSIS:
    ================
    Analyze the pattern and properties of ugly numbers.
    
    Time Complexity: O(max_n) - generate and analyze
    Space Complexity: O(max_n) - store for analysis
    """
    _, sequence = nth_ugly_number_with_sequence(max_n)
    
    print(f"First {max_n} ugly numbers:")
    print(sequence)
    
    print(f"\nFactorization analysis:")
    for i, ugly in enumerate(sequence[:min(20, len(sequence))]):
        factors = []
        temp = ugly
        
        for prime in [2, 3, 5]:
            while temp % prime == 0:
                factors.append(prime)
                temp //= prime
        
        print(f"U({i+1}) = {ugly} = {' × '.join(map(str, factors)) if factors else '1'}")
    
    # Growth rate analysis
    if len(sequence) > 1:
        print(f"\nGrowth rate analysis:")
        for i in range(1, min(10, len(sequence))):
            ratio = sequence[i] / sequence[i-1]
            print(f"U({i+1})/U({i}) = {sequence[i]}/{sequence[i-1]} = {ratio:.3f}")
    
    return sequence


# Test cases
def test_nth_ugly_number():
    """Test all implementations with various inputs"""
    test_cases = [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 12),
        (15, 24),
        (20, 36),
        (25, 45),
        (100, 1536),
        (150, 5832)
    ]
    
    print("Testing Ugly Number II Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: n = {n}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large n)
        if n <= 20:
            brute = nth_ugly_number_bruteforce(n)
            print(f"Brute Force:      {brute:>6} {'✓' if brute == expected else '✗'}")
        
        three_ptr = nth_ugly_number_dp_three_pointers(n)
        pq = nth_ugly_number_priority_queue(n)
        opt_dp = nth_ugly_number_optimized_dp(n)
        math_approach = nth_ugly_number_mathematical(n)
        space_opt = nth_ugly_number_space_optimized(n)
        
        print(f"Three Pointers:   {three_ptr:>6} {'✓' if three_ptr == expected else '✗'}")
        print(f"Priority Queue:   {pq:>6} {'✓' if pq == expected else '✗'}")
        print(f"Optimized DP:     {opt_dp:>6} {'✓' if opt_dp == expected else '✗'}")
        print(f"Mathematical:     {math_approach:>6} {'✓' if math_approach == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>6} {'✓' if space_opt == expected else '✗'}")
    
    # Show sequence analysis
    print(f"\nUgly Number Sequence Analysis:")
    analyze_ugly_number_pattern(20)
    
    # Show complete sequence for small n
    print(f"\nComplete sequences:")
    for n in [5, 10, 15]:
        nth_val, seq = nth_ugly_number_with_sequence(n)
        print(f"First {n} ugly numbers: {seq}")
        print(f"U({n}) = {nth_val}")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(n*log(max)), Space: O(1)")
    print("Three Pointers:   Time: O(n),          Space: O(n)")
    print("Priority Queue:   Time: O(n log n),    Space: O(n)")
    print("Optimized DP:     Time: O(n),          Space: O(n)")
    print("Mathematical:     Time: O(n),          Space: O(n)")
    print("Space Optimized:  Time: O(n),          Space: O(n)")


if __name__ == "__main__":
    test_nth_ugly_number()


"""
PATTERN RECOGNITION:
==================
This is a sequence generation DP problem:
- Generate numbers with specific properties (only factors 2, 3, 5)
- Maintain sorted order efficiently
- Use multiple pointers to merge sorted sequences
- Classic example of "three pointers" technique

KEY INSIGHT - THREE POINTERS TECHNIQUE:
======================================
1. Ugly numbers = {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, ...}
2. Each ugly number (except 1) = previous ugly number × 2, 3, or 5
3. Use three pointers to track next multiples of 2, 3, 5
4. Always choose minimum to maintain sorted order

ALGORITHM EXPLANATION:
=====================
1. Start with ugly = [1]
2. Maintain pointers i2, i3, i5 (initially 0)
3. Next candidates: ugly[i2]×2, ugly[i3]×3, ugly[i5]×5
4. Choose minimum as next ugly number
5. Advance corresponding pointer(s)
6. Repeat until we have n ugly numbers

STATE DEFINITION:
================
ugly[i] = ith ugly number in the sequence
Pointers track where we are in generating multiples

RECURRENCE RELATION:
===================
ugly[i] = min(ugly[i2]×2, ugly[i3]×3, ugly[i5]×5)
Update pointers based on which value was chosen

OPTIMIZATION TECHNIQUES:
=======================
1. Three pointers: Avoid sorting, maintain order naturally
2. Duplicate handling: Update all matching pointers simultaneously
3. Priority queue: Alternative approach with heap
4. Set-based: Avoid duplicates using hash set

ALTERNATIVE APPROACHES:
======================
1. Brute force: Check each number for ugliness
2. Priority queue: Use min heap for sorted generation
3. Set-based: Generate and sort dynamically
4. Mathematical: Direct formula (complex for this problem)

VARIANTS TO PRACTICE:
====================
- Ugly Number (263) - check if single number is ugly
- Super Ugly Numbers (313) - generalize to any set of prime factors
- Hamming Numbers - same as ugly numbers
- Perfect Squares (279) - similar sequence generation

INTERVIEW TIPS:
==============
1. Start with brute force approach
2. Identify inefficiency: checking every number
3. Realize we can generate ugly numbers directly
4. Explain three pointers technique clearly
5. Handle duplicate elimination correctly
6. Discuss time/space trade-offs
7. Mention alternative approaches (heap, set)
8. Test with small examples to verify logic
"""
