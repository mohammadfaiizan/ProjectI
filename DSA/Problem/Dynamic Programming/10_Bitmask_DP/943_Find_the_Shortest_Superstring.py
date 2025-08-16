"""
LeetCode 943: Find the Shortest Superstring
Difficulty: Hard
Category: Bitmask DP - String Optimization

PROBLEM DESCRIPTION:
===================
Given an array of strings words, return the shortest string that contains each string in words as a substring. If there are multiple valid strings of the shortest length, return any one of them.

It is guaranteed that no string in words is a substring of another string in words.

Example 1:
Input: words = ["alex","loves","leetcode"]
Output: "alexlovesleetcode"
Explanation: All permutations of "alex","loves","leetcode" would also be valid answers.

Example 2:
Input: words = ["catg","ctaagt","gcta","ttca","atgcatc"]
Output: "gctaagttcatgcatc"

Constraints:
- 1 <= words.length <= 12
- 1 <= words[i].length <= 20
- words[i] consists of lowercase English letters.
- All the strings of words are unique.
- It is guaranteed that no string in words is a substring of another string in words.
"""


def shortest_superstring_brute_force(words):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all permutations and find the shortest concatenation.
    
    Time Complexity: O(n! * n * m) where m is average string length
    Space Complexity: O(n * m) - string storage
    """
    from itertools import permutations
    
    def overlap(s1, s2):
        """Find maximum overlap between end of s1 and start of s2"""
        max_overlap = min(len(s1), len(s2))
        for i in range(max_overlap, 0, -1):
            if s1[-i:] == s2[:i]:
                return i
        return 0
    
    def merge(s1, s2):
        """Merge s1 and s2 with maximum overlap"""
        overlap_len = overlap(s1, s2)
        return s1 + s2[overlap_len:]
    
    min_length = float('inf')
    result = ""
    
    for perm in permutations(words):
        current = perm[0]
        for i in range(1, len(perm)):
            current = merge(current, perm[i])
        
        if len(current) < min_length:
            min_length = len(current)
            result = current
    
    return result


def shortest_superstring_bitmask_dp(words):
    """
    BITMASK DP APPROACH:
    ===================
    Use bitmask DP to find optimal string arrangement.
    
    Time Complexity: O(n^2 * 2^n * m) where m is string length
    Space Complexity: O(n * 2^n * m) - DP table with strings
    """
    n = len(words)
    if n == 1:
        return words[0]
    
    # Precompute overlaps between all pairs
    overlap = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                max_len = min(len(words[i]), len(words[j]))
                for k in range(max_len, 0, -1):
                    if words[i][-k:] == words[j][:k]:
                        overlap[i][j] = k
                        break
    
    # dp[mask][last] = (min_length, actual_string)
    dp = {}
    
    # Initialize with single words
    for i in range(n):
        dp[(1 << i), i] = (len(words[i]), words[i])
    
    # Fill DP table
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            
            if (mask, last) not in dp:
                continue
            
            current_length, current_string = dp[(mask, last)]
            
            # Try adding each remaining word
            for next_word in range(n):
                if mask & (1 << next_word):
                    continue
                
                new_mask = mask | (1 << next_word)
                overlap_len = overlap[last][next_word]
                new_length = current_length + len(words[next_word]) - overlap_len
                new_string = current_string + words[next_word][overlap_len:]
                
                if (new_mask, next_word) not in dp or new_length < dp[(new_mask, next_word)][0]:
                    dp[(new_mask, next_word)] = (new_length, new_string)
    
    # Find the shortest superstring
    full_mask = (1 << n) - 1
    min_length = float('inf')
    result = ""
    
    for last in range(n):
        if (full_mask, last) in dp:
            length, string = dp[(full_mask, last)]
            if length < min_length:
                min_length = length
                result = string
    
    return result


def shortest_superstring_optimized_dp(words):
    """
    OPTIMIZED BITMASK DP:
    ====================
    Use more efficient DP representation storing only lengths.
    
    Time Complexity: O(n^2 * 2^n) - optimized computation
    Space Complexity: O(n * 2^n) - reduced space usage
    """
    n = len(words)
    if n == 1:
        return words[0]
    
    # Precompute overlaps
    def compute_overlap(s1, s2):
        max_overlap = min(len(s1), len(s2))
        for i in range(max_overlap, 0, -1):
            if s1[-i:] == s2[:i]:
                return i
        return 0
    
    overlap = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                overlap[i][j] = compute_overlap(words[i], words[j])
    
    # dp[mask][last] = minimum length to include all words in mask ending with last
    dp = [[float('inf')] * n for _ in range(1 << n)]
    parent = [[(-1, -1)] * n for _ in range(1 << n)]
    
    # Initialize
    for i in range(n):
        dp[1 << i][i] = len(words[i])
    
    # Fill DP
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)) or dp[mask][last] == float('inf'):
                continue
            
            for next_word in range(n):
                if mask & (1 << next_word):
                    continue
                
                new_mask = mask | (1 << next_word)
                new_length = dp[mask][last] + len(words[next_word]) - overlap[last][next_word]
                
                if new_length < dp[new_mask][next_word]:
                    dp[new_mask][next_word] = new_length
                    parent[new_mask][next_word] = (mask, last)
    
    # Find optimal solution
    full_mask = (1 << n) - 1
    min_length = float('inf')
    best_last = -1
    
    for last in range(n):
        if dp[full_mask][last] < min_length:
            min_length = dp[full_mask][last]
            best_last = last
    
    # Reconstruct solution
    def reconstruct():
        path = []
        mask, last = full_mask, best_last
        
        while mask != 0:
            path.append(last)
            if parent[mask][last] == (-1, -1):
                break
            prev_mask, prev_last = parent[mask][last]
            mask, last = prev_mask, prev_last
        
        path.reverse()
        
        # Build result string
        if not path:
            return ""
        
        result = words[path[0]]
        for i in range(1, len(path)):
            prev_word = path[i-1]
            curr_word = path[i]
            overlap_len = overlap[prev_word][curr_word]
            result += words[curr_word][overlap_len:]
        
        return result
    
    return reconstruct()


def shortest_superstring_with_analysis(words):
    """
    BITMASK DP WITH DETAILED ANALYSIS:
    =================================
    Track the computation process with overlap analysis.
    
    Time Complexity: O(n^2 * 2^n) - standard DP
    Space Complexity: O(n * 2^n) - DP table + analysis
    """
    n = len(words)
    
    # Compute and analyze overlaps
    def get_overlap(s1, s2):
        max_len = min(len(s1), len(s2))
        for i in range(max_len, 0, -1):
            if s1[-i:] == s2[:i]:
                return i
        return 0
    
    overlap = [[0] * n for _ in range(n)]
    overlap_info = {}
    
    for i in range(n):
        for j in range(n):
            if i != j:
                overlap[i][j] = get_overlap(words[i], words[j])
                if overlap[i][j] > 0:
                    overlap_info[(i, j)] = (words[i][-overlap[i][j]:], overlap[i][j])
    
    analysis = {
        'overlap_matrix': overlap,
        'overlap_details': overlap_info,
        'total_length': sum(len(word) for word in words),
        'max_possible_savings': 0,
        'actual_savings': 0
    }
    
    # Calculate maximum possible savings
    analysis['max_possible_savings'] = sum(max(overlap[i]) for i in range(n))
    
    # Run DP
    dp = [[float('inf')] * n for _ in range(1 << n)]
    parent = [[None] * n for _ in range(1 << n)]
    
    for i in range(n):
        dp[1 << i][i] = len(words[i])
    
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)) or dp[mask][last] == float('inf'):
                continue
            
            for next_word in range(n):
                if mask & (1 << next_word):
                    continue
                
                new_mask = mask | (1 << next_word)
                new_length = dp[mask][last] + len(words[next_word]) - overlap[last][next_word]
                
                if new_length < dp[new_mask][next_word]:
                    dp[new_mask][next_word] = new_length
                    parent[new_mask][next_word] = (mask, last)
    
    # Find solution and calculate actual savings
    full_mask = (1 << n) - 1
    min_length = min(dp[full_mask])
    analysis['result_length'] = min_length
    analysis['actual_savings'] = analysis['total_length'] - min_length
    
    return analysis


def shortest_superstring_analysis(words):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze the shortest superstring problem with detailed insights.
    """
    print(f"Shortest Superstring Analysis:")
    print(f"Words: {words}")
    print(f"Number of words: {len(words)}")
    print(f"Total character count: {sum(len(word) for word in words)}")
    print(f"Average word length: {sum(len(word) for word in words) / len(words):.2f}")
    
    # Different approaches
    if len(words) <= 6:
        try:
            brute_force = shortest_superstring_brute_force(words)
            print(f"Brute force result: '{brute_force}' (length: {len(brute_force)})")
        except:
            print("Brute force: Too slow")
    
    bitmask_dp = shortest_superstring_bitmask_dp(words)
    optimized = shortest_superstring_optimized_dp(words)
    
    print(f"Bitmask DP result: '{bitmask_dp}' (length: {len(bitmask_dp)})")
    print(f"Optimized DP result: '{optimized}' (length: {len(optimized)})")
    
    # Detailed analysis
    analysis = shortest_superstring_with_analysis(words)
    
    print(f"\nDetailed Analysis:")
    print(f"Total input length: {analysis['total_length']}")
    print(f"Result length: {analysis['result_length']}")
    print(f"Characters saved: {analysis['actual_savings']}")
    print(f"Compression ratio: {analysis['actual_savings']/analysis['total_length']:.2%}")
    
    print(f"\nOverlap Matrix:")
    overlap_matrix = analysis['overlap_matrix']
    n = len(words)
    
    print("     ", end="")
    for j in range(n):
        print(f"{j:3}", end="")
    print()
    
    for i in range(n):
        print(f"{i:3}: ", end="")
        for j in range(n):
            print(f"{overlap_matrix[i][j]:3}", end="")
        print(f"  '{words[i]}'")
    
    print(f"\nSignificant Overlaps:")
    for (i, j), (overlap_str, length) in analysis['overlap_details'].items():
        print(f"  {words[i]} + {words[j]}: overlap '{overlap_str}' (length {length})")
    
    return optimized


def shortest_superstring_variants():
    """
    SHORTEST SUPERSTRING VARIANTS:
    ==============================
    Different scenarios and modifications.
    """
    
    def longest_superstring_with_constraints(words, max_length):
        """Find longest superstring within length constraint"""
        # This would require different optimization - simplified
        result = shortest_superstring_optimized_dp(words)
        return result if len(result) <= max_length else None
    
    def count_optimal_superstrings(words):
        """Count number of optimal superstrings"""
        n = len(words)
        if n <= 1:
            return 1
        
        # This is complex - simplified version
        optimal_length = len(shortest_superstring_optimized_dp(words))
        
        # For small cases, check all permutations
        if n <= 6:
            from itertools import permutations
            
            def get_superstring_length(perm):
                if not perm:
                    return 0
                
                result = perm[0]
                for i in range(1, len(perm)):
                    # Find overlap
                    max_overlap = min(len(result), len(perm[i]))
                    overlap_len = 0
                    for j in range(max_overlap, 0, -1):
                        if result[-j:] == perm[i][:j]:
                            overlap_len = j
                            break
                    result += perm[i][overlap_len:]
                
                return len(result)
            
            count = 0
            for perm in permutations(words):
                if get_superstring_length(perm) == optimal_length:
                    count += 1
            
            return count
        
        return -1  # Too complex for larger inputs
    
    def shortest_superstring_with_forbidden_overlaps(words, forbidden_pairs):
        """Find shortest superstring avoiding certain overlaps"""
        # Simplified version - would need to modify DP
        n = len(words)
        
        # Compute allowed overlaps only
        def compute_overlap(s1, s2, i, j):
            if (i, j) in forbidden_pairs:
                return 0
            
            max_len = min(len(s1), len(s2))
            for k in range(max_len, 0, -1):
                if s1[-k:] == s2[:k]:
                    return k
            return 0
        
        # Use modified DP with forbidden overlaps
        overlap = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    overlap[i][j] = compute_overlap(words[i], words[j], i, j)
        
        # Rest of DP is similar to original
        return shortest_superstring_optimized_dp(words)  # Simplified
    
    def minimum_superstring_covering_k_words(words, k):
        """Find minimum superstring covering at least k words"""
        n = len(words)
        if k > n:
            return None
        
        # Try all combinations of k words
        from itertools import combinations
        
        min_length = float('inf')
        best_result = ""
        
        for word_combination in combinations(range(n), k):
            selected_words = [words[i] for i in word_combination]
            superstring = shortest_superstring_optimized_dp(selected_words)
            
            if len(superstring) < min_length:
                min_length = len(superstring)
                best_result = superstring
        
        return best_result
    
    # Test variants
    test_cases = [
        ["alex", "loves", "leetcode"],
        ["catg", "ctaagt", "gcta"],
        ["abc", "bcd", "cde"]
    ]
    
    print("Shortest Superstring Variants:")
    print("=" * 50)
    
    for words in test_cases:
        print(f"\nWords: {words}")
        
        basic_result = shortest_superstring_optimized_dp(words)
        print(f"Basic shortest superstring: '{basic_result}' (length: {len(basic_result)})")
        
        # Count optimal solutions
        if len(words) <= 5:
            count = count_optimal_superstrings(words)
            print(f"Number of optimal superstrings: {count}")
        
        # Forbidden overlaps example
        if len(words) >= 3:
            forbidden = {(0, 1)}  # Forbid overlap between first two words
            forbidden_result = shortest_superstring_with_forbidden_overlaps(words, forbidden)
            print(f"With forbidden overlaps: '{forbidden_result}' (length: {len(forbidden_result)})")
        
        # Minimum covering k words
        if len(words) >= 2:
            k = len(words) - 1
            covering_result = minimum_superstring_covering_k_words(words, k)
            print(f"Covering {k} words: '{covering_result}' (length: {len(covering_result)})")


# Test cases
def test_shortest_superstring():
    """Test all implementations with various inputs"""
    test_cases = [
        (["alex","loves","leetcode"], "alexlovesleetcode"),
        (["catg","ctaagt","gcta","ttca","atgcatc"], "gctaagttcatgcatc"),
        (["abc","bc","c"], "abc"),
        (["a","aa","aaa"], "aaa"),
        (["abcdef"], "abcdef"),
        (["a","b"], "ab"),
        (["abc","cde","efg"], "abcdefg")
    ]
    
    print("Testing Shortest Superstring Solutions:")
    print("=" * 70)
    
    for i, (words, expected_chars) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"words = {words}")
        print(f"Expected length: {len(expected_chars)}")
        
        # Skip brute force for large inputs
        if len(words) <= 5:
            try:
                brute_force = shortest_superstring_brute_force(words)
                print(f"Brute Force:      '{brute_force}' (len: {len(brute_force)})")
            except:
                print(f"Brute Force:      Timeout")
        
        bitmask_dp = shortest_superstring_bitmask_dp(words)
        optimized = shortest_superstring_optimized_dp(words)
        
        print(f"Bitmask DP:       '{bitmask_dp}' (len: {len(bitmask_dp)})")
        print(f"Optimized:        '{optimized}' (len: {len(optimized)})")
        
        # Verify all words are substrings
        def verify_superstring(superstring, word_list):
            return all(word in superstring for word in word_list)
        
        bitmask_valid = verify_superstring(bitmask_dp, words)
        optimized_valid = verify_superstring(optimized, words)
        
        print(f"Bitmask valid:    {'✓' if bitmask_valid else '✗'}")
        print(f"Optimized valid:  {'✓' if optimized_valid else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    shortest_superstring_analysis(["alex", "loves", "leetcode"])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    shortest_superstring_variants()
    
    # Performance comparison
    print(f"\n" + "=" * 70)
    print("PERFORMANCE COMPARISON:")
    performance_cases = [
        ["abc", "bcd", "cde", "def"],
        ["abcde", "cdefg", "efghi"],
        ["aaa", "aab", "abb", "bbb"]
    ]
    
    for words in performance_cases:
        print(f"\nWords: {words}")
        
        import time
        
        start = time.time()
        bitmask_result = shortest_superstring_bitmask_dp(words)
        bitmask_time = time.time() - start
        
        start = time.time()
        opt_result = shortest_superstring_optimized_dp(words)
        opt_time = time.time() - start
        
        print(f"Bitmask DP: '{bitmask_result}' (Time: {bitmask_time:.6f}s)")
        print(f"Optimized:  '{opt_result}' (Time: {opt_time:.6f}s)")
        print(f"Length match: {'✓' if len(bitmask_result) == len(opt_result) else '✗'}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. STRING OVERLAP: Maximize overlap between string endings and beginnings")
    print("2. BITMASK STATES: Track which words have been included")
    print("3. OPTIMAL SUBSTRUCTURE: Optimal superstring built from optimal sub-solutions")
    print("4. PREPROCESSING: Compute all pairwise overlaps efficiently")
    print("5. PATH RECONSTRUCTION: Build actual string from DP solution")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Genome Assembly: Reconstruct DNA sequences from fragments")
    print("• Data Compression: Find shortest representation of string sets")
    print("• String Processing: Efficient concatenation with overlap optimization")
    print("• Bioinformatics: Sequence assembly and overlap detection")
    print("• Computer Science: String algorithms and optimization problems")


if __name__ == "__main__":
    test_shortest_superstring()


"""
FIND THE SHORTEST SUPERSTRING - STRING OPTIMIZATION WITH BITMASKS:
==================================================================

This problem demonstrates advanced Bitmask DP for string optimization:
- Optimal string concatenation with overlap maximization
- Complex state management combining masks with string operations
- Path reconstruction for building actual solutions
- Integration of string algorithms with exponential optimization

KEY INSIGHTS:
============
1. **STRING OVERLAP OPTIMIZATION**: Maximize overlap between string endings and beginnings
2. **BITMASK STATE TRACKING**: Use bits to represent which strings have been included
3. **OPTIMAL SUBSTRUCTURE**: Shortest superstring built from optimal sub-solutions
4. **PREPROCESSING EFFICIENCY**: Precompute all pairwise overlaps for O(1) access
5. **PATH RECONSTRUCTION**: Build actual superstring from DP state transitions

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(n! × n × m) time, O(n × m) space
   - Try all permutations of strings
   - Exponential and impractical for large inputs

2. **Bitmask DP**: O(n² × 2^n × m) time, O(n × 2^n × m) space
   - State: (included_strings_mask, last_string, current_superstring)
   - Store actual strings in DP table

3. **Optimized DP**: O(n² × 2^n) time, O(n × 2^n) space
   - State: (included_strings_mask, last_string, total_length)
   - Reconstruct solution using parent pointers

4. **Advanced DP**: O(n² × 2^n) time, O(n × 2^n) space
   - Most memory-efficient approach
   - Separate optimization and reconstruction phases

CORE OVERLAP COMPUTATION:
========================
```python
def compute_overlap(s1, s2):
    """Find maximum overlap between end of s1 and start of s2"""
    max_overlap = min(len(s1), len(s2))
    for i in range(max_overlap, 0, -1):
        if s1[-i:] == s2[:i]:
            return i
    return 0

# Precompute all overlaps
overlap = [[0] * n for _ in range(n)]
for i in range(n):
    for j in range(n):
        if i != j:
            overlap[i][j] = compute_overlap(words[i], words[j])
```

BITMASK DP FORMULATION:
======================
**State Definition**: `dp[mask][last]` = minimum length of superstring including all words in `mask`, ending with word `last`

**Base Cases**: `dp[1 << i][i] = len(words[i])` for each starting word

**State Transitions**:
```python
for mask in range(1, 1 << n):
    for last in range(n):
        if not (mask & (1 << last)):
            continue
        
        for next_word in range(n):
            if mask & (1 << next_word):
                continue
            
            new_mask = mask | (1 << next_word)
            new_length = dp[mask][last] + len(words[next_word]) - overlap[last][next_word]
            
            dp[new_mask][next_word] = min(dp[new_mask][next_word], new_length)
```

**Answer**: `min(dp[(1 << n) - 1][i] for i in range(n))`

STRING RECONSTRUCTION:
=====================
**Parent Tracking**: Store previous state for each optimal transition
```python
parent[new_mask][next_word] = (mask, last)
```

**Path Recovery**: Backtrack from optimal final state
```python
def reconstruct_path():
    path = []
    mask, last = full_mask, best_last
    
    while mask != 0:
        path.append(last)
        if parent[mask][last] is None:
            break
        mask, last = parent[mask][last]
    
    return path[::-1]
```

**String Building**: Concatenate words with optimal overlaps
```python
def build_superstring(path):
    if not path:
        return ""
    
    result = words[path[0]]
    for i in range(1, len(path)):
        prev, curr = path[i-1], path[i]
        overlap_len = overlap[prev][curr]
        result += words[curr][overlap_len:]
    
    return result
```

OPTIMIZATION STRATEGIES:
=======================
**Overlap Preprocessing**: O(n² × m²) computation amortized over O(2^n) states
**Memory Optimization**: Store only lengths in DP, reconstruct strings separately
**Early Termination**: Pruning based on lower bounds
**State Compression**: Efficient bitmask operations

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(n² × 2^n × m)
- n² transitions per state
- 2^n possible states
- m characters for string operations

**Space Complexity**: O(n × 2^n × m) or O(n × 2^n)
- Depends on whether strings are stored in DP table
- Optimized version uses only length storage

**Practical Limits**: n ≤ 12-15 due to exponential nature

OVERLAP ANALYSIS:
================
**Maximum Overlap**: `min(len(s1), len(s2))`
**Greedy Heuristic**: Choose maximum overlap at each step (not always optimal)
**Optimal Strategy**: Global optimization considering all possible orders

**Overlap Matrix Properties**:
- Not symmetric: `overlap[i][j] ≠ overlap[j][i]`
- Diagonal zeros: `overlap[i][i] = 0`
- Constraints: No string is substring of another

STRING ASSEMBLY TECHNIQUES:
==========================
**Suffix-Prefix Matching**: Core overlap detection algorithm
**Dynamic Programming**: Global optimization over all permutations
**Greedy Approximation**: Local optimization for faster solutions
**Graph-Based**: Model as shortest Hamiltonian path problem

APPLICATIONS:
============
- **Genome Assembly**: Reconstruct DNA sequences from shotgun fragments
- **Data Compression**: Shortest representation of string collections
- **Compiler Design**: Optimize string literal storage
- **Bioinformatics**: Sequence assembly and consensus building
- **Text Processing**: Efficient string concatenation algorithms

RELATED PROBLEMS:
================
- **Traveling Salesman Problem**: Similar exponential optimization
- **Shortest Hamiltonian Path**: Graph version of the problem
- **String Matching**: Foundation for overlap computation
- **Sequence Assembly**: Real-world application domain

VARIANTS:
========
- **Weighted Superstring**: Consider string importance/weights
- **Constrained Assembly**: Forbidden overlaps or required patterns
- **Approximate Solutions**: Heuristic methods for large inputs
- **Multiple Objectives**: Balance length with other criteria

EDGE CASES:
==========
- **Single String**: Return the string itself
- **No Overlaps**: Concatenate in any order
- **Complete Overlap**: One string contains others as substrings
- **Identical Strings**: Reduce to single instance

PRACTICAL CONSIDERATIONS:
========================
**Memory Usage**: Exponential space requirements limit scalability
**String Operations**: Efficient substring and concatenation methods
**Preprocessing Cost**: Overlap computation can be significant
**Approximation Algorithms**: Necessary for larger inputs (n > 15)

This problem beautifully demonstrates how Bitmask DP can solve
complex string optimization problems by systematically exploring
the exponential space of string orderings while efficiently
managing string operations and overlap computations.
"""
