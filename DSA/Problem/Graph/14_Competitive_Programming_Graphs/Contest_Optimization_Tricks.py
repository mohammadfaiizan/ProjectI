"""
Contest Optimization Tricks - Advanced Competitive Programming Techniques
Difficulty: Medium

This file contains advanced optimization techniques and tricks commonly used
in competitive programming contests. Focus on performance optimization,
clever implementations, and contest-specific strategies.

Key Concepts:
1. Bit Manipulation Optimizations
2. Fast Modular Arithmetic
3. Memory Pool Techniques
4. Cache-Friendly Algorithms
5. Compiler Optimizations
6. Contest-Specific Heuristics
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import sys
import bisect

class ContestOptimizationTricks:
    """Advanced optimization techniques for competitive programming"""
    
    def __init__(self):
        self.MOD = 10**9 + 7
        self.INF = float('inf')
    
    def fast_io_template(self):
        """
        Approach 1: Fast I/O Template
        
        Optimized input/output for competitive programming.
        """
        import sys
        from io import StringIO
        
        # Fast input
        def fast_input():
            return sys.stdin.readline().strip()
        
        def fast_int():
            return int(sys.stdin.readline())
        
        def fast_ints():
            return list(map(int, sys.stdin.readline().split()))
        
        # Fast output
        def fast_print(*args):
            print(*args)
        
        # Bulk operations
        def read_matrix(n, m):
            return [fast_ints() for _ in range(n)]
        
        def print_matrix(matrix):
            for row in matrix:
                print(' '.join(map(str, row)))
        
        return {
            'input': fast_input,
            'int': fast_int,
            'ints': fast_ints,
            'print': fast_print,
            'read_matrix': read_matrix,
            'print_matrix': print_matrix
        }
    
    def bit_manipulation_tricks(self):
        """
        Approach 2: Bit Manipulation Optimization Tricks
        
        Advanced bit manipulation techniques for contests.
        """
        
        def count_set_bits(n):
            """Count set bits using Brian Kernighan's algorithm"""
            count = 0
            while n:
                n &= n - 1  # Remove rightmost set bit
                count += 1
            return count
        
        def is_power_of_two(n):
            """Check if number is power of 2"""
            return n > 0 and (n & (n - 1)) == 0
        
        def next_power_of_two(n):
            """Find next power of 2"""
            if n <= 1:
                return 1
            return 1 << (n - 1).bit_length()
        
        def subset_enumeration(mask):
            """Enumerate all subsets of a bitmask"""
            subsets = []
            submask = mask
            while submask > 0:
                subsets.append(submask)
                submask = (submask - 1) & mask
            subsets.append(0)  # Empty subset
            return subsets
        
        def gray_code_generation(n):
            """Generate Gray code sequence"""
            if n == 0:
                return [0]
            
            prev = gray_code_generation(n - 1)
            result = prev[:]
            
            for i in range(len(prev) - 1, -1, -1):
                result.append(prev[i] | (1 << (n - 1)))
            
            return result
        
        def bit_reverse(n, bits):
            """Reverse bits in n-bit number"""
            result = 0
            for _ in range(bits):
                result = (result << 1) | (n & 1)
                n >>= 1
            return result
        
        return {
            'count_bits': count_set_bits,
            'is_power_2': is_power_of_two,
            'next_power_2': next_power_of_two,
            'subsets': subset_enumeration,
            'gray_code': gray_code_generation,
            'bit_reverse': bit_reverse
        }
    
    def fast_modular_arithmetic(self):
        """
        Approach 3: Fast Modular Arithmetic
        
        Optimized modular arithmetic operations.
        """
        MOD = self.MOD
        
        def mod_add(a, b, mod=MOD):
            """Fast modular addition"""
            return (a + b) % mod
        
        def mod_sub(a, b, mod=MOD):
            """Fast modular subtraction"""
            return (a - b + mod) % mod
        
        def mod_mul(a, b, mod=MOD):
            """Fast modular multiplication"""
            return (a * b) % mod
        
        def mod_pow(base, exp, mod=MOD):
            """Fast modular exponentiation"""
            result = 1
            base %= mod
            while exp > 0:
                if exp & 1:
                    result = (result * base) % mod
                base = (base * base) % mod
                exp >>= 1
            return result
        
        def mod_inv(a, mod=MOD):
            """Modular multiplicative inverse using Fermat's little theorem"""
            return mod_pow(a, mod - 2, mod)
        
        def mod_div(a, b, mod=MOD):
            """Modular division"""
            return mod_mul(a, mod_inv(b, mod), mod)
        
        # Precompute factorials and inverse factorials
        def precompute_factorials(n, mod=MOD):
            fact = [1] * (n + 1)
            for i in range(1, n + 1):
                fact[i] = (fact[i - 1] * i) % mod
            
            inv_fact = [1] * (n + 1)
            inv_fact[n] = mod_inv(fact[n], mod)
            for i in range(n - 1, -1, -1):
                inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % mod
            
            return fact, inv_fact
        
        def nCr(n, r, fact, inv_fact, mod=MOD):
            """Combination using precomputed factorials"""
            if r < 0 or r > n:
                return 0
            return (fact[n] * inv_fact[r] % mod) * inv_fact[n - r] % mod
        
        return {
            'add': mod_add,
            'sub': mod_sub,
            'mul': mod_mul,
            'pow': mod_pow,
            'inv': mod_inv,
            'div': mod_div,
            'precompute_fact': precompute_factorials,
            'nCr': nCr
        }
    
    def graph_optimization_tricks(self):
        """
        Approach 4: Graph Algorithm Optimizations
        
        Optimized graph algorithms for contests.
        """
        
        def adjacency_list_optimized(n, edges, directed=False):
            """Memory-optimized adjacency list"""
            # Pre-allocate exact space needed
            adj = [[] for _ in range(n + 1)]
            
            for u, v in edges:
                adj[u].append(v)
                if not directed:
                    adj[v].append(u)
            
            # Sort adjacency lists for binary search
            for i in range(n + 1):
                adj[i].sort()
            
            return adj
        
        def bfs_optimized(adj, start, n):
            """Cache-friendly BFS implementation"""
            dist = [-1] * (n + 1)
            parent = [-1] * (n + 1)
            
            # Use list as queue for better cache performance
            queue = [start]
            dist[start] = 0
            head = 0
            
            while head < len(queue):
                node = queue[head]
                head += 1
                
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
            
            return dist, parent
        
        def dfs_iterative_optimized(adj, start, n):
            """Memory-optimized iterative DFS"""
            visited = [False] * (n + 1)
            stack = [start]
            result = []
            
            while stack:
                node = stack.pop()
                
                if not visited[node]:
                    visited[node] = True
                    result.append(node)
                    
                    # Add neighbors in reverse order
                    for neighbor in reversed(adj[node]):
                        if not visited[neighbor]:
                            stack.append(neighbor)
            
            return result
        
        def dijkstra_optimized(adj, weights, start, n):
            """Optimized Dijkstra with binary heap"""
            import heapq
            
            dist = [float('inf')] * (n + 1)
            dist[start] = 0
            pq = [(0, start)]
            
            while pq:
                d, u = heapq.heappop(pq)
                
                if d > dist[u]:
                    continue
                
                for i, v in enumerate(adj[u]):
                    weight = weights[u][i]
                    if dist[u] + weight < dist[v]:
                        dist[v] = dist[u] + weight
                        heapq.heappush(pq, (dist[v], v))
            
            return dist
        
        return {
            'adj_list': adjacency_list_optimized,
            'bfs': bfs_optimized,
            'dfs': dfs_iterative_optimized,
            'dijkstra': dijkstra_optimized
        }
    
    def string_optimization_tricks(self):
        """
        Approach 5: String Processing Optimizations
        
        Fast string algorithms for contests.
        """
        
        def kmp_preprocessing(pattern):
            """KMP failure function preprocessing"""
            m = len(pattern)
            lps = [0] * m
            length = 0
            i = 1
            
            while i < m:
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            
            return lps
        
        def kmp_search(text, pattern):
            """KMP string matching"""
            n, m = len(text), len(pattern)
            if m == 0:
                return []
            
            lps = kmp_preprocessing(pattern)
            matches = []
            
            i = j = 0
            while i < n:
                if pattern[j] == text[i]:
                    i += 1
                    j += 1
                
                if j == m:
                    matches.append(i - j)
                    j = lps[j - 1]
                elif i < n and pattern[j] != text[i]:
                    if j != 0:
                        j = lps[j - 1]
                    else:
                        i += 1
            
            return matches
        
        def rolling_hash(s, base=31, mod=10**9 + 7):
            """Rolling hash for string"""
            n = len(s)
            hash_val = 0
            power = 1
            
            for i in range(n):
                hash_val = (hash_val + (ord(s[i]) - ord('a') + 1) * power) % mod
                power = (power * base) % mod
            
            return hash_val
        
        def z_algorithm(s):
            """Z-algorithm for string matching"""
            n = len(s)
            z = [0] * n
            l = r = 0
            
            for i in range(1, n):
                if i <= r:
                    z[i] = min(r - i + 1, z[i - l])
                
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
                
                if i + z[i] - 1 > r:
                    l, r = i, i + z[i] - 1
            
            return z
        
        return {
            'kmp_preprocess': kmp_preprocessing,
            'kmp_search': kmp_search,
            'rolling_hash': rolling_hash,
            'z_algorithm': z_algorithm
        }
    
    def contest_debugging_tricks(self):
        """
        Approach 6: Contest Debugging and Testing Tricks
        
        Debugging techniques for competitive programming.
        """
        
        def stress_testing_template(brute_force_func, optimized_func, 
                                  test_generator, num_tests=1000):
            """Stress testing template"""
            import random
            
            for test_num in range(num_tests):
                test_case = test_generator()
                
                try:
                    brute_result = brute_force_func(test_case)
                    optimized_result = optimized_func(test_case)
                    
                    if brute_result != optimized_result:
                        print(f"Test {test_num} failed!")
                        print(f"Input: {test_case}")
                        print(f"Brute force: {brute_result}")
                        print(f"Optimized: {optimized_result}")
                        return False
                        
                except Exception as e:
                    print(f"Test {test_num} crashed: {e}")
                    print(f"Input: {test_case}")
                    return False
            
            print(f"All {num_tests} tests passed!")
            return True
        
        def debug_print(var_name, var_value):
            """Debug printing with variable name"""
            print(f"DEBUG: {var_name} = {var_value}")
        
        def assert_equal(actual, expected, message=""):
            """Custom assertion with message"""
            if actual != expected:
                print(f"Assertion failed: {message}")
                print(f"Expected: {expected}")
                print(f"Actual: {actual}")
                raise AssertionError(message)
        
        def time_function(func, *args, **kwargs):
            """Time function execution"""
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Function took {end - start:.4f} seconds")
            return result
        
        return {
            'stress_test': stress_testing_template,
            'debug': debug_print,
            'assert_eq': assert_equal,
            'time_func': time_function
        }

def demonstrate_optimization_tricks():
    """Demonstrate various optimization tricks"""
    print("=== Contest Optimization Tricks Demo ===")
    
    optimizer = ContestOptimizationTricks()
    
    # Bit manipulation tricks
    print("\n--- Bit Manipulation Tricks ---")
    bit_tricks = optimizer.bit_manipulation_tricks()
    print(f"Count bits in 15: {bit_tricks['count_bits'](15)}")
    print(f"Is 16 power of 2: {bit_tricks['is_power_2'](16)}")
    print(f"Next power of 2 after 10: {bit_tricks['next_power_2'](10)}")
    
    # Modular arithmetic
    print("\n--- Fast Modular Arithmetic ---")
    mod_ops = optimizer.fast_modular_arithmetic()
    print(f"2^10 mod 1000000007: {mod_ops['pow'](2, 10)}")
    print(f"Inverse of 3 mod 1000000007: {mod_ops['inv'](3)}")
    
    # Graph optimizations
    print("\n--- Graph Optimizations ---")
    graph_ops = optimizer.graph_optimization_tricks()
    edges = [(1, 2), (2, 3), (3, 4), (1, 4)]
    adj = graph_ops['adj_list'](4, edges)
    print(f"Optimized adjacency list: {adj[1:5]}")
    
    # String optimizations
    print("\n--- String Optimizations ---")
    string_ops = optimizer.string_optimization_tricks()
    pattern = "aba"
    lps = string_ops['kmp_preprocess'](pattern)
    print(f"LPS array for '{pattern}': {lps}")
    
    print("\nOptimization techniques demonstrated successfully!")

def analyze_performance_tips():
    """Analyze performance optimization tips"""
    print("\n=== Performance Optimization Tips ===")
    
    print("Memory Optimizations:")
    print("• Use arrays instead of dictionaries when possible")
    print("• Pre-allocate data structures with known sizes")
    print("• Reuse arrays and clear them instead of creating new ones")
    print("• Use appropriate data types (int vs long long)")
    
    print("\nAlgorithm Optimizations:")
    print("• Choose optimal algorithms for problem constraints")
    print("• Use iterative instead of recursive when possible")
    print("• Implement early termination conditions")
    print("• Cache frequently computed values")
    
    print("\nImplementation Optimizations:")
    print("• Use fast I/O for large inputs")
    print("• Minimize function call overhead")
    print("• Use bit manipulation for boolean operations")
    print("• Optimize inner loops carefully")
    
    print("\nContest-Specific Tips:")
    print("• Read all problems before starting")
    print("• Implement easier problems first")
    print("• Use template code for common algorithms")
    print("• Test with edge cases and large inputs")

if __name__ == "__main__":
    demonstrate_optimization_tricks()
    analyze_performance_tips()

"""
Contest Optimization Tricks - Key Insights:

1. **Performance Focus:**
   - Every microsecond counts in contests
   - Optimize both algorithms and implementation
   - Use appropriate data structures
   - Consider memory access patterns

2. **Implementation Speed:**
   - Use template code for common patterns
   - Fast I/O for large inputs
   - Bit manipulation for efficiency
   - Modular arithmetic optimizations

3. **Algorithm Selection:**
   - Choose optimal complexity for constraints
   - Consider constant factors
   - Use problem-specific optimizations
   - Implement early termination

4. **Debugging Strategies:**
   - Stress testing with random inputs
   - Compare brute force vs optimized
   - Use assertion-based testing
   - Time critical functions

5. **Contest Strategy:**
   - Balance correctness and speed
   - Use proven implementations
   - Handle edge cases systematically
   - Optimize after correctness

These optimization tricks provide the competitive edge
needed for success in programming contests.
"""
