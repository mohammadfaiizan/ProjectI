"""
952. Largest Component Size by Common Factor
Difficulty: Hard

Problem:
You are given an integer array nums. You need to find the largest connected component 
size in the following graph:

- There are nums.length nodes, labeled nums[0] to nums[nums.length - 1].
- There is an undirected edge between nodes nums[i] and nums[j] if nums[i] and nums[j] 
  share a common factor greater than 1.

Return the size of the largest connected component in the graph.

Examples:
Input: nums = [4,6,15,35]
Output: 4

Input: nums = [20,50,9,63]
Output: 2

Input: nums = [2,3,6,7,4,12,21,39]
Output: 8

Constraints:
- 1 <= nums.length <= 2 * 10^4
- 1 <= nums[i] <= 10^5
- All the values of nums are unique
"""

from typing import List
from collections import defaultdict

class UnionFind:
    """Union-Find for grouping numbers by common factors"""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.max_size = 1
    
    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by size, track maximum component size"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        # Union by size
        if self.size[root_x] < self.size[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        self.max_size = max(self.max_size, self.size[root_x])
    
    def get_max_component_size(self):
        """Get size of largest component"""
        return self.max_size

class Solution:
    def largestComponentSize_approach1_prime_factorization(self, nums: List[int]) -> int:
        """
        Approach 1: Prime Factorization + Union-Find (Optimal)
        
        Use prime factorization to connect numbers through their prime factors.
        
        Time: O(N * sqrt(M)) where M = max(nums)
        Space: O(N + P) where P = number of unique prime factors
        """
        def get_prime_factors(n):
            """Get all prime factors of n"""
            factors = set()
            
            # Check for factor 2
            if n % 2 == 0:
                factors.add(2)
                while n % 2 == 0:
                    n //= 2
            
            # Check for odd factors
            i = 3
            while i * i <= n:
                if n % i == 0:
                    factors.add(i)
                    while n % i == 0:
                        n //= i
                i += 2
            
            # If n is still > 1, then it's a prime
            if n > 1:
                factors.add(n)
            
            return factors
        
        # Map from prime factor to number indices
        prime_to_indices = defaultdict(list)
        
        # Get prime factors for each number
        for i, num in enumerate(nums):
            prime_factors = get_prime_factors(num)
            for prime in prime_factors:
                prime_to_indices[prime].append(i)
        
        # Union-Find on number indices
        uf = UnionFind(len(nums))
        
        # Union all numbers that share a prime factor
        for indices in prime_to_indices.values():
            for i in range(1, len(indices)):
                uf.union(indices[0], indices[i])
        
        return uf.get_max_component_size()
    
    def largestComponentSize_approach2_factor_mapping(self, nums: List[int]) -> int:
        """
        Approach 2: All Factor Mapping
        
        Map each number to all its factors (not just primes).
        
        Time: O(N * sqrt(M))
        Space: O(N * F) where F = average factors per number
        """
        def get_all_factors(n):
            """Get all factors of n greater than 1"""
            factors = set()
            
            i = 2
            while i * i <= n:
                if n % i == 0:
                    factors.add(i)
                    if i != n // i:
                        factors.add(n // i)
                i += 1
            
            if n > 1:
                factors.add(n)
            
            return factors
        
        # Map from factor to number indices
        factor_to_indices = defaultdict(list)
        
        for i, num in enumerate(nums):
            factors = get_all_factors(num)
            for factor in factors:
                factor_to_indices[factor].append(i)
        
        # Union-Find
        uf = UnionFind(len(nums))
        
        # Union numbers sharing any factor
        for indices in factor_to_indices.values():
            for i in range(1, len(indices)):
                uf.union(indices[0], indices[i])
        
        return uf.get_max_component_size()
    
    def largestComponentSize_approach3_sieve_optimization(self, nums: List[int]) -> int:
        """
        Approach 3: Sieve of Eratosthenes for Prime Finding
        
        Pre-compute primes using sieve for efficiency.
        
        Time: O(M log log M + N * sqrt(M))
        Space: O(M + N)
        """
        def sieve_of_eratosthenes(max_num):
            """Generate all primes up to max_num"""
            is_prime = [True] * (max_num + 1)
            is_prime[0] = is_prime[1] = False
            
            for i in range(2, int(max_num**0.5) + 1):
                if is_prime[i]:
                    for j in range(i*i, max_num + 1, i):
                        is_prime[j] = False
            
            return [i for i in range(2, max_num + 1) if is_prime[i]]
        
        def get_prime_factors_with_sieve(n, primes):
            """Get prime factors using precomputed primes"""
            factors = set()
            
            for prime in primes:
                if prime * prime > n:
                    break
                
                if n % prime == 0:
                    factors.add(prime)
                    while n % prime == 0:
                        n //= prime
            
            if n > 1:
                factors.add(n)
            
            return factors
        
        if not nums:
            return 0
        
        max_num = max(nums)
        primes = sieve_of_eratosthenes(max_num)
        
        # Map prime factors to indices
        prime_to_indices = defaultdict(list)
        
        for i, num in enumerate(nums):
            prime_factors = get_prime_factors_with_sieve(num, primes)
            for prime in prime_factors:
                prime_to_indices[prime].append(i)
        
        # Union-Find
        uf = UnionFind(len(nums))
        
        for indices in prime_to_indices.values():
            for i in range(1, len(indices)):
                uf.union(indices[0], indices[i])
        
        return uf.get_max_component_size()
    
    def largestComponentSize_approach4_optimized_union_find(self, nums: List[int]) -> int:
        """
        Approach 4: Optimized Union-Find with Factor Caching
        
        Cache factors to avoid recomputation and optimize Union-Find.
        
        Time: O(N * sqrt(M))
        Space: O(N + F)
        """
        # Cache for memoizing prime factors
        factor_cache = {}
        
        def get_prime_factors_cached(n):
            """Get prime factors with caching"""
            if n in factor_cache:
                return factor_cache[n]
            
            original_n = n
            factors = set()
            
            # Check for factor 2
            if n % 2 == 0:
                factors.add(2)
                while n % 2 == 0:
                    n //= 2
            
            # Check odd factors
            i = 3
            while i * i <= n:
                if n % i == 0:
                    factors.add(i)
                    while n % i == 0:
                        n //= i
                i += 2
            
            if n > 1:
                factors.add(n)
            
            factor_cache[original_n] = factors
            return factors
        
        # Union-Find with factor-based unions
        parent = list(range(len(nums)))
        size = [1] * len(nums)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x == root_y:
                return
            
            if size[root_x] < size[root_y]:
                root_x, root_y = root_y, root_x
            
            parent[root_y] = root_x
            size[root_x] += size[root_y]
        
        # Map factors to first occurrence
        factor_to_first = {}
        
        for i, num in enumerate(nums):
            prime_factors = get_prime_factors_cached(num)
            
            for prime in prime_factors:
                if prime in factor_to_first:
                    union(i, factor_to_first[prime])
                else:
                    factor_to_first[prime] = i
        
        # Find maximum component size
        return max(size[find(i)] for i in range(len(nums)))

def test_largest_component_size():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (nums, expected)
        ([4,6,15,35], 4),
        ([20,50,9,63], 2),
        ([2,3,6,7,4,12,21,39], 8),
        ([83,99,39,11,19,30,31], 4),
        ([1], 1),
        ([2,4,8,16], 4),
        ([3,9,27], 3),
    ]
    
    approaches = [
        ("Prime Factorization", solution.largestComponentSize_approach1_prime_factorization),
        ("All Factor Mapping", solution.largestComponentSize_approach2_factor_mapping),
        ("Sieve Optimization", solution.largestComponentSize_approach3_sieve_optimization),
        ("Optimized Union-Find", solution.largestComponentSize_approach4_optimized_union_find),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (nums, expected) in enumerate(test_cases):
            result = func(nums[:])  # Copy to avoid modification
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} nums={nums}, expected={expected}, got={result}")

def demonstrate_prime_factorization():
    """Demonstrate prime factorization process"""
    print("\n=== Prime Factorization Demo ===")
    
    nums = [4, 6, 15, 35]
    print(f"Numbers: {nums}")
    
    def get_prime_factors(n):
        factors = set()
        
        if n % 2 == 0:
            factors.add(2)
            while n % 2 == 0:
                n //= 2
        
        i = 3
        while i * i <= n:
            if n % i == 0:
                factors.add(i)
                while n % i == 0:
                    n //= i
            i += 2
        
        if n > 1:
            factors.add(n)
        
        return factors
    
    print(f"\nPrime factorization:")
    prime_to_numbers = defaultdict(list)
    
    for i, num in enumerate(nums):
        prime_factors = get_prime_factors(num)
        print(f"  {num}: prime factors = {sorted(prime_factors)}")
        
        for prime in prime_factors:
            prime_to_numbers[prime].append((i, num))
    
    print(f"\nGrouping by prime factors:")
    for prime in sorted(prime_to_numbers.keys()):
        numbers = [num for _, num in prime_to_numbers[prime]]
        indices = [i for i, _ in prime_to_numbers[prime]]
        print(f"  Prime {prime}: numbers {numbers} (indices {indices})")
    
    # Union-Find simulation
    print(f"\nUnion-Find simulation:")
    uf = UnionFind(len(nums))
    
    for prime, num_indices in prime_to_numbers.items():
        indices = [i for i, _ in num_indices]
        print(f"\n  Prime factor {prime} connects indices: {indices}")
        
        # Union all indices sharing this prime
        for i in range(1, len(indices)):
            print(f"    Union({indices[0]}, {indices[i]})")
            uf.union(indices[0], indices[i])
        
        # Show current components
        components = defaultdict(list)
        for i in range(len(nums)):
            root = uf.find(i)
            components[root].append(i)
        
        comp_list = [indices for indices in components.values() if len(indices) > 0]
        print(f"    Current components: {comp_list}")
    
    print(f"\nFinal result: {uf.get_max_component_size()}")

def analyze_mathematical_connections():
    """Analyze mathematical relationships in the problem"""
    print("\n=== Mathematical Connections Analysis ===")
    
    print("Problem Foundation:")
    print("• Two numbers are connected if gcd(a,b) > 1")
    print("• This creates equivalence classes of numbers")
    print("• Union-Find naturally models these connections")
    print("• Goal: Find largest equivalence class")
    
    print("\nKey Mathematical Insights:")
    
    print("\n1. **Prime Factorization Approach:**")
    print("   • gcd(a,b) > 1 ⟺ a and b share a prime factor")
    print("   • Only need to consider prime factors")
    print("   • Composite factors are redundant")
    print("   • More efficient than checking all factors")
    
    print("\n2. **Transitivity Property:**")
    print("   • If gcd(a,b) > 1 and gcd(b,c) > 1, then a and c are connected")
    print("   • Even if gcd(a,c) = 1 directly")
    print("   • Union-Find handles transitivity automatically")
    
    print("\n3. **Factor vs Prime Factor Trade-off:**")
    print("   • All factors: More connections, higher memory")
    print("   • Prime factors: Sufficient connections, optimal memory")
    print("   • Prime factorization is the key insight")
    
    print("\nExample Analysis:")
    examples = [
        (12, [2, 3]),
        (18, [2, 3]),
        (20, [2, 5]),
        (15, [3, 5])
    ]
    
    print("Numbers and their prime factors:")
    for num, primes in examples:
        print(f"  {num} = {primes}")
    
    print("\nConnections through shared primes:")
    print("  12 ↔ 18 (share prime 2 and 3)")
    print("  12 ↔ 20 (share prime 2)")
    print("  18 ↔ 15 (share prime 3)")
    print("  20 ↔ 15 (share prime 5)")
    print("  → All four numbers are connected transitively")

def demonstrate_optimization_techniques():
    """Demonstrate various optimization techniques"""
    print("\n=== Optimization Techniques Demo ===")
    
    print("1. **Prime vs All Factors:**")
    
    def get_all_factors(n):
        factors = set()
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                factors.add(i)
                if i != n // i:
                    factors.add(n // i)
        if n > 1:
            factors.add(n)
        return factors
    
    def get_prime_factors(n):
        factors = set()
        if n % 2 == 0:
            factors.add(2)
            while n % 2 == 0:
                n //= 2
        
        i = 3
        while i * i <= n:
            if n % i == 0:
                factors.add(i)
                while n % i == 0:
                    n //= i
            i += 2
        
        if n > 1:
            factors.add(n)
        return factors
    
    test_num = 60
    all_factors = get_all_factors(test_num)
    prime_factors = get_prime_factors(test_num)
    
    print(f"  Number: {test_num}")
    print(f"  All factors: {sorted(all_factors)} (count: {len(all_factors)})")
    print(f"  Prime factors: {sorted(prime_factors)} (count: {len(prime_factors)})")
    print(f"  Memory reduction: {len(all_factors) - len(prime_factors)} factors saved")
    
    print("\n2. **Sieve of Eratosthenes:**")
    print("   • Pre-compute all primes up to max(nums)")
    print("   • Faster factorization for multiple numbers")
    print("   • Trade memory for time")
    
    print("\n3. **Factor Caching:**")
    print("   • Memoize prime factorizations")
    print("   • Avoid recomputing for duplicate numbers")
    print("   • Significant speedup for repeated values")
    
    print("\n4. **Union-Find Optimizations:**")
    print("   • Path compression: O(α(n)) find operations")
    print("   • Union by size: balanced tree structures")
    print("   • Track max component size during unions")

def compare_complexity_analysis():
    """Compare complexity of different approaches"""
    print("\n=== Complexity Analysis Comparison ===")
    
    print("Approach Comparison (N = array length, M = max value):")
    
    print("\n1. **Naive Pairwise GCD:**")
    print("   • Time: O(N² * log M) - check all pairs")
    print("   • Space: O(N²) - store all connections")
    print("   • Impractical for large inputs")
    
    print("\n2. **Prime Factorization + Union-Find:**")
    print("   • Time: O(N * sqrt(M) + N * α(N))")
    print("   • Space: O(N + P) where P = unique primes")
    print("   • Optimal for most cases")
    
    print("\n3. **All Factor Mapping:**")
    print("   • Time: O(N * sqrt(M))")
    print("   • Space: O(N * F) where F = avg factors per number")
    print("   • Higher space complexity")
    
    print("\n4. **Sieve + Prime Factorization:**")
    print("   • Time: O(M log log M + N * log log M)")
    print("   • Space: O(M + N)")
    print("   • Better for multiple queries")
    
    print("\nOptimal Choice:")
    print("• **Small arrays, large numbers:** Prime factorization")
    print("• **Large arrays, small numbers:** Sieve approach")
    print("• **Multiple queries:** Sieve with caching")
    print("• **Memory constrained:** Prime factorization only")
    
    print("\nReal-world Applications:")
    print("• **Number Theory:** Finding related mathematical objects")
    print("• **Cryptography:** Analyzing number relationships")
    print("• **Data Clustering:** Grouping by mathematical properties")
    print("• **Algorithm Design:** Prime-based partitioning")
    print("• **Computational Mathematics:** Factor analysis")

if __name__ == "__main__":
    test_largest_component_size()
    demonstrate_prime_factorization()
    analyze_mathematical_connections()
    demonstrate_optimization_techniques()
    compare_complexity_analysis()

"""
Union-Find Concepts:
1. Mathematical Relationship Modeling
2. Prime Factorization in Graph Problems
3. Equivalence Classes Through Common Factors
4. Optimization in Number Theory Applications

Key Problem Insights:
- Numbers connected if they share common factor > 1
- Prime factorization sufficient for determining connections
- Union-Find models transitivity of factor relationships
- Mathematical optimization reduces computational complexity

Algorithm Strategy:
1. Extract prime factors for each number
2. Group numbers by shared prime factors
3. Use Union-Find to merge related groups
4. Find largest connected component

Mathematical Foundation:
- gcd(a,b) > 1 ⟺ a,b share prime factor
- Transitivity: factor relationships propagate
- Prime factorization captures all necessary connections
- Factor-based equivalence classes

Advanced Optimizations:
- Sieve of Eratosthenes for prime generation
- Factor caching for repeated computations
- Union by size for balanced structures
- Prime-only analysis vs all factors

Real-world Applications:
- Computational number theory
- Mathematical clustering problems
- Cryptographic analysis
- Algorithm optimization
- Data partitioning by mathematical properties

This problem demonstrates Union-Find applications
in advanced mathematical and number theory contexts.
"""
