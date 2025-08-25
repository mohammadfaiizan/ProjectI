"""
1286. Iterator for Combination - Multiple Approaches
Difficulty: Medium

Design the CombinationIterator class:
- CombinationIterator(string characters, int combinationLength) Initializes the object with a string characters of sorted distinct lowercase English letters and a number combinationLength as arguments.
- next() Returns the next combination of length combinationLength in lexicographical order.
- hasNext() Returns true if and only if there exists a next combination.
"""

from typing import List, Iterator
import itertools

class CombinationIteratorPrecompute:
    """
    Approach 1: Precompute All Combinations
    
    Generate all combinations upfront and iterate through them.
    
    Time Complexity:
    - __init__: O(C(n,k)) where n is len(characters), k is combinationLength
    - next: O(1)
    - hasNext: O(1)
    
    Space Complexity: O(C(n,k) * k)
    """
    
    def __init__(self, characters: str, combinationLength: int):
        self.combinations = []
        self.index = 0
        
        # Generate all combinations
        from itertools import combinations
        for combo in combinations(characters, combinationLength):
            self.combinations.append(''.join(combo))
    
    def next(self) -> str:
        if self.hasNext():
            result = self.combinations[self.index]
            self.index += 1
            return result
        return ""
    
    def hasNext(self) -> bool:
        return self.index < len(self.combinations)

class CombinationIteratorLazy:
    """
    Approach 2: Lazy Generation with Backtracking
    
    Generate combinations on-demand using backtracking.
    
    Time Complexity:
    - __init__: O(1)
    - next: O(k) amortized
    - hasNext: O(k)
    
    Space Complexity: O(k)
    """
    
    def __init__(self, characters: str, combinationLength: int):
        self.characters = characters
        self.k = combinationLength
        self.n = len(characters)
        
        # Current combination represented as indices
        self.current = list(range(self.k))
        self.has_next = True
        
        # Flag to track if we need to generate next combination
        self.first_call = True
    
    def next(self) -> str:
        if not self.hasNext():
            return ""
        
        # Return current combination
        result = ''.join(self.characters[i] for i in self.current)
        
        # Generate next combination
        self._generate_next()
        
        return result
    
    def hasNext(self) -> bool:
        return self.has_next
    
    def _generate_next(self) -> None:
        """Generate next combination using next permutation logic"""
        # Find rightmost index that can be incremented
        i = self.k - 1
        while i >= 0 and self.current[i] == self.n - self.k + i:
            i -= 1
        
        if i < 0:
            # No more combinations
            self.has_next = False
            return
        
        # Increment the found index
        self.current[i] += 1
        
        # Set all following indices to consecutive values
        for j in range(i + 1, self.k):
            self.current[j] = self.current[i] + (j - i)

class CombinationIteratorRecursive:
    """
    Approach 3: Recursive Generation
    
    Use recursive approach to generate combinations.
    
    Time Complexity:
    - __init__: O(1)
    - next: O(k)
    - hasNext: O(1)
    
    Space Complexity: O(k)
    """
    
    def __init__(self, characters: str, combinationLength: int):
        self.characters = characters
        self.k = combinationLength
        self.n = len(characters)
        
        # Stack for recursive state
        self.stack = []
        self.current_combination = []
        self.next_combination = None
        
        # Initialize the iteration
        self._generate_next()
    
    def next(self) -> str:
        if not self.hasNext():
            return ""
        
        result = self.next_combination
        self._generate_next()
        return result
    
    def hasNext(self) -> bool:
        return self.next_combination is not None
    
    def _generate_next(self) -> None:
        """Generate next combination recursively"""
        if not self.stack and not self.current_combination:
            # Initialize: start with empty combination at position 0
            self.stack = [(0, [])]
        
        while self.stack:
            pos, combo = self.stack.pop()
            
            if len(combo) == self.k:
                # Found complete combination
                self.next_combination = ''.join(combo)
                return
            
            # Try all possible next characters
            start_idx = pos
            max_remaining = self.k - len(combo)
            max_start = self.n - max_remaining
            
            # Add states in reverse order for correct lexicographical order
            for i in range(min(max_start + 1, self.n), start_idx - 1, -1):
                if i < self.n:
                    self.stack.append((i + 1, combo + [self.characters[i]]))
        
        # No more combinations
        self.next_combination = None

class CombinationIteratorBitmask:
    """
    Approach 4: Bitmask-based Generation
    
    Use bitmasks to represent and generate combinations.
    
    Time Complexity:
    - __init__: O(1)
    - next: O(n)
    - hasNext: O(1)
    
    Space Complexity: O(1)
    """
    
    def __init__(self, characters: str, combinationLength: int):
        self.characters = characters
        self.k = combinationLength
        self.n = len(characters)
        
        # Start with the lexicographically smallest combination
        # This is represented as the k rightmost bits set
        self.current_mask = (1 << self.k) - 1
        self.max_mask = ((1 << self.k) - 1) << (self.n - self.k)
        
        self.finished = False
    
    def next(self) -> str:
        if not self.hasNext():
            return ""
        
        # Convert current mask to string
        result = []
        for i in range(self.n):
            if (self.current_mask >> i) & 1:
                result.append(self.characters[self.n - 1 - i])
        
        result.reverse()  # Correct order
        
        # Generate next mask
        self._next_mask()
        
        return ''.join(result)
    
    def hasNext(self) -> bool:
        return not self.finished
    
    def _next_mask(self) -> None:
        """Generate next bitmask combination"""
        if self.current_mask == self.max_mask:
            self.finished = True
            return
        
        # Find next combination using bit manipulation
        # This implements the "next permutation" algorithm for bitmasks
        
        # Find the rightmost set bit that can be moved left
        temp = self.current_mask
        c0 = 0  # Count of trailing zeros
        c1 = 0  # Count of ones to the right of trailing zeros
        
        # Count trailing zeros
        while ((temp & 1) == 0) and temp != 0:
            c0 += 1
            temp >>= 1
        
        # Count ones after trailing zeros
        while (temp & 1) == 1:
            c1 += 1
            temp >>= 1
        
        # Error check: if c0 + c1 == 31 or c0 + c1 == 0, then no next combination
        if c0 + c1 == self.n or c0 + c1 == 0:
            self.finished = True
            return
        
        # Position of rightmost non-trailing zero
        pos = c0 + c1
        
        # Flip the rightmost non-trailing zero
        self.current_mask |= (1 << pos)
        
        # Clear all bits to the right of pos
        self.current_mask &= ~((1 << pos) - 1)
        
        # Insert (c1-1) ones on the right
        self.current_mask |= (1 << (c1 - 1)) - 1

class CombinationIteratorAdvanced:
    """
    Approach 5: Advanced with Multiple Features
    
    Enhanced iterator with additional functionality and optimizations.
    
    Time Complexity:
    - __init__: O(1)
    - next: O(k)
    - hasNext: O(1)
    
    Space Complexity: O(k)
    """
    
    def __init__(self, characters: str, combinationLength: int):
        self.characters = characters
        self.k = combinationLength
        self.n = len(characters)
        
        # Current combination as indices
        self.indices = list(range(self.k))
        self.has_next_combo = True
        
        # Statistics
        self.total_combinations = self._calculate_total_combinations()
        self.combinations_generated = 0
        
        # Caching for performance
        self.cache_size = min(100, self.total_combinations)
        self.cache = []
        self.cache_index = 0
        
        # Pre-generate first batch
        self._populate_cache()
    
    def _calculate_total_combinations(self) -> int:
        """Calculate total number of combinations C(n,k)"""
        if self.k > self.n:
            return 0
        
        # Calculate C(n,k) = n! / (k! * (n-k)!)
        result = 1
        for i in range(min(self.k, self.n - self.k)):
            result = result * (self.n - i) // (i + 1)
        
        return result
    
    def _populate_cache(self) -> None:
        """Populate cache with next batch of combinations"""
        while len(self.cache) < self.cache_size and self.has_next_combo:
            # Generate current combination
            combo = ''.join(self.characters[i] for i in self.indices)
            self.cache.append(combo)
            
            # Move to next combination
            self._advance_indices()
    
    def _advance_indices(self) -> None:
        """Advance to next combination indices"""
        # Find rightmost index that can be incremented
        i = self.k - 1
        while i >= 0 and self.indices[i] == self.n - self.k + i:
            i -= 1
        
        if i < 0:
            self.has_next_combo = False
            return
        
        # Increment the found index
        self.indices[i] += 1
        
        # Set following indices to consecutive values
        for j in range(i + 1, self.k):
            self.indices[j] = self.indices[i] + (j - i)
    
    def next(self) -> str:
        if not self.hasNext():
            return ""
        
        # Get from cache if available
        if self.cache_index < len(self.cache):
            result = self.cache[self.cache_index]
            self.cache_index += 1
            self.combinations_generated += 1
            
            # Refresh cache if needed
            if self.cache_index >= len(self.cache) and self.has_next_combo:
                self.cache = []
                self.cache_index = 0
                self._populate_cache()
            
            return result
        
        return ""
    
    def hasNext(self) -> bool:
        return self.cache_index < len(self.cache) or self.has_next_combo
    
    def get_progress(self) -> dict:
        """Get iteration progress information"""
        progress_percent = (self.combinations_generated / self.total_combinations) * 100
        
        return {
            'combinations_generated': self.combinations_generated,
            'total_combinations': self.total_combinations,
            'progress_percent': progress_percent,
            'remaining': self.total_combinations - self.combinations_generated
        }
    
    def peek(self) -> str:
        """Peek at next combination without advancing"""
        if not self.hasNext():
            return ""
        
        if self.cache_index < len(self.cache):
            return self.cache[self.cache_index]
        
        return ""
    
    def skip(self, count: int) -> int:
        """Skip next count combinations, return actual skipped count"""
        skipped = 0
        
        while skipped < count and self.hasNext():
            self.next()
            skipped += 1
        
        return skipped
    
    def reset(self) -> None:
        """Reset iterator to beginning"""
        self.indices = list(range(self.k))
        self.has_next_combo = True
        self.combinations_generated = 0
        
        # Reset cache
        self.cache = []
        self.cache_index = 0
        self._populate_cache()


def test_combination_iterator_basic():
    """Test basic combination iterator functionality"""
    print("=== Testing Basic Combination Iterator Functionality ===")
    
    implementations = [
        ("Precompute", CombinationIteratorPrecompute),
        ("Lazy", CombinationIteratorLazy),
        ("Recursive", CombinationIteratorRecursive),
        ("Bitmask", CombinationIteratorBitmask),
        ("Advanced", CombinationIteratorAdvanced)
    ]
    
    test_cases = [
        ("abc", 2),
        ("abcd", 3),
        ("ab", 1)
    ]
    
    for characters, combination_length in test_cases:
        print(f"\nTest case: characters='{characters}', combinationLength={combination_length}")
        
        # Generate expected results using itertools for verification
        from itertools import combinations
        expected = [''.join(combo) for combo in combinations(characters, combination_length)]
        
        for name, IteratorClass in implementations:
            print(f"\n{name}:")
            
            iterator = IteratorClass(characters, combination_length)
            results = []
            
            while iterator.hasNext():
                results.append(iterator.next())
            
            print(f"  Generated: {results}")
            print(f"  Expected:  {expected}")
            print(f"  Correct:   {results == expected}")

def test_combination_iterator_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Combination Iterator Edge Cases ===")
    
    # Test single character
    print("Single character:")
    iterator = CombinationIteratorAdvanced("a", 1)
    
    results = []
    while iterator.hasNext():
        results.append(iterator.next())
    
    print(f"  'a', length 1: {results}")
    
    # Test combination length equals string length
    print(f"\nCombination length equals string length:")
    iterator = CombinationIteratorLazy("xyz", 3)
    
    results = []
    while iterator.hasNext():
        results.append(iterator.next())
    
    print(f"  'xyz', length 3: {results}")
    
    # Test larger example
    print(f"\nLarger example:")
    iterator = CombinationIteratorBitmask("abcde", 3)
    
    count = 0
    first_few = []
    
    while iterator.hasNext() and count < 5:
        combo = iterator.next()
        first_few.append(combo)
        count += 1
    
    # Count remaining
    while iterator.hasNext():
        iterator.next()
        count += 1
    
    print(f"  'abcde', length 3: first 5 = {first_few}, total = {count}")
    
    # Test calling next() when hasNext() is false
    print(f"\nCalling next() when finished:")
    iterator = CombinationIteratorPrecompute("ab", 2)
    
    # Exhaust iterator
    while iterator.hasNext():
        iterator.next()
    
    # Try to call next() again
    result = iterator.next()
    print(f"  next() after exhaustion: '{result}'")

def test_performance_comparison():
    """Test performance of different implementations"""
    print("\n=== Testing Performance Comparison ===")
    
    import time
    
    implementations = [
        ("Precompute", CombinationIteratorPrecompute),
        ("Lazy", CombinationIteratorLazy),
        ("Bitmask", CombinationIteratorBitmask),
        ("Advanced", CombinationIteratorAdvanced)
    ]
    
    test_case = ("abcdefghij", 4)  # Moderate size for timing
    characters, combination_length = test_case
    
    print(f"Performance test: '{characters}', length {combination_length}")
    
    for name, IteratorClass in implementations:
        # Time initialization
        start_time = time.time()
        iterator = IteratorClass(characters, combination_length)
        init_time = (time.time() - start_time) * 1000
        
        # Time iteration
        start_time = time.time()
        count = 0
        while iterator.hasNext():
            iterator.next()
            count += 1
        iteration_time = (time.time() - start_time) * 1000
        
        total_time = init_time + iteration_time
        
        print(f"  {name}:")
        print(f"    Init: {init_time:.2f}ms")
        print(f"    Iteration: {iteration_time:.2f}ms")
        print(f"    Total: {total_time:.2f}ms")
        print(f"    Combinations: {count}")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    iterator = CombinationIteratorAdvanced("abcdef", 3)
    
    # Test progress tracking
    print("Progress tracking:")
    
    for i in range(5):
        if iterator.hasNext():
            combo = iterator.next()
            progress = iterator.get_progress()
            
            print(f"  Combination {i+1}: {combo}")
            print(f"    Progress: {progress['progress_percent']:.1f}%")
            print(f"    Generated: {progress['combinations_generated']}")
            print(f"    Remaining: {progress['remaining']}")
    
    # Test peek functionality
    print(f"\nPeek functionality:")
    
    if iterator.hasNext():
        peeked = iterator.peek()
        next_combo = iterator.next()
        
        print(f"  Peeked: {peeked}")
        print(f"  Next:   {next_combo}")
        print(f"  Match:  {peeked == next_combo}")
    
    # Test skip functionality
    print(f"\nSkip functionality:")
    
    skipped = iterator.skip(5)
    print(f"  Requested to skip 5, actually skipped: {skipped}")
    
    if iterator.hasNext():
        next_after_skip = iterator.next()
        print(f"  Next combination after skip: {next_after_skip}")
    
    # Test reset functionality
    print(f"\nReset functionality:")
    
    before_reset = iterator.get_progress()
    iterator.reset()
    after_reset = iterator.get_progress()
    
    print(f"  Before reset - generated: {before_reset['combinations_generated']}")
    print(f"  After reset - generated: {after_reset['combinations_generated']}")
    
    if iterator.hasNext():
        first_after_reset = iterator.next()
        print(f"  First combination after reset: {first_after_reset}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Password generation
    print("Application 1: Password Combination Generation")
    
    charset = "abcdef123"
    password_length = 4
    
    password_iterator = CombinationIteratorAdvanced(charset, password_length)
    
    print(f"  Generating {password_length}-character combinations from '{charset}':")
    
    password_count = 0
    sample_passwords = []
    
    while password_iterator.hasNext() and password_count < 10:
        password = password_iterator.next()
        sample_passwords.append(password)
        password_count += 1
    
    for i, password in enumerate(sample_passwords):
        print(f"    Password {i+1}: {password}")
    
    total_progress = password_iterator.get_progress()
    print(f"  Total possible passwords: {total_progress['total_combinations']}")
    
    # Application 2: Feature selection for ML
    print(f"\nApplication 2: Feature Selection for Machine Learning")
    
    features = "ABCDEFGH"  # 8 features labeled A through H
    select_count = 3
    
    feature_iterator = CombinationIteratorLazy(features, select_count)
    
    print(f"  Selecting {select_count} features from {len(features)} available features:")
    
    feature_combinations = []
    while feature_iterator.hasNext() and len(feature_combinations) < 8:
        combo = feature_iterator.next()
        feature_combinations.append(combo)
    
    for i, combo in enumerate(feature_combinations):
        feature_names = [f"Feature_{char}" for char in combo]
        print(f"    Combination {i+1}: {', '.join(feature_names)}")
    
    # Application 3: Test case generation
    print(f"\nApplication 3: Test Case Generation")
    
    test_parameters = "xyz"  # 3 different parameters
    param_combination_size = 2
    
    test_iterator = CombinationIteratorBitmask(test_parameters, param_combination_size)
    
    print(f"  Generating test cases with {param_combination_size} parameters:")
    
    test_cases = []
    while test_iterator.hasNext():
        params = test_iterator.next()
        test_cases.append(params)
    
    for i, test_case in enumerate(test_cases):
        param_descriptions = []
        for param in test_case:
            if param == 'x':
                param_descriptions.append("Network_Delay")
            elif param == 'y':
                param_descriptions.append("High_Load")
            elif param == 'z':
                param_descriptions.append("Low_Memory")
        
        print(f"    Test Case {i+1}: {', '.join(param_descriptions)}")
    
    # Application 4: Menu combination generator
    print(f"\nApplication 4: Restaurant Menu Combinations")
    
    ingredients = "ABCDE"  # 5 ingredients: A=Tomato, B=Cheese, C=Lettuce, D=Onion, E=Pickle
    combo_size = 3
    
    menu_iterator = CombinationIteratorAdvanced(ingredients, combo_size)
    
    ingredient_names = {
        'A': 'Tomato',
        'B': 'Cheese', 
        'C': 'Lettuce',
        'D': 'Onion',
        'E': 'Pickle'
    }
    
    print(f"  Generating sandwich combinations with {combo_size} ingredients:")
    
    sandwich_count = 0
    while menu_iterator.hasNext() and sandwich_count < 6:
        combo = menu_iterator.next()
        ingredients_list = [ingredient_names[ing] for ing in combo]
        
        sandwich_count += 1
        print(f"    Sandwich {sandwich_count}: {', '.join(ingredients_list)}")
    
    total_combos = menu_iterator.get_progress()['total_combinations']
    print(f"  Total possible sandwich combinations: {total_combos}")

def test_large_scale():
    """Test with larger scale inputs"""
    print("\n=== Testing Large Scale Inputs ===")
    
    # Test with larger alphabet
    large_alphabet = "abcdefghijklmnop"  # 16 characters
    
    test_cases = [
        (large_alphabet, 3),
        (large_alphabet, 4),
        (large_alphabet, 5)
    ]
    
    for characters, k in test_cases:
        print(f"\nCharacters: '{characters}' (length {len(characters)}), k={k}")
        
        # Calculate expected total
        n = len(characters)
        total_expected = 1
        for i in range(k):
            total_expected = total_expected * (n - i) // (i + 1)
        
        print(f"  Expected total combinations: {total_expected}")
        
        # Test different implementations for correctness
        implementations = [
            ("Lazy", CombinationIteratorLazy),
            ("Bitmask", CombinationIteratorBitmask)
        ]
        
        for name, IteratorClass in implementations:
            iterator = IteratorClass(characters, k)
            
            count = 0
            first_few = []
            last_few = []
            
            while iterator.hasNext():
                combo = iterator.next()
                count += 1
                
                if count <= 3:
                    first_few.append(combo)
                
                if count > total_expected - 3:
                    last_few.append(combo)
            
            print(f"    {name}: Generated {count} combinations")
            print(f"      First 3: {first_few}")
            print(f"      Last 3: {last_few}")
            print(f"      Correct count: {count == total_expected}")

def stress_test_iterators():
    """Stress test iterators"""
    print("\n=== Stress Testing Iterators ===")
    
    import time
    
    # Stress test parameters
    alphabet = "abcdefghijkl"  # 12 characters
    combination_length = 5
    
    print(f"Stress test: '{alphabet}' (length {len(alphabet)}), k={combination_length}")
    
    # Calculate expected combinations
    n = len(alphabet)
    k = combination_length
    expected_total = 1
    for i in range(k):
        expected_total = expected_total * (n - i) // (i + 1)
    
    print(f"Expected combinations: {expected_total}")
    
    implementations = [
        ("Lazy", CombinationIteratorLazy),
        ("Advanced", CombinationIteratorAdvanced)
    ]
    
    for name, IteratorClass in implementations:
        print(f"\n{name}:")
        
        start_time = time.time()
        
        iterator = IteratorClass(alphabet, combination_length)
        
        init_time = time.time() - start_time
        
        # Iterate through all combinations
        start_time = time.time()
        
        count = 0
        while iterator.hasNext():
            iterator.next()
            count += 1
        
        iteration_time = time.time() - start_time
        total_time = init_time + iteration_time
        
        print(f"    Initialization: {init_time:.3f}s")
        print(f"    Iteration: {iteration_time:.3f}s")
        print(f"    Total time: {total_time:.3f}s")
        print(f"    Combinations generated: {count}")
        print(f"    Rate: {count/total_time:.0f} combinations/second")
        print(f"    Correct count: {count == expected_total}")

def benchmark_memory_usage():
    """Benchmark memory usage of different approaches"""
    print("\n=== Benchmarking Memory Usage ===")
    
    test_case = ("abcdefghij", 4)  # Should generate C(10,4) = 210 combinations
    characters, k = test_case
    
    implementations = [
        ("Precompute", CombinationIteratorPrecompute),
        ("Lazy", CombinationIteratorLazy),
        ("Advanced", CombinationIteratorAdvanced)
    ]
    
    print(f"Memory usage test: '{characters}', k={k}")
    
    for name, IteratorClass in implementations:
        iterator = IteratorClass(characters, k)
        
        # Estimate memory usage
        if hasattr(iterator, 'combinations'):
            # Precompute approach
            memory_estimate = len(iterator.combinations) * k  # Approximate
            approach_type = "Pre-computed"
        elif hasattr(iterator, 'cache'):
            # Advanced approach with cache
            memory_estimate = len(iterator.cache) * k + k  # Cache + current state
            approach_type = "Cached"
        else:
            # Lazy approaches
            memory_estimate = k  # Just current state
            approach_type = "Lazy"
        
        print(f"  {name} ({approach_type}):")
        print(f"    Estimated memory: ~{memory_estimate} characters")
        
        # Test a few iterations to verify functionality
        sample_count = 0
        while iterator.hasNext() and sample_count < 3:
            iterator.next()
            sample_count += 1

def test_iterator_correctness():
    """Test iterator correctness with known cases"""
    print("\n=== Testing Iterator Correctness ===")
    
    # Test cases with known results
    test_cases = [
        ("abc", 1, ["a", "b", "c"]),
        ("abc", 2, ["ab", "ac", "bc"]),
        ("abc", 3, ["abc"]),
        ("abcd", 2, ["ab", "ac", "ad", "bc", "bd", "cd"])
    ]
    
    implementations = [
        ("Lazy", CombinationIteratorLazy),
        ("Bitmask", CombinationIteratorBitmask),
        ("Advanced", CombinationIteratorAdvanced)
    ]
    
    for characters, k, expected in test_cases:
        print(f"\nTest: '{characters}', k={k}")
        print(f"Expected: {expected}")
        
        for name, IteratorClass in implementations:
            iterator = IteratorClass(characters, k)
            
            results = []
            while iterator.hasNext():
                results.append(iterator.next())
            
            correct = results == expected
            print(f"  {name}: {results} - {'✓' if correct else '✗'}")

if __name__ == "__main__":
    test_combination_iterator_basic()
    test_combination_iterator_edge_cases()
    test_performance_comparison()
    test_advanced_features()
    demonstrate_applications()
    test_large_scale()
    stress_test_iterators()
    benchmark_memory_usage()
    test_iterator_correctness()

"""
Combination Iterator Design demonstrates key concepts:

Core Approaches:
1. Precompute - Generate all combinations upfront for O(1) access
2. Lazy - Generate combinations on-demand using next permutation logic
3. Recursive - Use recursive state machine for generation
4. Bitmask - Leverage bit manipulation for efficient generation
5. Advanced - Enhanced with caching, progress tracking, and utilities

Key Design Principles:
- Lexicographical ordering of combinations
- Memory vs time trade-offs in generation strategies
- Lazy evaluation for large combination spaces
- Iterator pattern implementation with hasNext/next

Performance Characteristics:
- Precompute: O(C(n,k)) space, O(1) next operation
- Lazy: O(k) space, O(k) amortized next operation
- Bitmask: O(1) space, O(n) next operation
- Advanced: Balanced with caching for repeated access patterns

Real-world Applications:
- Password and security key generation
- Feature selection in machine learning
- Test case generation for software testing
- Menu and product combination generation
- Genetic algorithm population initialization
- Distributed system configuration enumeration

The lazy generation approach provides the best balance
for most use cases, offering minimal memory usage
while maintaining reasonable performance for iteration.
"""
