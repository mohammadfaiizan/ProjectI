"""
1268. Search Suggestions System - Multiple Approaches
Difficulty: Medium

You are given an array of strings products and a string searchWord.

Design a system that suggests at most three product names from products after each 
character of searchWord is typed. Suggested products should have common prefix with 
searchWord. If there are more than three products with a common prefix return the 
three lexicographically smallest products.

Return a list of lists of the suggested products after each character of searchWord is typed.

LeetCode Problem: https://leetcode.com/problems/search-suggestions-system/

Example:
Input: products = ["mobile","mouse","moneypot","monitor","mousepad"], searchWord = "mouse"
Output: [["mobile","moneypot","monitor"],["mobile","moneypot","monitor"],["mouse","mousepad"],["mouse","mousepad"],["mouse","mousepad"]]
"""

from typing import List
import bisect

class TrieNode:
    """Trie node for autocomplete suggestions"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.suggestions = []  # Store up to 3 lexicographically smallest suggestions

class Solution:
    
    def suggestedProducts1(self, products: List[str], searchWord: str) -> List[List[str]]:
        """
        Approach 1: Trie with DFS Collection
        
        Build trie and collect suggestions using DFS.
        
        Time: O(sum of product lengths + |searchWord| * 3 * max_product_length)
        Space: O(sum of product lengths)
        """
        # Build trie
        root = TrieNode()
        
        for product in products:
            node = root
            for char in product:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
        
        def dfs_collect(node: TrieNode, prefix: str, suggestions: List[str]) -> None:
            """Collect suggestions using DFS"""
            if len(suggestions) >= 3:
                return
            
            if node.is_end:
                suggestions.append(prefix)
            
            # Traverse children in lexicographical order
            for char in sorted(node.children.keys()):
                dfs_collect(node.children[char], prefix + char, suggestions)
        
        result = []
        node = root
        prefix = ""
        
        for char in searchWord:
            prefix += char
            
            if char in node.children:
                node = node.children[char]
                suggestions = []
                dfs_collect(node, prefix, suggestions)
                result.append(suggestions[:3])
            else:
                # No more suggestions possible
                result.append([])
                node = None  # Mark as invalid
                break
        
        # Fill remaining with empty lists if node became None
        while len(result) < len(searchWord):
            result.append([])
        
        return result
    
    def suggestedProducts2(self, products: List[str], searchWord: str) -> List[List[str]]:
        """
        Approach 2: Optimized Trie with Pre-computed Suggestions
        
        Store top 3 suggestions at each trie node.
        
        Time: O(sum of product lengths + |searchWord|)
        Space: O(sum of product lengths)
        """
        # Sort products first for lexicographical order
        products.sort()
        
        # Build trie with suggestions
        root = TrieNode()
        
        for product in products:
            node = root
            for char in product:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                
                # Add product to suggestions if less than 3
                if len(node.suggestions) < 3:
                    node.suggestions.append(product)
            
            node.is_end = True
        
        result = []
        node = root
        
        for char in searchWord:
            if node and char in node.children:
                node = node.children[char]
                result.append(node.suggestions[:])
            else:
                result.append([])
                node = None  # No more valid suggestions
        
        return result
    
    def suggestedProducts3(self, products: List[str], searchWord: str) -> List[List[str]]:
        """
        Approach 3: Binary Search Approach
        
        Sort products and use binary search for each prefix.
        
        Time: O(n log n + |searchWord| * n * max_product_length)
        Space: O(1) excluding output
        """
        products.sort()
        result = []
        
        for i in range(len(searchWord)):
            prefix = searchWord[:i+1]
            suggestions = []
            
            # Find first product that starts with prefix
            left = 0
            right = len(products)
            
            while left < right:
                mid = (left + right) // 2
                if products[mid] < prefix:
                    left = mid + 1
                else:
                    right = mid
            
            # Collect up to 3 products starting from left
            start_idx = left
            for j in range(start_idx, min(start_idx + 3, len(products))):
                if j < len(products) and products[j].startswith(prefix):
                    suggestions.append(products[j])
                else:
                    break
            
            result.append(suggestions)
        
        return result
    
    def suggestedProducts4(self, products: List[str], searchWord: str) -> List[List[str]]:
        """
        Approach 4: Optimized Binary Search with Bisect
        
        Use Python's bisect module for efficient binary search.
        
        Time: O(n log n + |searchWord| * log n * max_product_length)
        Space: O(1)
        """
        products.sort()
        result = []
        
        for i in range(len(searchWord)):
            prefix = searchWord[:i+1]
            
            # Find insertion point for prefix
            start_idx = bisect.bisect_left(products, prefix)
            
            suggestions = []
            for j in range(start_idx, min(start_idx + 3, len(products))):
                if j < len(products) and products[j].startswith(prefix):
                    suggestions.append(products[j])
                else:
                    break
            
            result.append(suggestions)
        
        return result
    
    def suggestedProducts5(self, products: List[str], searchWord: str) -> List[List[str]]:
        """
        Approach 5: Two Pointers Optimization
        
        Maintain left and right pointers to valid range.
        
        Time: O(n log n + |searchWord| * n)
        Space: O(1)
        """
        products.sort()
        result = []
        left, right = 0, len(products) - 1
        
        for i in range(len(searchWord)):
            char = searchWord[i]
            
            # Narrow down left pointer
            while left <= right and (len(products[left]) <= i or products[left][i] != char):
                left += 1
            
            # Narrow down right pointer
            while left <= right and (len(products[right]) <= i or products[right][i] != char):
                right -= 1
            
            # Collect suggestions
            suggestions = []
            for j in range(left, min(left + 3, right + 1)):
                suggestions.append(products[j])
            
            result.append(suggestions)
        
        return result
    
    def suggestedProducts6(self, products: List[str], searchWord: str) -> List[List[str]]:
        """
        Approach 6: Prefix Tree with Heap
        
        Use min-heap to maintain top 3 suggestions efficiently.
        
        Time: O(sum of product lengths + |searchWord| * 3)
        Space: O(sum of product lengths)
        """
        import heapq
        
        # Group products by prefix
        prefix_map = {}
        
        for product in products:
            for i in range(1, len(product) + 1):
                prefix = product[:i]
                if prefix not in prefix_map:
                    prefix_map[prefix] = []
                prefix_map[prefix].append(product)
        
        # Sort suggestions for each prefix and keep only top 3
        for prefix in prefix_map:
            prefix_map[prefix].sort()
            prefix_map[prefix] = prefix_map[prefix][:3]
        
        result = []
        
        for i in range(len(searchWord)):
            prefix = searchWord[:i+1]
            suggestions = prefix_map.get(prefix, [])
            result.append(suggestions)
        
        return result


def test_basic_cases():
    """Test basic functionality"""
    print("=== Testing Basic Cases ===")
    
    solution = Solution()
    
    test_cases = [
        # LeetCode example
        (["mobile","mouse","moneypot","monitor","mousepad"], "mouse",
         [["mobile","moneypot","monitor"],["mobile","moneypot","monitor"],["mouse","mousepad"],["mouse","mousepad"],["mouse","mousepad"]]),
        
        # Simple case
        (["havana"], "havana",
         [["havana"],["havana"],["havana"],["havana"],["havana"],["havana"]]),
        
        # No matches
        (["mobile","mouse","moneypot","monitor","mousepad"], "xyz",
         [[],[],[],[],[],[],[],[],[],[]]),
        
        # Partial matches
        (["apple", "app", "application"], "app",
         [["app","apple","application"],["app","apple","application"],["app","apple","application"]]),
        
        # More than 3 matches
        (["app1", "app2", "app3", "app4", "app5"], "app",
         [["app1","app2","app3"],["app1","app2","app3"],["app1","app2","app3"]]),
    ]
    
    approaches = [
        ("Trie DFS", solution.suggestedProducts1),
        ("Optimized Trie", solution.suggestedProducts2),
        ("Binary Search", solution.suggestedProducts3),
        ("Bisect Search", solution.suggestedProducts4),
        ("Two Pointers", solution.suggestedProducts5),
        ("Prefix Map", solution.suggestedProducts6),
    ]
    
    for i, (products, searchWord, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: products={products[:3]}{'...' if len(products)>3 else ''}, searchWord='{searchWord}'")
        print(f"Expected: {expected}")
        
        for name, method in approaches:
            try:
                result = method(products[:], searchWord)
                status = "✓" if result == expected else "✗"
                print(f"  {name:15}: {result} {status}")
            except Exception as e:
                print(f"  {name:15}: Error - {e}")


def demonstrate_trie_construction():
    """Demonstrate trie construction with suggestions"""
    print("\n=== Trie Construction Demo ===")
    
    products = ["mobile", "mouse", "moneypot", "monitor"]
    products.sort()
    
    print(f"Products (sorted): {products}")
    
    # Build trie step by step
    root = TrieNode()
    
    print(f"\nBuilding trie with suggestions:")
    
    for product in products:
        print(f"\nInserting '{product}':")
        node = root
        
        for i, char in enumerate(product):
            if char not in node.children:
                node.children[char] = TrieNode()
                print(f"  Created node for '{char}' at prefix '{product[:i+1]}'")
            
            node = node.children[char]
            
            # Add to suggestions if less than 3
            if len(node.suggestions) < 3:
                node.suggestions.append(product)
                print(f"  Added '{product}' to suggestions at prefix '{product[:i+1]}'")
                print(f"  Current suggestions: {node.suggestions}")
    
    # Test search process
    searchWord = "mo"
    print(f"\nSearching for '{searchWord}':")
    
    node = root
    for i, char in enumerate(searchWord):
        prefix = searchWord[:i+1]
        print(f"\nStep {i+1}: Looking for '{char}' (prefix: '{prefix}')")
        
        if char in node.children:
            node = node.children[char]
            print(f"  Found '{char}'")
            print(f"  Suggestions: {node.suggestions}")
        else:
            print(f"  '{char}' not found - no more suggestions")
            break


def demonstrate_search_process():
    """Demonstrate the search process step by step"""
    print("\n=== Search Process Demo ===")
    
    products = ["mobile", "mouse", "moneypot", "monitor", "mousepad"]
    searchWord = "mous"
    
    print(f"Products: {products}")
    print(f"Search word: '{searchWord}'")
    
    # Show binary search approach
    products.sort()
    print(f"Sorted products: {products}")
    
    for i in range(len(searchWord)):
        prefix = searchWord[:i+1]
        print(f"\nStep {i+1}: Searching for prefix '{prefix}'")
        
        # Find starting position
        left = 0
        right = len(products)
        
        while left < right:
            mid = (left + right) // 2
            if products[mid] < prefix:
                left = mid + 1
            else:
                right = mid
        
        start_idx = left
        print(f"  Binary search found start index: {start_idx}")
        
        # Collect suggestions
        suggestions = []
        for j in range(start_idx, min(start_idx + 3, len(products))):
            if j < len(products) and products[j].startswith(prefix):
                suggestions.append(products[j])
                print(f"    Added: '{products[j]}'")
            else:
                print(f"    '{products[j] if j < len(products) else 'N/A'}' doesn't start with '{prefix}'")
                break
        
        print(f"  Final suggestions: {suggestions}")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import time
    import random
    import string
    
    solution = Solution()
    
    # Generate test data
    def generate_products(n: int, avg_length: int) -> List[str]:
        products = []
        for _ in range(n):
            length = max(1, avg_length + random.randint(-2, 2))
            product = ''.join(random.choices(string.ascii_lowercase, k=length))
            products.append(product)
        return list(set(products))  # Remove duplicates
    
    def generate_search_word(products: List[str], length: int) -> str:
        # Create search word that might have matches
        if products and random.random() < 0.7:  # 70% chance to use existing prefix
            base_product = random.choice(products)
            return base_product[:min(length, len(base_product))]
        else:
            return ''.join(random.choices(string.ascii_lowercase, k=length))
    
    test_scenarios = [
        ("Small", generate_products(50, 6), 4),
        ("Medium", generate_products(200, 8), 6),
        ("Large", generate_products(1000, 10), 8),
    ]
    
    approaches = [
        ("Optimized Trie", solution.suggestedProducts2),
        ("Binary Search", solution.suggestedProducts3),
        ("Two Pointers", solution.suggestedProducts5),
    ]
    
    for scenario_name, products, search_length in test_scenarios:
        searchWord = generate_search_word(products, search_length)
        
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Products: {len(products)}, Search word: '{searchWord}' (length {len(searchWord)})")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            # Run multiple times for better measurement
            for _ in range(10):
                result = method(products[:], searchWord)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            
            print(f"  {approach_name:15}: {avg_time*1000:.2f}ms")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: E-commerce search
    print("1. E-commerce Product Search:")
    
    ecommerce_products = [
        "laptop", "laptop stand", "laptop bag", "laptop charger",
        "smartphone", "smartphone case", "smartphone charger",
        "tablet", "tablet case", "tablet pen",
        "headphones", "headphones wireless", "headphones gaming"
    ]
    
    user_query = "laptop"
    
    print(f"   Available products: {len(ecommerce_products)} items")
    print(f"   User types: '{user_query}'")
    
    suggestions = solution.suggestedProducts2(ecommerce_products, user_query)
    
    for i, char_suggestions in enumerate(suggestions):
        prefix = user_query[:i+1]
        print(f"     After typing '{prefix}': {char_suggestions}")
    
    # Application 2: Code editor autocomplete
    print(f"\n2. Code Editor Autocomplete:")
    
    code_keywords = [
        "function", "function_call", "func_parameter",
        "variable", "var_name", "var_type",
        "class", "class_method", "class_property",
        "import", "import_from", "include"
    ]
    
    partial_code = "func"
    
    print(f"   Available keywords: {code_keywords}")
    print(f"   Developer types: '{partial_code}'")
    
    code_suggestions = solution.suggestedProducts2(code_keywords, partial_code)
    
    for i, suggestions in enumerate(code_suggestions):
        prefix = partial_code[:i+1]
        print(f"     After '{prefix}': {suggestions}")
    
    # Application 3: Location search
    print(f"\n3. Location/Address Autocomplete:")
    
    locations = [
        "new york", "new jersey", "new mexico", "new hampshire",
        "california", "canada", "cambridge", "carolina",
        "texas", "tennessee", "tokyo", "toronto"
    ]
    
    location_query = "new"
    
    print(f"   Available locations: {locations}")
    print(f"   User searches for: '{location_query}'")
    
    location_suggestions = solution.suggestedProducts2(locations, location_query)
    
    for i, suggestions in enumerate(location_suggestions):
        prefix = location_query[:i+1]
        print(f"     After '{prefix}': {suggestions}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Empty products
        ([], "search", "Empty products list"),
        
        # Empty search word
        (["product1", "product2"], "", "Empty search word"),
        
        # Single product
        (["single"], "single", "Single product"),
        
        # Search longer than all products
        (["a", "ab"], "abc", "Search longer than products"),
        
        # All products have same prefix
        (["test1", "test2", "test3", "test4"], "test", "All same prefix"),
        
        # No matching products
        (["apple", "banana"], "xyz", "No matches"),
        
        # Products with different lengths
        (["a", "aa", "aaa", "aaaa"], "aa", "Different lengths"),
        
        # Case sensitivity
        (["Apple", "apple"], "app", "Case sensitivity"),
    ]
    
    for products, searchWord, description in edge_cases:
        print(f"\n{description}:")
        print(f"  Products: {products}")
        print(f"  Search: '{searchWord}'")
        
        try:
            result = solution.suggestedProducts2(products, searchWord)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Trie DFS",
         "Time: O(sum(product_lengths) + |searchWord| * 3 * max_length)",
         "Space: O(sum(product_lengths))"),
        
        ("Optimized Trie",
         "Time: O(sum(product_lengths) + |searchWord|)",
         "Space: O(sum(product_lengths))"),
        
        ("Binary Search",
         "Time: O(n*log(n) + |searchWord| * n * max_length)",
         "Space: O(1)"),
        
        ("Two Pointers",
         "Time: O(n*log(n) + |searchWord| * n)",
         "Space: O(1)"),
        
        ("Prefix Map",
         "Time: O(sum(product_lengths))",
         "Space: O(sum(product_lengths))"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nRecommendations:")
    print(f"  • Use Optimized Trie for frequent searches with same product set")
    print(f"  • Use Binary Search for one-time searches or memory constraints")
    print(f"  • Use Two Pointers for online/streaming scenarios")
    print(f"  • Use Prefix Map for small product sets with many searches")


if __name__ == "__main__":
    test_basic_cases()
    demonstrate_trie_construction()
    demonstrate_search_process()
    benchmark_approaches()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_complexity()

"""
1268. Search Suggestions System demonstrates multiple autocomplete approaches:

1. Trie with DFS - Build trie and collect suggestions using depth-first search
2. Optimized Trie - Pre-compute top 3 suggestions at each trie node
3. Binary Search - Sort products and use binary search for each prefix
4. Bisect Search - Leverage Python's bisect module for efficient searching
5. Two Pointers - Maintain valid range with left and right pointers
6. Prefix Map - Pre-compute all prefix mappings for fast lookup

Each approach offers different trade-offs between preprocessing time,
search efficiency, and memory usage for autocomplete systems.
"""
