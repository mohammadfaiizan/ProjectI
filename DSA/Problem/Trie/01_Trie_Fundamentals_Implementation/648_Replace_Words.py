"""
648. Replace Words - Multiple Approaches
Difficulty: Medium

In English, we have a concept called root, which can be followed by some other word 
to form another longer word - let's call this word successor. For example, when the 
root "an" is followed by the successor word "other", we can form a new word "another".

Given a dictionary consisting of many roots and a sentence consisting of words 
separated by spaces, you need to replace all the successors in the sentence with 
the root forming it. If a successor can be replaced by more than one root, replace 
it with the root that has the shortest length.

LeetCode Problem: https://leetcode.com/problems/replace-words/

Example:
Input: dictionary = ["cat","bat","rat"], sentence = "the cattle was rattled by the battery"
Output: "the cat was rat by the bat"
"""

from typing import List, Dict, Set
from collections import defaultdict

class TrieNode:
    """Trie node for root storage"""
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_root: bool = False
        self.root_word: str = ""  # Store the actual root word

class Solution:
    
    def replaceWords1(self, dictionary: List[str], sentence: str) -> str:
        """
        Approach 1: Trie-based Solution
        
        Build trie from dictionary roots and find shortest root for each word.
        
        Time: O(D + S) where D is total chars in dictionary, S is sentence length
        Space: O(D) for trie storage
        """
        # Build trie
        root = TrieNode()
        
        for word in dictionary:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_root = True
            node.root_word = word
        
        def find_root(word: str) -> str:
            """Find shortest root for given word"""
            node = root
            for i, char in enumerate(word):
                if char not in node.children:
                    return word  # No root found
                node = node.children[char]
                if node.is_root:
                    return node.root_word  # Found root
            return word  # No root found
        
        # Process sentence
        words = sentence.split()
        result_words = []
        
        for word in words:
            result_words.append(find_root(word))
        
        return " ".join(result_words)
    
    def replaceWords2(self, dictionary: List[str], sentence: str) -> str:
        """
        Approach 2: Set-based with Prefix Checking
        
        Use set for O(1) lookup and check all possible prefixes.
        
        Time: O(D + S * W) where W is average word length
        Space: O(D) for dictionary set
        """
        root_set = set(dictionary)
        words = sentence.split()
        result_words = []
        
        for word in words:
            # Find shortest root by checking all prefixes
            replaced = False
            for i in range(1, len(word) + 1):
                prefix = word[:i]
                if prefix in root_set:
                    result_words.append(prefix)
                    replaced = True
                    break
            
            if not replaced:
                result_words.append(word)
        
        return " ".join(result_words)
    
    def replaceWords3(self, dictionary: List[str], sentence: str) -> str:
        """
        Approach 3: Sorted Dictionary with Binary Search
        
        Sort dictionary and use binary search for prefix matching.
        
        Time: O(D log D + S * W * log D)
        Space: O(D)
        """
        dictionary.sort()  # Sort for binary search
        words = sentence.split()
        result_words = []
        
        def find_shortest_root(word: str) -> str:
            """Find shortest root using binary search"""
            for i in range(1, len(word) + 1):
                prefix = word[:i]
                
                # Binary search for prefix
                left, right = 0, len(dictionary)
                while left < right:
                    mid = (left + right) // 2
                    if dictionary[mid] < prefix:
                        left = mid + 1
                    else:
                        right = mid
                
                # Check if found exact match
                if left < len(dictionary) and dictionary[left] == prefix:
                    return prefix
            
            return word
        
        for word in words:
            result_words.append(find_shortest_root(word))
        
        return " ".join(result_words)
    
    def replaceWords4(self, dictionary: List[str], sentence: str) -> str:
        """
        Approach 4: Optimized Trie with Early Termination
        
        Enhanced trie with optimizations for common cases.
        
        Time: O(D + S)
        Space: O(D)
        """
        # Build optimized trie
        root = TrieNode()
        
        # Sort dictionary by length for shortest root preference
        dictionary.sort(key=len)
        
        for word in dictionary:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            
            # Only mark as root if no shorter root exists
            if not node.is_root:
                node.is_root = True
                node.root_word = word
        
        def find_root_optimized(word: str) -> str:
            """Find root with early termination"""
            node = root
            for char in word:
                if char not in node.children:
                    return word  # No root found
                node = node.children[char]
                if node.is_root:
                    return node.root_word  # Return first (shortest) root found
            return word
        
        # Process sentence
        words = sentence.split()
        return " ".join(find_root_optimized(word) for word in words)
    
    def replaceWords5(self, dictionary: List[str], sentence: str) -> str:
        """
        Approach 5: Hash Map with Length Optimization
        
        Group dictionary by length and check from shortest to longest.
        
        Time: O(D + S * W)
        Space: O(D)
        """
        # Group roots by length
        roots_by_length = defaultdict(set)
        max_length = 0
        
        for root in dictionary:
            roots_by_length[len(root)].add(root)
            max_length = max(max_length, len(root))
        
        def find_shortest_root(word: str) -> str:
            """Find shortest root by checking lengths in order"""
            for length in range(1, min(len(word), max_length) + 1):
                prefix = word[:length]
                if prefix in roots_by_length[length]:
                    return prefix
            return word
        
        words = sentence.split()
        return " ".join(find_shortest_root(word) for word in words)
    
    def replaceWords6(self, dictionary: List[str], sentence: str) -> str:
        """
        Approach 6: Suffix Array based Approach
        
        Advanced approach using suffix array for pattern matching.
        
        Time: O(D log D + S * log D)
        Space: O(D)
        """
        # Create suffix array from dictionary
        dictionary.sort()
        words = sentence.split()
        result_words = []
        
        def binary_search_prefix(word: str) -> str:
            """Use binary search to find matching prefix"""
            best_root = word
            
            for i in range(1, len(word) + 1):
                prefix = word[:i]
                
                # Binary search for exact match
                left, right = 0, len(dictionary) - 1
                
                while left <= right:
                    mid = (left + right) // 2
                    if dictionary[mid] == prefix:
                        if len(prefix) < len(best_root):
                            best_root = prefix
                        break
                    elif dictionary[mid] < prefix:
                        left = mid + 1
                    else:
                        right = mid - 1
            
            return best_root
        
        for word in words:
            result_words.append(binary_search_prefix(word))
        
        return " ".join(result_words)


def test_basic_cases():
    """Test basic replace words functionality"""
    print("=== Testing Basic Cases ===")
    
    solution = Solution()
    
    test_cases = [
        (["cat", "bat", "rat"], "the cattle was rattled by the battery", 
         "the cat was rat by the bat"),
        (["a", "b", "c"], "aadsfasf absbs bbab cadsfafs", 
         "a a b c"),
        (["a", "aa", "aaa"], "a aa a aaaa aaa aaa aaa aaaaaa bbb baba ababa", 
         "a a a a a a a a bbb baba a"),
        (["catt", "cat", "bat", "rat"], "the cattle was rattled by the battery", 
         "the cat was rat by the bat"),
        ([], "hello world", "hello world"),  # Empty dictionary
        (["hello"], "", ""),  # Empty sentence
    ]
    
    approaches = [
        ("Trie-based", solution.replaceWords1),
        ("Set-based", solution.replaceWords2),
        ("Binary Search", solution.replaceWords3),
        ("Optimized Trie", solution.replaceWords4),
        ("Length Groups", solution.replaceWords5),
        ("Suffix Array", solution.replaceWords6),
    ]
    
    for i, (dictionary, sentence, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Dictionary: {dictionary}")
        print(f"Sentence: '{sentence}'")
        print(f"Expected: '{expected}'")
        
        for name, method in approaches:
            try:
                # Create copy of dictionary since some methods modify it
                dict_copy = dictionary.copy()
                result = method(dict_copy, sentence)
                status = "✓" if result == expected else "✗"
                print(f"  {name:15}: '{result}' {status}")
            except Exception as e:
                print(f"  {name:15}: Error - {e}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Case 1: Word is shorter than root
        (["application"], "app", "app"),
        
        # Case 2: Multiple roots of same length
        (["cat", "bat"], "category", "cat"),
        
        # Case 3: Root is same as word
        (["hello"], "hello world", "hello world"),
        
        # Case 4: No matching roots
        (["xyz"], "hello world", "hello world"),
        
        # Case 5: Single character roots
        (["a", "b"], "apple banana", "a b"),
        
        # Case 6: Very long words
        (["test"], "testing" * 100, "test" + "ing" * 100),
        
        # Case 7: Overlapping roots
        (["a", "ab", "abc"], "abcdef", "a"),
    ]
    
    approaches = [
        ("Trie", solution.replaceWords1),
        ("Set", solution.replaceWords2),
        ("Optimized", solution.replaceWords4),
    ]
    
    for i, (dictionary, sentence, expected) in enumerate(edge_cases):
        print(f"\nEdge Case {i+1}: {dictionary} | '{sentence[:50]}{'...' if len(sentence) > 50 else ''}'")
        
        for name, method in approaches:
            try:
                dict_copy = dictionary.copy()
                result = method(dict_copy, sentence)
                status = "✓" if result == expected else "✗"
                result_display = result[:50] + "..." if len(result) > 50 else result
                print(f"  {name:10}: '{result_display}' {status}")
            except Exception as e:
                print(f"  {name:10}: Error - {e}")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import time
    import random
    import string
    
    solution = Solution()
    
    # Generate test data
    def generate_dictionary(n: int, avg_length: int) -> List[str]:
        dictionary = []
        for _ in range(n):
            length = max(1, avg_length + random.randint(-2, 2))
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            dictionary.append(word)
        return list(set(dictionary))  # Remove duplicates
    
    def generate_sentence(n_words: int, avg_length: int) -> str:
        words = []
        for _ in range(n_words):
            length = max(1, avg_length + random.randint(-3, 3))
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
        return " ".join(words)
    
    test_scenarios = [
        ("Small", generate_dictionary(10, 4), generate_sentence(20, 6)),
        ("Medium", generate_dictionary(100, 5), generate_sentence(200, 7)),
        ("Large", generate_dictionary(1000, 6), generate_sentence(1000, 8)),
    ]
    
    approaches = [
        ("Trie", solution.replaceWords1),
        ("Set", solution.replaceWords2),
        ("Optimized Trie", solution.replaceWords4),
        ("Length Groups", solution.replaceWords5),
    ]
    
    for scenario_name, dictionary, sentence in test_scenarios:
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Dictionary: {len(dictionary)} roots, Sentence: {len(sentence.split())} words")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            # Run multiple times for better measurement
            for _ in range(5):
                dict_copy = dictionary.copy()
                result = method(dict_copy, sentence)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 5
            
            print(f"  {approach_name:15}: {avg_time*1000:.2f} ms")


def demonstrate_trie_construction():
    """Demonstrate trie construction process"""
    print("\n=== Trie Construction Demo ===")
    
    dictionary = ["cat", "cats", "dog", "dodge", "door", "doors"]
    print(f"Building trie for dictionary: {dictionary}")
    
    # Build trie step by step
    root = TrieNode()
    
    for word in dictionary:
        print(f"\nInserting '{word}':")
        node = root
        path = ""
        
        for char in word:
            path += char
            if char not in node.children:
                node.children[char] = TrieNode()
                print(f"  Created node for '{path}'")
            else:
                print(f"  Reusing node for '{path}'")
            node = node.children[char]
        
        node.is_root = True
        node.root_word = word
        print(f"  Marked '{word}' as root")
    
    # Test word replacement
    test_sentence = "the cats were dodging through doors"
    print(f"\nTesting sentence: '{test_sentence}'")
    
    words = test_sentence.split()
    result_words = []
    
    for word in words:
        print(f"\nProcessing '{word}':")
        node = root
        found_root = None
        
        for i, char in enumerate(word):
            if char not in node.children:
                print(f"  No path for '{char}' at position {i}")
                break
            node = node.children[char]
            current_prefix = word[:i+1]
            print(f"  Following path: '{current_prefix}'")
            
            if node.is_root:
                found_root = node.root_word
                print(f"  Found root: '{found_root}'")
                break
        
        result_word = found_root if found_root else word
        result_words.append(result_word)
        print(f"  Result: '{word}' -> '{result_word}'")
    
    final_result = " ".join(result_words)
    print(f"\nFinal result: '{final_result}'")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: Text Processing and Normalization
    print("1. Text Normalization:")
    
    # Common word stems for text normalization
    stems = ["run", "walk", "talk", "work", "help", "play", "look", "call"]
    text = "running walking talking working helping playing looking calling"
    
    result = solution.replaceWords1(stems, text)
    print(f"   Original: {text}")
    print(f"   Normalized: {result}")
    
    # Application 2: Code Refactoring
    print("\n2. Code Variable Refactoring:")
    
    # Short variable names to replace long ones
    short_vars = ["btn", "txt", "img", "usr", "pwd", "cfg"]
    code_line = "button_submit text_input image_logo user_name password_field config_file"
    
    result = solution.replaceWords1(short_vars, code_line)
    print(f"   Original: {code_line}")
    print(f"   Refactored: {result}")
    
    # Application 3: URL Path Simplification
    print("\n3. URL Path Simplification:")
    
    # Base paths
    base_paths = ["/api", "/admin", "/user", "/static"]
    url_paths = "/api/v1/users /admin/dashboard /user/profile /static/css/main"
    
    result = solution.replaceWords1(base_paths, url_paths)
    print(f"   Original: {url_paths}")
    print(f"   Simplified: {result}")
    
    # Application 4: Database Table Prefix Normalization
    print("\n4. Database Table Normalization:")
    
    # Table prefixes
    prefixes = ["usr", "ord", "prod", "cat"]
    table_names = "user_profiles order_details product_inventory category_mapping"
    
    result = solution.replaceWords1(prefixes, table_names)
    print(f"   Original: {table_names}")
    print(f"   Normalized: {result}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Trie-based",
         "Time: O(D + S) - D=dict chars, S=sentence length",
         "Space: O(D) - trie storage"),
        
        ("Set-based",
         "Time: O(D + S*W) - W=avg word length",
         "Space: O(D) - dictionary set"),
        
        ("Binary Search",
         "Time: O(D log D + S*W*log D) - sorting + search",
         "Space: O(D) - sorted dictionary"),
        
        ("Optimized Trie",
         "Time: O(D log D + S) - sort dict + process",
         "Space: O(D) - trie storage"),
        
        ("Length Groups",
         "Time: O(D + S*W) - group + search",
         "Space: O(D) - grouped dictionary"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nRecommendations:")
    print(f"  • Use Trie-based for large dictionaries with frequent queries")
    print(f"  • Use Set-based for small dictionaries or one-time processing")
    print(f"  • Use Length Groups when dictionary has many short roots")
    print(f"  • Use Optimized Trie when shortest root preference is critical")


if __name__ == "__main__":
    test_basic_cases()
    test_edge_cases()
    benchmark_approaches()
    demonstrate_trie_construction()
    demonstrate_real_world_applications()
    analyze_complexity()

"""
648. Replace Words demonstrates multiple approaches for root-based word replacement:

1. Trie-based - Efficient prefix tree for finding roots
2. Set-based - Simple hash set with prefix checking
3. Binary Search - Sorted dictionary with binary search
4. Optimized Trie - Enhanced trie with shortest root preference
5. Length Groups - Dictionary grouped by root length for optimization
6. Suffix Array - Advanced pattern matching approach

Each approach offers different trade-offs in terms of preprocessing time,
query efficiency, and memory usage, suitable for various text processing scenarios.
"""
