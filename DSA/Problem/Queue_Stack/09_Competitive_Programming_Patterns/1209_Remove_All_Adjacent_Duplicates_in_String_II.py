"""
1209. Remove All Adjacent Duplicates in String II - Multiple Approaches
Difficulty: Medium

You are given a string s and an integer k. A k duplicate removal consists of choosing k adjacent and equal letters from s and removing them, causing the left and the right side of the deleted substring to concatenate together.

We repeatedly make k duplicate removals on s until we no longer can.

Return the final string after all such duplicate removals have been made. It is guaranteed that the answer is unique.
"""

from typing import List, Tuple

class RemoveAdjacentDuplicatesII:
    """Multiple approaches to remove k adjacent duplicates"""
    
    def removeDuplicates_stack_with_count(self, s: str, k: int) -> str:
        """
        Approach 1: Stack with Count (Optimal)
        
        Use stack to store characters with their counts.
        
        Time: O(n), Space: O(n)
        """
        stack = []  # [(char, count)]
        
        for char in s:
            if stack and stack[-1][0] == char:
                # Same character, increment count
                stack[-1] = (char, stack[-1][1] + 1)
                
                # Remove if count reaches k
                if stack[-1][1] == k:
                    stack.pop()
            else:
                # New character
                stack.append((char, 1))
        
        # Build result string
        result = []
        for char, count in stack:
            result.append(char * count)
        
        return ''.join(result)
    
    def removeDuplicates_stack_chars(self, s: str, k: int) -> str:
        """
        Approach 2: Stack of Characters
        
        Use stack to store individual characters and check for k duplicates.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        
        for char in s:
            stack.append(char)
            
            # Check if we have k consecutive same characters
            if len(stack) >= k and all(stack[-i] == char for i in range(1, k + 1)):
                # Remove k characters
                for _ in range(k):
                    stack.pop()
        
        return ''.join(stack)
    
    def removeDuplicates_two_pointers(self, s: str, k: int) -> str:
        """
        Approach 3: Two Pointers In-Place
        
        Use two pointers to modify string in place.
        
        Time: O(n), Space: O(1) excluding result
        """
        chars = list(s)
        counts = [0] * len(s)
        write = 0
        
        for read in range(len(s)):
            chars[write] = chars[read]
            
            if write > 0 and chars[write] == chars[write - 1]:
                counts[write] = counts[write - 1] + 1
            else:
                counts[write] = 1
            
            if counts[write] == k:
                write -= k
            else:
                write += 1
        
        return ''.join(chars[:write])
    
    def removeDuplicates_recursive(self, s: str, k: int) -> str:
        """
        Approach 4: Recursive Solution
        
        Recursively remove k duplicates until no more can be removed.
        
        Time: O(n²) worst case, Space: O(n) due to recursion
        """
        def remove_once(string: str) -> str:
            """Remove one occurrence of k consecutive duplicates"""
            i = 0
            while i < len(string):
                j = i
                # Find end of current character sequence
                while j < len(string) and string[j] == string[i]:
                    j += 1
                
                # If sequence length is >= k, remove k characters
                if j - i >= k:
                    # Remove k characters and return modified string
                    return string[:i] + string[i + k:]
                
                i = j
            
            return string
        
        prev = ""
        current = s
        
        # Keep removing until no changes
        while prev != current:
            prev = current
            current = remove_once(current)
        
        return current
    
    def removeDuplicates_iterative_multiple_passes(self, s: str, k: int) -> str:
        """
        Approach 5: Multiple Passes
        
        Make multiple passes until no more removals possible.
        
        Time: O(n²) worst case, Space: O(n)
        """
        while True:
            new_s = []
            i = 0
            removed = False
            
            while i < len(s):
                count = 1
                j = i + 1
                
                # Count consecutive same characters
                while j < len(s) and s[j] == s[i]:
                    count += 1
                    j += 1
                
                if count >= k:
                    # Remove k characters
                    remaining = count % k
                    if remaining > 0:
                        new_s.append(s[i] * remaining)
                    removed = True
                else:
                    # Keep all characters
                    new_s.append(s[i] * count)
                
                i = j
            
            s = ''.join(new_s)
            
            if not removed:
                break
        
        return s
    
    def removeDuplicates_deque(self, s: str, k: int) -> str:
        """
        Approach 6: Deque Implementation
        
        Use deque for efficient operations at both ends.
        
        Time: O(n), Space: O(n)
        """
        from collections import deque
        
        stack = deque()  # [(char, count)]
        
        for char in s:
            if stack and stack[-1][0] == char:
                # Increment count
                prev_char, count = stack.pop()
                new_count = count + 1
                
                if new_count < k:
                    stack.append((char, new_count))
                # If new_count == k, don't add back (remove k duplicates)
            else:
                # New character
                stack.append((char, 1))
        
        # Build result
        result = []
        for char, count in stack:
            result.append(char * count)
        
        return ''.join(result)


def test_remove_adjacent_duplicates_ii():
    """Test remove adjacent duplicates II algorithms"""
    solver = RemoveAdjacentDuplicatesII()
    
    test_cases = [
        ("abcd", 2, "abcd", "No duplicates"),
        ("deeedbbcccbdaa", 3, "aa", "Example 1"),
        ("pbbcggttciiippooaais", 2, "ps", "Example 2"),
        ("aaabbbccc", 3, "", "All removed"),
        ("aaabbbccc", 2, "abc", "Partial removal"),
        ("abccba", 2, "abba", "Middle removal"),
        ("aabbcc", 2, "", "All pairs removed"),
        ("aaabbaacd", 3, "cd", "Multiple groups"),
        ("a", 2, "a", "Single character"),
        ("aa", 2, "", "Exact k duplicates"),
        ("aaa", 2, "a", "More than k duplicates"),
        ("abcdef", 1, "", "k=1 removes all"),
        ("aabbbabaaba", 3, "aabbbabaaba", "No k consecutive"),
    ]
    
    algorithms = [
        ("Stack with Count", solver.removeDuplicates_stack_with_count),
        ("Stack of Chars", solver.removeDuplicates_stack_chars),
        ("Two Pointers", solver.removeDuplicates_two_pointers),
        ("Recursive", solver.removeDuplicates_recursive),
        ("Multiple Passes", solver.removeDuplicates_iterative_multiple_passes),
        ("Deque", solver.removeDuplicates_deque),
    ]
    
    print("=== Testing Remove Adjacent Duplicates II ===")
    
    for s, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input: s='{s}', k={k}")
        print(f"Expected: '{expected}'")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(s, k)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: '{result}'")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_stack_with_count_approach():
    """Demonstrate stack with count approach step by step"""
    print("\n=== Stack with Count Approach Step-by-Step Demo ===")
    
    s, k = "deeedbbcccbdaa", 3
    
    print(f"Input: s='{s}', k={k}")
    print("Strategy: Use stack to store (character, count) pairs")
    
    stack = []
    
    print(f"\nStep-by-step processing:")
    
    for i, char in enumerate(s):
        print(f"\nStep {i+1}: Processing '{char}'")
        print(f"  Current stack: {stack}")
        
        if stack and stack[-1][0] == char:
            # Same character, increment count
            old_count = stack[-1][1]
            new_count = old_count + 1
            stack[-1] = (char, new_count)
            
            print(f"  Same character as top, count: {old_count} -> {new_count}")
            
            # Remove if count reaches k
            if new_count == k:
                removed = stack.pop()
                print(f"  Count reached {k}, removed: {removed}")
        else:
            # New character
            stack.append((char, 1))
            print(f"  New character, added: ('{char}', 1)")
        
        print(f"  Stack after: {stack}")
    
    # Build result string
    result = []
    for char, count in stack:
        result.append(char * count)
    
    final_result = ''.join(result)
    print(f"\nFinal result: '{final_result}'")


def visualize_removal_process():
    """Visualize the removal process"""
    print("\n=== Removal Process Visualization ===")
    
    s, k = "aaabbbccc", 3
    
    print(f"Input: '{s}', k={k}")
    print("Visualizing character-by-character processing:")
    
    stack = []
    
    for i, char in enumerate(s):
        # Show current state
        current_string = ''.join(c * count for c, count in stack)
        print(f"\nStep {i+1}: Current result: '{current_string}'")
        print(f"Processing: '{char}'")
        
        if stack and stack[-1][0] == char:
            stack[-1] = (char, stack[-1][1] + 1)
            print(f"  Increment count: {stack[-1]}")
            
            if stack[-1][1] == k:
                removed = stack.pop()
                print(f"  Removed {k} '{removed[0]}' characters")
        else:
            stack.append((char, 1))
            print(f"  Added new character: {stack[-1]}")
        
        # Show stack state
        print(f"  Stack: {stack}")
    
    final_result = ''.join(c * count for c, count in stack)
    print(f"\nFinal result: '{final_result}'")


def demonstrate_competitive_programming_patterns():
    """Demonstrate competitive programming patterns"""
    print("\n=== Competitive Programming Patterns ===")
    
    solver = RemoveAdjacentDuplicatesII()
    
    # Pattern 1: Stack with auxiliary information
    print("1. Stack with Auxiliary Information:")
    print("   Store (character, count) pairs instead of individual characters")
    print("   Reduces space and improves efficiency")
    
    example1 = "aabbbaaa"
    k1 = 3
    result1 = solver.removeDuplicates_stack_with_count(example1, k1)
    print(f"   '{example1}' with k={k1} -> '{result1}'")
    
    # Pattern 2: Two pointers for in-place modification
    print(f"\n2. Two Pointers In-Place Modification:")
    print("   Use read/write pointers to modify array/string in place")
    print("   Saves space by avoiding extra data structures")
    
    example2 = "abbbaaca"
    k2 = 3
    result2 = solver.removeDuplicates_two_pointers(example2, k2)
    print(f"   '{example2}' with k={k2} -> '{result2}'")
    
    # Pattern 3: Amortized analysis
    print(f"\n3. Amortized Analysis:")
    print("   Each character is pushed and popped at most once")
    print("   Total operations: O(n) despite nested loops")
    
    # Pattern 4: State compression
    print(f"\n4. State Compression:")
    print("   Instead of storing k identical characters, store (char, count)")
    print("   Reduces memory usage and improves cache performance")
    
    # Pattern 5: Early termination optimization
    print(f"\n5. Early Termination:")
    print("   Stop processing as soon as no more removals are possible")
    print("   Useful for multiple-pass algorithms")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack with Count", "O(n)", "O(n)", "Each char processed once"),
        ("Stack of Chars", "O(n)", "O(n)", "Each char pushed/popped once"),
        ("Two Pointers", "O(n)", "O(1)", "In-place modification"),
        ("Recursive", "O(n²)", "O(n)", "Multiple string reconstructions"),
        ("Multiple Passes", "O(n²)", "O(n)", "Multiple passes over string"),
        ("Deque", "O(n)", "O(n)", "Similar to stack approach"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 65)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<8} | {space_comp:<8} | {notes}")
    
    print(f"\nStack with Count is optimal for most cases")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = RemoveAdjacentDuplicatesII()
    
    edge_cases = [
        ("", 2, "", "Empty string"),
        ("a", 1, "", "Single char, k=1"),
        ("a", 2, "a", "Single char, k>1"),
        ("aa", 2, "", "Exact k duplicates"),
        ("aaa", 2, "a", "k+1 duplicates"),
        ("aaaa", 2, "", "2k duplicates"),
        ("aaaaa", 2, "a", "2k+1 duplicates"),
        ("abcdef", 1, "", "All different, k=1"),
        ("aabbcc", 2, "", "All pairs"),
        ("aaabbbcccdddeee", 3, "ddd", "Multiple triplets"),
        ("abccba", 2, "abba", "Removal in middle"),
        ("abccbaccba", 2, "abaccba", "Multiple potential removals"),
    ]
    
    for s, k, expected, description in edge_cases:
        try:
            result = solver.removeDuplicates_stack_with_count(s, k)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | s='{s}', k={k} -> '{result}'")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def demonstrate_cascading_removals():
    """Demonstrate cascading removals"""
    print("\n=== Cascading Removals Demo ===")
    
    solver = RemoveAdjacentDuplicatesII()
    
    # Example where removal creates new duplicates
    s, k = "abccbaccba", 2
    
    print(f"Input: '{s}', k={k}")
    print("Demonstrating how removals can create new duplicates:")
    
    # Manual step-by-step to show cascading
    print(f"\nOriginal: '{s}'")
    print("Looking for 'cc' -> found at position 2-3")
    print("After removing 'cc': 'abbaccba'")
    print("Now looking for 'bb' -> found at position 1-2")
    print("After removing 'bb': 'aaccba'")
    print("Now looking for 'aa' -> found at position 0-1")
    print("After removing 'aa': 'ccba'")
    print("Now looking for 'cc' -> found at position 0-1")
    print("After removing 'cc': 'ba'")
    print("No more duplicates found")
    
    result = solver.removeDuplicates_stack_with_count(s, k)
    print(f"\nActual result: '{result}'")
    
    # Another example
    s2, k2 = "deeedbbcccbdaa", 3
    print(f"\nAnother example: '{s2}', k={k2}")
    result2 = solver.removeDuplicates_stack_with_count(s2, k2)
    print(f"Result: '{result2}'")
    
    print("\nKey insight: Stack-based approach handles cascading automatically")
    print("No need for multiple passes - each character is processed once")


def benchmark_approaches():
    """Benchmark different approaches"""
    import time
    import random
    import string
    
    approaches = [
        ("Stack with Count", RemoveAdjacentDuplicatesII().removeDuplicates_stack_with_count),
        ("Two Pointers", RemoveAdjacentDuplicatesII().removeDuplicates_two_pointers),
        ("Stack of Chars", RemoveAdjacentDuplicatesII().removeDuplicates_stack_chars),
    ]
    
    # Generate test strings
    def generate_test_string(length: int, alphabet_size: int = 5) -> str:
        """Generate string with potential duplicates"""
        chars = [random.choice(string.ascii_lowercase[:alphabet_size]) for _ in range(length)]
        return ''.join(chars)
    
    test_sizes = [100, 1000, 5000]
    k_values = [2, 3, 5]
    
    print(f"\n=== Performance Benchmark ===")
    
    for size in test_sizes:
        print(f"\nString length: {size}")
        
        for k in k_values:
            print(f"  k = {k}:")
            
            # Generate test string
            test_string = generate_test_string(size)
            
            for name, func in approaches:
                try:
                    start_time = time.time()
                    
                    # Run multiple times for better measurement
                    for _ in range(10):
                        func(test_string, k)
                    
                    end_time = time.time()
                    avg_time = (end_time - start_time) / 10
                    
                    print(f"    {name:20} | Avg time: {avg_time:.6f}s")
                    
                except Exception as e:
                    print(f"    {name:20} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_remove_adjacent_duplicates_ii()
    demonstrate_stack_with_count_approach()
    visualize_removal_process()
    demonstrate_competitive_programming_patterns()
    analyze_time_complexity()
    test_edge_cases()
    demonstrate_cascading_removals()
    benchmark_approaches()

"""
Remove All Adjacent Duplicates in String II demonstrates competitive
programming patterns with stack-based optimization, amortized analysis,
and efficient state management for string processing problems.
"""
