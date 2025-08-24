"""
1047. Remove All Adjacent Duplicates In String - Multiple Approaches
Difficulty: Easy

You are given a string s consisting of lowercase English letters. A duplicate removal consists of choosing two adjacent and equal characters and removing them.

We repeatedly make duplicate removals on s until we no longer can.

Return the final string after all such duplicate removals have been made.
"""

from typing import List

class RemoveAdjacentDuplicates:
    """Multiple approaches to remove adjacent duplicates"""
    
    def removeDuplicates_stack_approach(self, s: str) -> str:
        """
        Approach 1: Stack-based Removal
        
        Use stack to track characters and remove duplicates.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        
        for char in s:
            if stack and stack[-1] == char:
                # Remove duplicate
                stack.pop()
            else:
                # Add new character
                stack.append(char)
        
        return ''.join(stack)
    
    def removeDuplicates_two_pointer_approach(self, s: str) -> str:
        """
        Approach 2: Two Pointer In-place Modification
        
        Use two pointers to modify string in-place.
        
        Time: O(n), Space: O(n) for result string
        """
        chars = list(s)
        write_index = 0
        
        for read_index in range(len(chars)):
            if write_index > 0 and chars[write_index - 1] == chars[read_index]:
                # Remove duplicate by moving write pointer back
                write_index -= 1
            else:
                # Keep character
                chars[write_index] = chars[read_index]
                write_index += 1
        
        return ''.join(chars[:write_index])
    
    def removeDuplicates_recursive_approach(self, s: str) -> str:
        """
        Approach 3: Recursive Removal
        
        Use recursion to remove duplicates.
        
        Time: O(n²) worst case, Space: O(n)
        """
        def remove_once(string: str) -> str:
            """Remove one pass of adjacent duplicates"""
            result = []
            i = 0
            
            while i < len(string):
                if i + 1 < len(string) and string[i] == string[i + 1]:
                    # Skip both duplicate characters
                    while i + 1 < len(string) and string[i] == string[i + 1]:
                        i += 2
                    if i < len(string):
                        i += 1
                else:
                    result.append(string[i])
                    i += 1
            
            return ''.join(result)
        
        prev_length = len(s)
        result = remove_once(s)
        
        # Keep removing until no more changes
        while len(result) != prev_length:
            prev_length = len(result)
            result = remove_once(result)
        
        return result
    
    def removeDuplicates_iterative_approach(self, s: str) -> str:
        """
        Approach 4: Iterative String Building
        
        Build result string iteratively.
        
        Time: O(n), Space: O(n)
        """
        result = []
        
        for char in s:
            if result and result[-1] == char:
                result.pop()
            else:
                result.append(char)
        
        return ''.join(result)
    
    def removeDuplicates_deque_approach(self, s: str) -> str:
        """
        Approach 5: Deque-based Approach
        
        Use deque for efficient operations.
        
        Time: O(n), Space: O(n)
        """
        from collections import deque
        
        dq = deque()
        
        for char in s:
            if dq and dq[-1] == char:
                dq.pop()
            else:
                dq.append(char)
        
        return ''.join(dq)
    
    def removeDuplicates_string_builder_approach(self, s: str) -> str:
        """
        Approach 6: String Builder Simulation
        
        Simulate StringBuilder behavior.
        
        Time: O(n), Space: O(n)
        """
        class StringBuilder:
            def __init__(self):
                self.chars = []
            
            def append(self, char: str):
                self.chars.append(char)
            
            def delete_last(self):
                if self.chars:
                    self.chars.pop()
            
            def peek_last(self) -> str:
                return self.chars[-1] if self.chars else None
            
            def to_string(self) -> str:
                return ''.join(self.chars)
            
            def is_empty(self) -> bool:
                return len(self.chars) == 0
        
        sb = StringBuilder()
        
        for char in s:
            if not sb.is_empty() and sb.peek_last() == char:
                sb.delete_last()
            else:
                sb.append(char)
        
        return sb.to_string()
    
    def removeDuplicates_optimized_stack(self, s: str) -> str:
        """
        Approach 7: Optimized Stack with Early Termination
        
        Optimize stack approach with early termination checks.
        
        Time: O(n), Space: O(n)
        """
        if not s:
            return ""
        
        stack = [s[0]]
        
        for i in range(1, len(s)):
            char = s[i]
            
            if stack and stack[-1] == char:
                stack.pop()
                # Early termination check
                if not stack and i == len(s) - 1:
                    return ""
            else:
                stack.append(char)
        
        return ''.join(stack)

def test_remove_duplicates():
    """Test adjacent duplicates removal algorithms"""
    solver = RemoveAdjacentDuplicates()
    
    test_cases = [
        ("abbaca", "ca", "Basic removal"),
        ("azxxzy", "ay", "Multiple removals"),
        ("aabbcc", "", "All pairs removed"),
        ("abccba", "", "Nested removals"),
        ("a", "a", "Single character"),
        ("", "", "Empty string"),
        ("abc", "abc", "No duplicates"),
        ("aab", "b", "Partial removal"),
        ("abba", "", "Palindrome removal"),
        ("abccbddeea", "a", "Complex case"),
        ("aaaaa", "a", "Odd count"),
        ("aaaa", "", "Even count"),
        ("abcddcba", "", "Symmetric removal"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.removeDuplicates_stack_approach),
        ("Two Pointer", solver.removeDuplicates_two_pointer_approach),
        ("Recursive Approach", solver.removeDuplicates_recursive_approach),
        ("Iterative Approach", solver.removeDuplicates_iterative_approach),
        ("Deque Approach", solver.removeDuplicates_deque_approach),
        ("String Builder", solver.removeDuplicates_string_builder_approach),
        ("Optimized Stack", solver.removeDuplicates_optimized_stack),
    ]
    
    print("=== Testing Remove Adjacent Duplicates ===")
    
    for input_str, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input:    '{input_str}'")
        print(f"Expected: '{expected}'")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(input_str)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: '{result}'")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")

def demonstrate_removal_process():
    """Demonstrate step-by-step removal process"""
    print("\n=== Adjacent Duplicates Removal Step-by-Step Demo ===")
    
    s = "abccba"
    print(f"Removing duplicates from: '{s}'")
    
    stack = []
    
    for i, char in enumerate(s):
        print(f"\nStep {i+1}: Processing '{char}'")
        
        if stack and stack[-1] == char:
            removed = stack.pop()
            print(f"  -> Found duplicate '{char}', removing '{removed}' from stack")
        else:
            stack.append(char)
            print(f"  -> Adding '{char}' to stack")
        
        print(f"  -> Current stack: {stack}")
        print(f"  -> Current string: '{''.join(stack)}'")
    
    result = ''.join(stack)
    print(f"\nFinal result: '{result}'")

def visualize_removal_steps():
    """Visualize the removal process"""
    print("\n=== Visualizing Removal Process ===")
    
    test_cases = [
        "abbaca",
        "abccba", 
        "azxxzy"
    ]
    
    for s in test_cases:
        print(f"\nProcessing: '{s}'")
        stack = []
        steps = []
        
        for char in s:
            if stack and stack[-1] == char:
                removed = stack.pop()
                steps.append(f"Remove '{char}' (duplicate)")
            else:
                stack.append(char)
                steps.append(f"Add '{char}'")
            
            steps.append(f"Stack: {''.join(stack)}")
        
        for i, step in enumerate(steps):
            if i % 2 == 0:
                print(f"  {step}")
            else:
                print(f"    -> {step}")
        
        print(f"  Final: '{''.join(stack)}'")

def benchmark_remove_duplicates():
    """Benchmark different removal approaches"""
    import time
    import random
    import string
    
    def generate_test_string(length: int, duplicate_prob: float = 0.3) -> str:
        """Generate test string with controlled duplicate probability"""
        chars = []
        
        for _ in range(length):
            if chars and random.random() < duplicate_prob:
                # Add duplicate of last character
                chars.append(chars[-1])
            else:
                # Add random character
                chars.append(random.choice(string.ascii_lowercase[:10]))
        
        return ''.join(chars)
    
    algorithms = [
        ("Stack Approach", RemoveAdjacentDuplicates().removeDuplicates_stack_approach),
        ("Two Pointer", RemoveAdjacentDuplicates().removeDuplicates_two_pointer_approach),
        ("Iterative Approach", RemoveAdjacentDuplicates().removeDuplicates_iterative_approach),
        ("Optimized Stack", RemoveAdjacentDuplicates().removeDuplicates_optimized_stack),
    ]
    
    string_lengths = [1000, 5000, 10000]
    
    print("\n=== Remove Duplicates Performance Benchmark ===")
    
    for length in string_lengths:
        print(f"\n--- String Length: {length} ---")
        test_strings = [generate_test_string(length) for _ in range(5)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            for test_str in test_strings:
                try:
                    result = alg_func(test_str)
                except:
                    pass  # Skip errors for benchmark
            
            end_time = time.time()
            
            print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s")

def test_edge_cases():
    """Test edge cases for duplicate removal"""
    print("\n=== Testing Edge Cases ===")
    
    solver = RemoveAdjacentDuplicates()
    
    edge_cases = [
        ("", "", "Empty string"),
        ("a", "a", "Single character"),
        ("aa", "", "Two same characters"),
        ("ab", "ab", "Two different characters"),
        ("aaa", "a", "Three same characters"),
        ("aaaa", "", "Four same characters"),
        ("abab", "abab", "Alternating pattern"),
        ("aabbcc", "", "All pairs"),
        ("abcabc", "abcabc", "No adjacent duplicates"),
        ("aabbaabb", "", "Multiple nested pairs"),
    ]
    
    for input_str, expected, description in edge_cases:
        result = solver.removeDuplicates_stack_approach(input_str)
        status = "✓" if result == expected else "✗"
        print(f"{description:25} | {status} | '{input_str}' -> '{result}'")

if __name__ == "__main__":
    test_remove_duplicates()
    demonstrate_removal_process()
    visualize_removal_steps()
    test_edge_cases()
    benchmark_remove_duplicates()

"""
Remove All Adjacent Duplicates demonstrates stack-based string processing
including two-pointer techniques, recursive approaches, and optimized
implementations for character removal and string manipulation.
"""
