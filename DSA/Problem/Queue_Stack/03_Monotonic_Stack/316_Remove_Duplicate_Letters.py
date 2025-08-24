"""
316. Remove Duplicate Letters - Multiple Approaches
Difficulty: Medium

Given a string s, remove duplicate letters so that every letter appears exactly once. You must make sure your result is the smallest in lexicographical order among all possible results.
"""

from typing import Dict, Set
from collections import Counter

class RemoveDuplicateLetters:
    """Multiple approaches to remove duplicate letters"""
    
    def removeDuplicateLetters_stack_approach(self, s: str) -> str:
        """
        Approach 1: Monotonic Stack with Greedy Strategy
        
        Use stack to build lexicographically smallest result.
        
        Time: O(n), Space: O(1) - constant alphabet size
        """
        # Count frequency of each character
        count = Counter(s)
        
        stack = []
        in_stack = set()
        
        for char in s:
            # Decrease count for current character
            count[char] -= 1
            
            # Skip if character already in result
            if char in in_stack:
                continue
            
            # Pop characters that are lexicographically larger
            # and will appear later in the string
            while (stack and 
                   stack[-1] > char and 
                   count[stack[-1]] > 0):
                removed = stack.pop()
                in_stack.remove(removed)
            
            # Add current character
            stack.append(char)
            in_stack.add(char)
        
        return ''.join(stack)
    
    def removeDuplicateLetters_recursive(self, s: str) -> str:
        """
        Approach 2: Recursive Approach
        
        Recursively find smallest character and process remaining string.
        
        Time: O(n²), Space: O(n)
        """
        if not s:
            return ""
        
        # Count characters
        count = Counter(s)
        
        # Find the smallest character that appears in the entire remaining string
        pos = 0
        for i in range(len(s)):
            if s[i] < s[pos]:
                pos = i
            count[s[i]] -= 1
            if count[s[i]] == 0:
                break
        
        # Remove all occurrences of chosen character from remaining string
        remaining = s[pos + 1:].replace(s[pos], '')
        
        return s[pos] + self.removeDuplicateLetters_recursive(remaining)
    
    def removeDuplicateLetters_last_occurrence(self, s: str) -> str:
        """
        Approach 3: Last Occurrence Tracking
        
        Track last occurrence of each character for decision making.
        
        Time: O(n), Space: O(1)
        """
        # Find last occurrence of each character
        last_occurrence = {}
        for i, char in enumerate(s):
            last_occurrence[char] = i
        
        stack = []
        visited = set()
        
        for i, char in enumerate(s):
            # Skip if already processed
            if char in visited:
                continue
            
            # Pop characters that are greater and will appear later
            while (stack and 
                   stack[-1] > char and 
                   last_occurrence[stack[-1]] > i):
                removed = stack.pop()
                visited.remove(removed)
            
            stack.append(char)
            visited.add(char)
        
        return ''.join(stack)
    
    def removeDuplicateLetters_greedy_selection(self, s: str) -> str:
        """
        Approach 4: Greedy Character Selection
        
        Greedily select characters while maintaining lexicographical order.
        
        Time: O(n), Space: O(1)
        """
        remaining_count = Counter(s)
        result = []
        used = set()
        
        for char in s:
            remaining_count[char] -= 1
            
            if char in used:
                continue
            
            # Remove characters that are lexicographically larger
            # and have remaining occurrences
            while (result and 
                   result[-1] > char and 
                   remaining_count[result[-1]] > 0):
                removed = result.pop()
                used.remove(removed)
            
            result.append(char)
            used.add(char)
        
        return ''.join(result)
    
    def removeDuplicateLetters_position_based(self, s: str) -> str:
        """
        Approach 5: Position-based Decision Making
        
        Make decisions based on character positions.
        
        Time: O(n), Space: O(1)
        """
        # Build position map
        positions = {}
        for i, char in enumerate(s):
            if char not in positions:
                positions[char] = []
            positions[char].append(i)
        
        result = []
        used = set()
        
        for i, char in enumerate(s):
            if char in used:
                continue
            
            # Check if we should wait for a better position
            should_add = True
            
            for res_char in result:
                if (res_char > char and 
                    any(pos > i for pos in positions[res_char])):
                    should_add = False
                    break
            
            if should_add:
                # Remove larger characters that appear later
                while (result and 
                       result[-1] > char and 
                       any(pos > i for pos in positions[result[-1]])):
                    removed = result.pop()
                    used.remove(removed)
                
                result.append(char)
                used.add(char)
        
        return ''.join(result)
    
    def removeDuplicateLetters_iterative_improvement(self, s: str) -> str:
        """
        Approach 6: Iterative Improvement
        
        Iteratively improve the result by making local optimizations.
        
        Time: O(n²), Space: O(n)
        """
        # Start with all unique characters in order of first appearance
        seen = set()
        initial = []
        
        for char in s:
            if char not in seen:
                initial.append(char)
                seen.add(char)
        
        # Iteratively improve by swapping adjacent characters
        result = initial[:]
        improved = True
        
        while improved:
            improved = False
            
            for i in range(len(result) - 1):
                if result[i] > result[i + 1]:
                    # Check if we can swap
                    char1, char2 = result[i], result[i + 1]
                    
                    # Find positions in original string
                    pos1 = [j for j, c in enumerate(s) if c == char1]
                    pos2 = [j for j, c in enumerate(s) if c == char2]
                    
                    # Check if char2 can come before char1
                    if any(p2 < p1 for p1 in pos1 for p2 in pos2):
                        result[i], result[i + 1] = result[i + 1], result[i]
                        improved = True
                        break
        
        return ''.join(result)


def test_remove_duplicate_letters():
    """Test remove duplicate letters algorithms"""
    solver = RemoveDuplicateLetters()
    
    test_cases = [
        ("bcabc", "abc", "Example 1"),
        ("cbacdcbc", "acdb", "Example 2"),
        ("a", "a", "Single character"),
        ("aa", "a", "Duplicate characters"),
        ("abacaba", "abc", "Multiple duplicates"),
        ("ecbacba", "eacb", "Complex case"),
        ("bbcaac", "bac", "Multiple same chars"),
        ("abcabc", "abc", "Repeated pattern"),
        ("dcba", "dcba", "Decreasing order"),
        ("abcd", "abcd", "Already optimal"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.removeDuplicateLetters_stack_approach),
        ("Recursive", solver.removeDuplicateLetters_recursive),
        ("Last Occurrence", solver.removeDuplicateLetters_last_occurrence),
        ("Greedy Selection", solver.removeDuplicateLetters_greedy_selection),
        ("Position Based", solver.removeDuplicateLetters_position_based),
    ]
    
    print("=== Testing Remove Duplicate Letters ===")
    
    for s, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input: '{s}'")
        print(f"Expected: '{expected}'")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(s)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: '{result}'")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    s = "cbacdcbc"
    print(f"Processing string: '{s}'")
    
    count = Counter(s)
    print(f"Character counts: {dict(count)}")
    
    stack = []
    in_stack = set()
    
    for i, char in enumerate(s):
        print(f"\nStep {i+1}: Processing '{char}'")
        print(f"  Remaining counts: {dict(count)}")
        print(f"  Stack before: {stack}")
        print(f"  In stack: {in_stack}")
        
        # Decrease count
        count[char] -= 1
        print(f"  Decreased count for '{char}': {count[char]}")
        
        # Skip if already in stack
        if char in in_stack:
            print(f"  '{char}' already in stack, skipping")
            continue
        
        # Pop larger characters that appear later
        popped = []
        while (stack and 
               stack[-1] > char and 
               count[stack[-1]] > 0):
            removed = stack.pop()
            in_stack.remove(removed)
            popped.append(removed)
        
        if popped:
            print(f"  Popped: {popped} (larger than '{char}' and appear later)")
        
        # Add current character
        stack.append(char)
        in_stack.add(char)
        
        print(f"  Stack after: {stack}")
        print(f"  In stack: {in_stack}")
    
    result = ''.join(stack)
    print(f"\nFinal result: '{result}'")


def demonstrate_lexicographical_concept():
    """Demonstrate lexicographical ordering concept"""
    print("\n=== Lexicographical Ordering Concept ===")
    
    print("Lexicographical order is like dictionary order:")
    print("- 'a' < 'b' < 'c' < ... < 'z'")
    print("- For strings, compare character by character")
    print("- First differing character determines order")
    
    examples = [
        ("bcabc", ["abc", "bac", "bca", "cab", "cba"], "abc"),
        ("cbacdcbc", ["acdb", "cadb", "cdab", "dcab"], "acdb"),
    ]
    
    for s, possibilities, optimal in examples:
        print(f"\nString: '{s}'")
        print("Possible results (removing duplicates):")
        
        for i, possibility in enumerate(possibilities):
            marker = " ← optimal" if possibility == optimal else ""
            print(f"  {i+1}. '{possibility}'{marker}")
        
        print(f"Lexicographically smallest: '{optimal}'")


def visualize_decision_making():
    """Visualize decision making process"""
    print("\n=== Decision Making Visualization ===")
    
    s = "bcabc"
    print(f"String: '{s}'")
    print("Character positions:")
    
    positions = {}
    for i, char in enumerate(s):
        if char not in positions:
            positions[char] = []
        positions[char].append(i)
    
    for char in sorted(positions.keys()):
        print(f"  '{char}': positions {positions[char]}")
    
    print("\nProcessing decisions:")
    
    count = Counter(s)
    stack = []
    in_stack = set()
    
    for i, char in enumerate(s):
        count[char] -= 1
        
        if char in in_stack:
            print(f"Position {i}: '{char}' already used, skip")
            continue
        
        print(f"Position {i}: Consider '{char}'")
        
        # Check what we can remove
        can_remove = []
        for stack_char in reversed(stack):
            if stack_char > char and count[stack_char] > 0:
                can_remove.append(stack_char)
            else:
                break
        
        if can_remove:
            print(f"  Can remove: {can_remove} (larger and appear later)")
            for remove_char in can_remove:
                stack.remove(remove_char)
                in_stack.remove(remove_char)
        
        stack.append(char)
        in_stack.add(char)
        print(f"  Added '{char}', current result: {''.join(stack)}")
    
    print(f"\nFinal result: {''.join(stack)}")


def benchmark_remove_duplicate_letters():
    """Benchmark different approaches"""
    import time
    import random
    import string
    
    algorithms = [
        ("Stack Approach", RemoveDuplicateLetters().removeDuplicateLetters_stack_approach),
        ("Last Occurrence", RemoveDuplicateLetters().removeDuplicateLetters_last_occurrence),
        ("Greedy Selection", RemoveDuplicateLetters().removeDuplicateLetters_greedy_selection),
    ]
    
    # Generate test strings of different lengths
    def generate_string(length: int) -> str:
        # Create string with some duplicate characters
        chars = random.choices(string.ascii_lowercase[:10], k=length)
        return ''.join(chars)
    
    test_lengths = [100, 1000, 5000]
    
    print("\n=== Remove Duplicate Letters Performance Benchmark ===")
    
    for length in test_lengths:
        print(f"\n--- String Length: {length} ---")
        
        test_string = generate_string(length)
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(test_string)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s | Result length: {len(result)}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = RemoveDuplicateLetters()
    
    edge_cases = [
        ("", "", "Empty string"),
        ("a", "a", "Single character"),
        ("aa", "a", "Two same characters"),
        ("ab", "ab", "Two different characters"),
        ("ba", "ab", "Two characters, reverse order"),
        ("abc", "abc", "Already optimal"),
        ("cba", "abc", "Reverse alphabetical"),
        ("aaa", "a", "All same characters"),
        ("abcabc", "abc", "Repeated pattern"),
        ("zyxwvu", "uvwxyz", "Reverse alphabet subset"),
    ]
    
    for s, expected, description in edge_cases:
        try:
            result = solver.removeDuplicateLetters_stack_approach(s)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | '{s}' -> '{result}'")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_cases = [
        "bcabc",
        "cbacdcbc",
        "abacaba",
        "ecbacba",
    ]
    
    solver = RemoveDuplicateLetters()
    
    approaches = [
        ("Stack", solver.removeDuplicateLetters_stack_approach),
        ("Recursive", solver.removeDuplicateLetters_recursive),
        ("Last Occurrence", solver.removeDuplicateLetters_last_occurrence),
        ("Greedy", solver.removeDuplicateLetters_greedy_selection),
    ]
    
    for i, s in enumerate(test_cases):
        print(f"\nTest case {i+1}: '{s}'")
        
        results = {}
        
        for name, func in approaches:
            try:
                result = func(s)
                results[name] = result
                print(f"{name:15} | Result: '{result}'")
            except Exception as e:
                print(f"{name:15} | ERROR: {str(e)[:40]}")
        
        # Check consistency
        if results:
            first_result = list(results.values())[0]
            all_same = all(result == first_result for result in results.values())
            print(f"All approaches agree: {'✓' if all_same else '✗'}")


def analyze_greedy_strategy():
    """Analyze the greedy strategy"""
    print("\n=== Greedy Strategy Analysis ===")
    
    print("Key insights of the greedy approach:")
    print("1. Process characters left to right")
    print("2. For each character, decide whether to include it now")
    print("3. Remove previous characters if:")
    print("   - They are lexicographically larger")
    print("   - They appear again later in the string")
    print("4. This ensures lexicographically smallest result")
    
    print("\nWhy this works:")
    print("- We want smaller characters to appear as early as possible")
    print("- If we can remove a larger character and place it later, we should")
    print("- The stack maintains the current best prefix")
    
    s = "cbacdcbc"
    print(f"\nExample with '{s}':")
    
    count = Counter(s)
    decisions = []
    
    stack = []
    in_stack = set()
    
    for char in s:
        count[char] -= 1
        
        if char in in_stack:
            decisions.append(f"Skip '{char}' (already used)")
            continue
        
        removed = []
        while (stack and 
               stack[-1] > char and 
               count[stack[-1]] > 0):
            removed_char = stack.pop()
            in_stack.remove(removed_char)
            removed.append(removed_char)
        
        if removed:
            decisions.append(f"Remove {removed} to place '{char}' earlier")
        else:
            decisions.append(f"Add '{char}' (no conflicts)")
        
        stack.append(char)
        in_stack.add(char)
    
    for i, decision in enumerate(decisions):
        print(f"  {i+1}. {decision}")


def demonstrate_stack_invariant():
    """Demonstrate stack invariant property"""
    print("\n=== Stack Invariant Demonstration ===")
    
    s = "bcabc"
    print(f"Processing '{s}':")
    print("Stack invariant: Characters in stack are in lexicographical order")
    
    count = Counter(s)
    stack = []
    in_stack = set()
    
    for i, char in enumerate(s):
        count[char] -= 1
        
        if char in in_stack:
            continue
        
        print(f"\nStep {i+1}: Adding '{char}'")
        print(f"  Stack before: {stack}")
        
        # Check invariant before modification
        if len(stack) > 1:
            is_sorted = all(stack[j] <= stack[j+1] for j in range(len(stack)-1))
            print(f"  Invariant before: {'✓' if is_sorted else '✗'}")
        
        # Remove violating characters
        while (stack and 
               stack[-1] > char and 
               count[stack[-1]] > 0):
            removed = stack.pop()
            in_stack.remove(removed)
            print(f"    Removed '{removed}' (violates invariant)")
        
        stack.append(char)
        in_stack.add(char)
        
        print(f"  Stack after: {stack}")
        
        # Check invariant after modification
        if len(stack) > 1:
            is_sorted = all(stack[j] <= stack[j+1] for j in range(len(stack)-1))
            print(f"  Invariant after: {'✓' if is_sorted else '✗'}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack Approach", "O(n)", "O(1)", "Each char pushed/popped at most once"),
        ("Recursive", "O(n²)", "O(n)", "Recursive calls with string operations"),
        ("Last Occurrence", "O(n)", "O(1)", "Single pass with last occurrence map"),
        ("Greedy Selection", "O(n)", "O(1)", "Similar to stack approach"),
        ("Position Based", "O(n)", "O(1)", "Position tracking with decisions"),
        ("Iterative Improvement", "O(n²)", "O(n)", "Multiple improvement passes"),
    ]
    
    print(f"{'Approach':<25} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 75)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<25} | {time_comp:<8} | {space_comp:<8} | {notes}")


if __name__ == "__main__":
    test_remove_duplicate_letters()
    demonstrate_lexicographical_concept()
    demonstrate_stack_approach()
    visualize_decision_making()
    demonstrate_stack_invariant()
    analyze_greedy_strategy()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_remove_duplicate_letters()

"""
Remove Duplicate Letters demonstrates advanced monotonic stack applications
for lexicographical optimization, including greedy strategies and
multiple approaches for string manipulation with ordering constraints.
"""
