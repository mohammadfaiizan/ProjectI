"""
394. Decode String - Multiple Approaches
Difficulty: Medium

Given an encoded string, return its decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; there are no extra white spaces, square brackets are well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat times, k, so that there won't be any confusion.
"""

from typing import List

class DecodeString:
    """Multiple approaches to decode string with nested brackets"""
    
    def decodeString_stack_approach(self, s: str) -> str:
        """
        Approach 1: Stack-based Decoding
        
        Use stack to handle nested brackets and repetitions.
        
        Time: O(n * m) where m is max repetition, Space: O(n)
        """
        stack = []
        current_string = ""
        current_num = 0
        
        for char in s:
            if char.isdigit():
                current_num = current_num * 10 + int(char)
            elif char == '[':
                # Push current state to stack
                stack.append((current_string, current_num))
                current_string = ""
                current_num = 0
            elif char == ']':
                # Pop from stack and decode
                prev_string, num = stack.pop()
                current_string = prev_string + current_string * num
            else:
                current_string += char
        
        return current_string
    
    def decodeString_recursive_approach(self, s: str) -> str:
        """
        Approach 2: Recursive Decoding
        
        Use recursion to handle nested structures.
        
        Time: O(n * m), Space: O(n)
        """
        def decode_helper(index: int) -> tuple:
            """Returns (decoded_string, next_index)"""
            result = ""
            num = 0
            
            while index < len(s):
                char = s[index]
                
                if char.isdigit():
                    num = num * 10 + int(char)
                elif char == '[':
                    # Recursively decode the content inside brackets
                    decoded_part, next_index = decode_helper(index + 1)
                    result += decoded_part * num
                    num = 0
                    index = next_index
                elif char == ']':
                    return result, index
                else:
                    result += char
                
                index += 1
            
            return result, index
        
        decoded, _ = decode_helper(0)
        return decoded
    
    def decodeString_two_stacks_approach(self, s: str) -> str:
        """
        Approach 3: Two Stacks Approach
        
        Use separate stacks for numbers and strings.
        
        Time: O(n * m), Space: O(n)
        """
        num_stack = []
        string_stack = []
        current_string = ""
        current_num = 0
        
        for char in s:
            if char.isdigit():
                current_num = current_num * 10 + int(char)
            elif char == '[':
                # Push current state
                num_stack.append(current_num)
                string_stack.append(current_string)
                current_num = 0
                current_string = ""
            elif char == ']':
                # Pop and decode
                num = num_stack.pop()
                prev_string = string_stack.pop()
                current_string = prev_string + current_string * num
            else:
                current_string += char
        
        return current_string
    
    def decodeString_iterative_parsing(self, s: str) -> str:
        """
        Approach 4: Iterative Parsing with State Machine
        
        Use state machine to parse the string.
        
        Time: O(n * m), Space: O(n)
        """
        stack = []
        i = 0
        
        while i < len(s):
            if s[i].isdigit():
                # Parse number
                num = 0
                while i < len(s) and s[i].isdigit():
                    num = num * 10 + int(s[i])
                    i += 1
                stack.append(num)
            elif s[i] == '[':
                stack.append('[')
                i += 1
            elif s[i] == ']':
                # Find matching '[' and decode
                temp = []
                while stack and stack[-1] != '[':
                    temp.append(stack.pop())
                
                # Remove '['
                if stack:
                    stack.pop()
                
                # Get the number
                num = stack.pop() if stack and isinstance(stack[-1], int) else 1
                
                # Decode the string
                decoded = ''.join(reversed(temp)) * num
                stack.append(decoded)
                i += 1
            else:
                # Regular character
                stack.append(s[i])
                i += 1
        
        return ''.join(str(item) for item in stack)
    
    def decodeString_regex_approach(self, s: str) -> str:
        """
        Approach 5: Regular Expression Approach
        
        Use regex to find and replace patterns.
        
        Time: O(n * m), Space: O(n)
        """
        import re
        
        # Keep decoding until no more patterns found
        while '[' in s:
            # Find innermost brackets
            pattern = r'(\d+)\[([^\[\]]*)\]'
            
            def replace_match(match):
                num = int(match.group(1))
                content = match.group(2)
                return content * num
            
            s = re.sub(pattern, replace_match, s)
        
        return s
    
    def decodeString_deque_approach(self, s: str) -> str:
        """
        Approach 6: Deque-based Approach
        
        Use deque for efficient operations.
        
        Time: O(n * m), Space: O(n)
        """
        from collections import deque
        
        stack = deque()
        current_string = ""
        current_num = 0
        
        for char in s:
            if char.isdigit():
                current_num = current_num * 10 + int(char)
            elif char == '[':
                stack.append((current_string, current_num))
                current_string = ""
                current_num = 0
            elif char == ']':
                prev_string, num = stack.pop()
                current_string = prev_string + current_string * num
            else:
                current_string += char
        
        return current_string
    
    def decodeString_optimized_memory(self, s: str) -> str:
        """
        Approach 7: Memory-optimized Approach
        
        Minimize memory usage during decoding.
        
        Time: O(n * m), Space: O(n)
        """
        def build_string(chars: List[str], multiplier: int) -> List[str]:
            """Build repeated string efficiently"""
            if multiplier == 1:
                return chars
            
            result = []
            for _ in range(multiplier):
                result.extend(chars)
            return result
        
        stack = []
        current_chars = []
        current_num = 0
        
        for char in s:
            if char.isdigit():
                current_num = current_num * 10 + int(char)
            elif char == '[':
                stack.append((current_chars, current_num))
                current_chars = []
                current_num = 0
            elif char == ']':
                prev_chars, num = stack.pop()
                repeated_chars = build_string(current_chars, num)
                current_chars = prev_chars + repeated_chars
            else:
                current_chars.append(char)
        
        return ''.join(current_chars)

def test_decode_string():
    """Test string decoding algorithms"""
    solver = DecodeString()
    
    test_cases = [
        ("3[a]2[bc]", "aaabcbc", "Simple repetition"),
        ("2[abc]3[cd]ef", "abcabccdcdcdef", "Multiple patterns"),
        ("abc3[cd]xyz", "abccdcdcdxyz", "Mixed content"),
        ("2[2[y]pq4[2[jk]e1[f]]]ef", "yypqjkjkef1fjkjkef1fjkjkef1fjkjkef1fyypqjkjkef1fjkjkef1fjkjkef1fjkjkef1fef", "Deeply nested"),
        ("3[a2[c]]", "accaccacc", "Nested brackets"),
        ("2[abc]3[cd]ef", "abcabccdcdcdef", "Sequential patterns"),
        ("abc", "abc", "No brackets"),
        ("10[a]", "aaaaaaaaaa", "Double digit"),
        ("2[b3[a]]", "baaabaaab", "Nested with different numbers"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.decodeString_stack_approach),
        ("Recursive Approach", solver.decodeString_recursive_approach),
        ("Two Stacks", solver.decodeString_two_stacks_approach),
        ("Iterative Parsing", solver.decodeString_iterative_parsing),
        ("Regex Approach", solver.decodeString_regex_approach),
        ("Deque Approach", solver.decodeString_deque_approach),
        ("Memory Optimized", solver.decodeString_optimized_memory),
    ]
    
    print("=== Testing Decode String ===")
    
    for encoded, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input:    '{encoded}'")
        print(f"Expected: '{expected}'")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(encoded)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: '{result[:50]}{'...' if len(result) > 50 else ''}'")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")

def demonstrate_decoding_process():
    """Demonstrate step-by-step decoding process"""
    print("\n=== String Decoding Step-by-Step Demo ===")
    
    s = "2[a3[b]]"
    print(f"Decoding: {s}")
    
    stack = []
    current_string = ""
    current_num = 0
    
    for i, char in enumerate(s):
        print(f"\nStep {i+1}: Processing '{char}'")
        
        if char.isdigit():
            current_num = current_num * 10 + int(char)
            print(f"  -> Building number: {current_num}")
        elif char == '[':
            stack.append((current_string, current_num))
            print(f"  -> Push to stack: ('{current_string}', {current_num})")
            current_string = ""
            current_num = 0
        elif char == ']':
            prev_string, num = stack.pop()
            decoded = current_string * num
            current_string = prev_string + decoded
            print(f"  -> Pop from stack: ('{prev_string}', {num})")
            print(f"  -> Decode: '{current_string}' * {num} = '{decoded}'")
            print(f"  -> Result: '{prev_string}' + '{decoded}' = '{current_string}'")
        else:
            current_string += char
            print(f"  -> Add character: '{current_string}'")
        
        print(f"  -> Current state: string='{current_string}', num={current_num}, stack={stack}")
    
    print(f"\nFinal result: '{current_string}'")

def benchmark_decode_string():
    """Benchmark different decoding approaches"""
    import time
    
    def generate_nested_string(depth: int, base_length: int) -> str:
        """Generate nested encoded string for testing"""
        if depth == 0:
            return 'a' * base_length
        
        inner = generate_nested_string(depth - 1, base_length)
        return f"2[{inner}]"
    
    algorithms = [
        ("Stack Approach", DecodeString().decodeString_stack_approach),
        ("Recursive Approach", DecodeString().decodeString_recursive_approach),
        ("Two Stacks", DecodeString().decodeString_two_stacks_approach),
        ("Memory Optimized", DecodeString().decodeString_optimized_memory),
    ]
    
    test_cases = [
        ("Simple repetition", "10[abc]"),
        ("Nested 2 levels", "3[2[ab]]"),
        ("Nested 3 levels", "2[3[2[xy]]]"),
        ("Complex pattern", "2[abc]3[2[de]f]"),
    ]
    
    print("\n=== Decode String Performance Benchmark ===")
    
    for case_name, test_string in test_cases:
        print(f"\n--- {case_name}: '{test_string}' ---")
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                # Run multiple times for better measurement
                for _ in range(100):
                    result = alg_func(test_string)
                
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")

def test_edge_cases():
    """Test edge cases for string decoding"""
    print("\n=== Testing Edge Cases ===")
    
    solver = DecodeString()
    
    edge_cases = [
        ("", "", "Empty string"),
        ("a", "a", "Single character"),
        ("1[a]", "a", "Single repetition"),
        ("0[a]", "", "Zero repetition"),
        ("100[a]", "a" * 100, "Large number"),
        ("2[2[2[a]]]", "aaaaaaaa", "Deep nesting"),
        ("a2[b]c", "abbc", "Mixed pattern"),
        ("2[]", "", "Empty brackets"),
    ]
    
    for encoded, expected, description in edge_cases:
        try:
            result = solver.decodeString_stack_approach(encoded)
            status = "✓" if result == expected else "✗"
            print(f"{description:20} | {status} | '{encoded}' -> '{result[:20]}{'...' if len(result) > 20 else ''}'")
        except Exception as e:
            print(f"{description:20} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_decode_string()
    demonstrate_decoding_process()
    test_edge_cases()
    benchmark_decode_string()

"""
Decode String demonstrates multiple approaches to handle nested
bracket structures including stack-based parsing, recursion,
regex processing, and memory-optimized techniques.
"""
