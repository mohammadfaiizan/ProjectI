"""
71. Simplify Path - Multiple Approaches
Difficulty: Medium

Given a string path, which is an absolute path (starting with a slash '/') to a file or directory in a Unix-style file system, convert it to the simplified canonical path.

In a Unix-style file system, a period '.' refers to the current directory, a double period '..' refers to the directory up a level, and any multiple consecutive slashes (i.e. '//') are treated as a single slash '/'. For this problem, any other format of periods such as '...' are treated as file/directory names.

The canonical path should have the following format:
- The path starts with a single slash '/'.
- Any two directories are separated by a single slash '/'.
- The path does not end with a trailing '/' unless the path is the root directory.
- The path only contains the directories on the path from the root directory to the target file or directory (i.e., no period '.' or double period '..')

Return the simplified canonical path.
"""

from typing import List

class SimplifyPath:
    """Multiple approaches to simplify Unix-style paths"""
    
    def simplifyPath_stack_approach(self, path: str) -> str:
        """
        Approach 1: Stack-based Path Simplification
        
        Use stack to handle directory navigation.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        components = path.split('/')
        
        for component in components:
            if component == '' or component == '.':
                # Skip empty components and current directory
                continue
            elif component == '..':
                # Go up one directory if possible
                if stack:
                    stack.pop()
            else:
                # Valid directory name
                stack.append(component)
        
        # Build the simplified path
        return '/' + '/'.join(stack)
    
    def simplifyPath_iterative_approach(self, path: str) -> str:
        """
        Approach 2: Iterative String Processing
        
        Process path character by character.
        
        Time: O(n), Space: O(n)
        """
        result = []
        i = 0
        n = len(path)
        
        while i < n:
            if path[i] == '/':
                # Skip multiple slashes
                while i < n and path[i] == '/':
                    i += 1
                
                if i < n:
                    # Extract directory name
                    start = i
                    while i < n and path[i] != '/':
                        i += 1
                    
                    component = path[start:i]
                    
                    if component == '.':
                        # Current directory, do nothing
                        continue
                    elif component == '..':
                        # Parent directory
                        if result:
                            result.pop()
                    else:
                        # Valid directory
                        result.append(component)
            else:
                i += 1
        
        return '/' + '/'.join(result)
    
    def simplifyPath_regex_approach(self, path: str) -> str:
        """
        Approach 3: Regular Expression Approach
        
        Use regex to clean and split path.
        
        Time: O(n), Space: O(n)
        """
        import re
        
        # Remove multiple slashes and split
        cleaned_path = re.sub(r'/+', '/', path)
        components = [comp for comp in cleaned_path.split('/') if comp]
        
        stack = []
        
        for component in components:
            if component == '.':
                continue
            elif component == '..':
                if stack:
                    stack.pop()
            else:
                stack.append(component)
        
        return '/' + '/'.join(stack)
    
    def simplifyPath_deque_approach(self, path: str) -> str:
        """
        Approach 4: Deque-based Approach
        
        Use deque for efficient operations.
        
        Time: O(n), Space: O(n)
        """
        from collections import deque
        
        stack = deque()
        components = path.split('/')
        
        for component in components:
            if component and component != '.':
                if component == '..':
                    if stack:
                        stack.pop()
                else:
                    stack.append(component)
        
        return '/' + '/'.join(stack)
    
    def simplifyPath_two_pass_approach(self, path: str) -> str:
        """
        Approach 5: Two-pass Approach
        
        First pass: normalize, Second pass: simplify.
        
        Time: O(n), Space: O(n)
        """
        # First pass: normalize slashes
        normalized = []
        i = 0
        while i < len(path):
            if path[i] == '/':
                normalized.append('/')
                # Skip consecutive slashes
                while i < len(path) and path[i] == '/':
                    i += 1
            else:
                normalized.append(path[i])
                i += 1
        
        normalized_path = ''.join(normalized)
        
        # Second pass: simplify using stack
        stack = []
        components = normalized_path.split('/')
        
        for component in components:
            if component == '' or component == '.':
                continue
            elif component == '..':
                if stack:
                    stack.pop()
            else:
                stack.append(component)
        
        return '/' + '/'.join(stack)
    
    def simplifyPath_recursive_approach(self, path: str) -> str:
        """
        Approach 6: Recursive Approach
        
        Use recursion to process path components.
        
        Time: O(n), Space: O(n)
        """
        def process_components(components: List[str], index: int, stack: List[str]) -> List[str]:
            if index >= len(components):
                return stack
            
            component = components[index]
            
            if component == '' or component == '.':
                # Skip empty and current directory
                return process_components(components, index + 1, stack)
            elif component == '..':
                # Go up one directory
                if stack:
                    stack.pop()
                return process_components(components, index + 1, stack)
            else:
                # Valid directory
                stack.append(component)
                return process_components(components, index + 1, stack)
        
        components = path.split('/')
        result_stack = process_components(components, 0, [])
        
        return '/' + '/'.join(result_stack)
    
    def simplifyPath_functional_approach(self, path: str) -> str:
        """
        Approach 7: Functional Programming Approach
        
        Use functional programming concepts.
        
        Time: O(n), Space: O(n)
        """
        from functools import reduce
        
        def process_component(stack, component):
            if component == '' or component == '.':
                return stack
            elif component == '..':
                return stack[:-1] if stack else stack
            else:
                return stack + [component]
        
        components = path.split('/')
        result_stack = reduce(process_component, components, [])
        
        return '/' + '/'.join(result_stack)

def test_simplify_path():
    """Test path simplification algorithms"""
    solver = SimplifyPath()
    
    test_cases = [
        ("/home/", "/home", "Trailing slash removal"),
        ("/../", "/", "Root parent directory"),
        ("/home//foo/", "/home/foo", "Multiple slashes"),
        ("/a/./b/../../c/", "/c", "Complex navigation"),
        ("/a/../../b/../c//.//", "/c", "Very complex case"),
        ("/a//b////c/d//././/..", "/a/b/c", "Mixed operations"),
        ("/..", "/", "Root parent"),
        ("/...", "/...", "Triple dots as directory"),
        ("/a/b/c", "/a/b/c", "Simple path"),
        ("/", "/", "Root directory"),
        ("/home/user/../documents/./file.txt", "/home/documents/file.txt", "Real file path"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.simplifyPath_stack_approach),
        ("Iterative Approach", solver.simplifyPath_iterative_approach),
        ("Regex Approach", solver.simplifyPath_regex_approach),
        ("Deque Approach", solver.simplifyPath_deque_approach),
        ("Two Pass Approach", solver.simplifyPath_two_pass_approach),
        ("Recursive Approach", solver.simplifyPath_recursive_approach),
        ("Functional Approach", solver.simplifyPath_functional_approach),
    ]
    
    print("=== Testing Simplify Path ===")
    
    for path, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input:    '{path}'")
        print(f"Expected: '{expected}'")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(path)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: '{result}'")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")

def demonstrate_path_simplification():
    """Demonstrate step-by-step path simplification"""
    print("\n=== Path Simplification Step-by-Step Demo ===")
    
    path = "/a/./b/../../c/"
    print(f"Original path: {path}")
    
    components = path.split('/')
    print(f"Split components: {components}")
    
    stack = []
    
    for i, component in enumerate(components):
        print(f"\nStep {i+1}: Processing '{component}'")
        
        if component == '' or component == '.':
            print(f"  -> Skip (empty or current directory)")
        elif component == '..':
            if stack:
                popped = stack.pop()
                print(f"  -> Go up one level, removed '{popped}'")
            else:
                print(f"  -> Already at root, cannot go up")
        else:
            stack.append(component)
            print(f"  -> Add directory '{component}'")
        
        print(f"  -> Current stack: {stack}")
    
    result = '/' + '/'.join(stack)
    print(f"\nFinal simplified path: {result}")

def benchmark_path_simplification():
    """Benchmark different path simplification approaches"""
    import time
    import random
    
    def generate_complex_path(length: int) -> str:
        """Generate complex path for testing"""
        components = ['/']
        
        for _ in range(length):
            choice = random.choice(['dir', '..', '.', '//'])
            if choice == 'dir':
                components.append(f'dir{random.randint(1, 100)}')
            elif choice == '..':
                components.append('..')
            elif choice == '.':
                components.append('.')
            else:  # '//'
                components.append('//')
            
            components.append('/')
        
        return ''.join(components)
    
    algorithms = [
        ("Stack Approach", SimplifyPath().simplifyPath_stack_approach),
        ("Iterative Approach", SimplifyPath().simplifyPath_iterative_approach),
        ("Deque Approach", SimplifyPath().simplifyPath_deque_approach),
        ("Functional Approach", SimplifyPath().simplifyPath_functional_approach),
    ]
    
    path_lengths = [100, 500, 1000]
    
    print("\n=== Path Simplification Performance Benchmark ===")
    
    for length in path_lengths:
        print(f"\n--- Path Length: ~{length} components ---")
        test_paths = [generate_complex_path(length) for _ in range(10)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            for path in test_paths:
                try:
                    alg_func(path)
                except:
                    pass  # Skip errors for benchmark
            
            end_time = time.time()
            
            print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s")

def test_edge_cases():
    """Test edge cases for path simplification"""
    print("\n=== Testing Edge Cases ===")
    
    solver = SimplifyPath()
    
    edge_cases = [
        ("", "/", "Empty path"),
        ("/", "/", "Root only"),
        ("//", "/", "Double slash"),
        ("///", "/", "Triple slash"),
        ("/.", "/", "Root with dot"),
        ("/..", "/", "Root with double dot"),
        ("/./.", "/", "Multiple dots"),
        ("/../..", "/", "Multiple double dots"),
        ("/a/../a/../a/../..", "/", "Complex back and forth"),
        ("/a/b/c/../../../../..", "/", "Too many parent dirs"),
    ]
    
    for path, expected, description in edge_cases:
        result = solver.simplifyPath_stack_approach(path)
        status = "✓" if result == expected else "✗"
        print(f"{description:25} | {status} | '{path}' -> '{result}'")

if __name__ == "__main__":
    test_simplify_path()
    demonstrate_path_simplification()
    test_edge_cases()
    benchmark_path_simplification()

"""
Simplify Path demonstrates multiple approaches to Unix path simplification
including stack-based processing, iterative parsing, regex cleaning,
and functional programming techniques for file system navigation.
"""
