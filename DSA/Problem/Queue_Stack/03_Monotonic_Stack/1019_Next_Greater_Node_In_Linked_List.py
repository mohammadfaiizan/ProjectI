"""
1019. Next Greater Node In Linked List - Multiple Approaches
Difficulty: Medium

You are given the head of a linked list with n nodes.

For each node in the list, find the value of the next greater node. That is, for each node, find the value of the first node that is next to it and has a strictly larger value.

Return an integer array answer where answer[i] is the value of the next greater node of the ith node (1-indexed). If the ith node does not have a next greater node, set answer[i] = 0.
"""

from typing import List, Optional

# Definition for singly-linked list
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class NextGreaterNodeInLinkedList:
    """Multiple approaches to find next greater nodes in linked list"""
    
    def nextLargerNodes_stack_approach(self, head: Optional[ListNode]) -> List[int]:
        """
        Approach 1: Monotonic Stack (Optimal)
        
        Use stack to track indices of nodes waiting for next greater element.
        
        Time: O(n), Space: O(n)
        """
        # Convert linked list to array for easier indexing
        values = []
        current = head
        while current:
            values.append(current.val)
            current = current.next
        
        result = [0] * len(values)
        stack = []  # Store indices
        
        for i, val in enumerate(values):
            # Pop indices with smaller values
            while stack and values[stack[-1]] < val:
                prev_idx = stack.pop()
                result[prev_idx] = val
            
            stack.append(i)
        
        return result
    
    def nextLargerNodes_single_pass(self, head: Optional[ListNode]) -> List[int]:
        """
        Approach 2: Single Pass with Stack
        
        Process linked list in single pass using stack.
        
        Time: O(n), Space: O(n)
        """
        stack = []  # Store (index, value) pairs
        result = []
        index = 0
        
        current = head
        while current:
            # Initialize result for current node
            result.append(0)
            
            # Pop nodes with smaller values
            while stack and stack[-1][1] < current.val:
                prev_idx, prev_val = stack.pop()
                result[prev_idx] = current.val
            
            # Add current node to stack
            stack.append((index, current.val))
            
            current = current.next
            index += 1
        
        return result
    
    def nextLargerNodes_brute_force(self, head: Optional[ListNode]) -> List[int]:
        """
        Approach 3: Brute Force
        
        For each node, scan remaining nodes to find next greater.
        
        Time: O(n²), Space: O(n)
        """
        # Convert to array first
        values = []
        current = head
        while current:
            values.append(current.val)
            current = current.next
        
        result = []
        
        for i in range(len(values)):
            next_greater = 0
            
            # Look for next greater element
            for j in range(i + 1, len(values)):
                if values[j] > values[i]:
                    next_greater = values[j]
                    break
            
            result.append(next_greater)
        
        return result
    
    def nextLargerNodes_recursive(self, head: Optional[ListNode]) -> List[int]:
        """
        Approach 4: Recursive Approach
        
        Use recursion to process nodes and find next greater elements.
        
        Time: O(n²), Space: O(n)
        """
        def find_next_greater(node: Optional[ListNode], target_val: int) -> int:
            """Find next greater value starting from given node"""
            while node:
                if node.val > target_val:
                    return node.val
                node = node.next
            return 0
        
        def process_node(node: Optional[ListNode]) -> List[int]:
            """Process current node and recursively handle rest"""
            if not node:
                return []
            
            # Find next greater for current node
            next_greater = find_next_greater(node.next, node.val)
            
            # Recursively process remaining nodes
            rest_result = process_node(node.next)
            
            return [next_greater] + rest_result
        
        return process_node(head)
    
    def nextLargerNodes_reverse_process(self, head: Optional[ListNode]) -> List[int]:
        """
        Approach 5: Reverse Processing
        
        Store all values first, then process from right to left.
        
        Time: O(n), Space: O(n)
        """
        # Convert linked list to array
        values = []
        current = head
        while current:
            values.append(current.val)
            current = current.next
        
        n = len(values)
        result = [0] * n
        stack = []
        
        # Process from right to left
        for i in range(n - 1, -1, -1):
            # Pop elements not greater than current
            while stack and stack[-1] <= values[i]:
                stack.pop()
            
            # Next greater element is top of stack
            result[i] = stack[-1] if stack else 0
            
            # Push current element
            stack.append(values[i])
        
        return result
    
    def nextLargerNodes_two_pass(self, head: Optional[ListNode]) -> List[int]:
        """
        Approach 6: Two Pass Approach
        
        First pass to collect values, second pass to find next greater.
        
        Time: O(n), Space: O(n)
        """
        # First pass: collect all values
        values = []
        current = head
        while current:
            values.append(current.val)
            current = current.next
        
        # Second pass: find next greater elements
        n = len(values)
        result = [0] * n
        stack = []
        
        for i in range(n):
            # Process stack for next greater elements
            while stack and values[stack[-1]] < values[i]:
                prev_idx = stack.pop()
                result[prev_idx] = values[i]
            
            stack.append(i)
        
        return result


def create_linked_list(values: List[int]) -> Optional[ListNode]:
    """Helper function to create linked list from array"""
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head


def linked_list_to_array(head: Optional[ListNode]) -> List[int]:
    """Helper function to convert linked list to array"""
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result


def test_next_greater_node_in_linked_list():
    """Test next greater node in linked list algorithms"""
    solver = NextGreaterNodeInLinkedList()
    
    test_cases = [
        ([2,1,5], [5,5,0], "Example 1"),
        ([1,7,5,1,9,2,5,1], [7,9,9,9,0,5,0,0], "Example 2"),
        ([1,2,3,4,5], [2,3,4,5,0], "Increasing sequence"),
        ([5,4,3,2,1], [0,0,0,0,0], "Decreasing sequence"),
        ([1], [0], "Single node"),
        ([1,1,1], [0,0,0], "All same values"),
        ([3,3,5,5], [5,5,0,0], "Duplicate values"),
        ([1,3,2,4], [3,4,4,0], "Mixed pattern"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.nextLargerNodes_stack_approach),
        ("Single Pass", solver.nextLargerNodes_single_pass),
        ("Brute Force", solver.nextLargerNodes_brute_force),
        ("Recursive", solver.nextLargerNodes_recursive),
        ("Reverse Process", solver.nextLargerNodes_reverse_process),
        ("Two Pass", solver.nextLargerNodes_two_pass),
    ]
    
    print("=== Testing Next Greater Node in Linked List ===")
    
    for values, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Linked List: {values}")
        print(f"Expected: {expected}")
        
        # Create linked list
        head = create_linked_list(values)
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(head)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    values = [2, 1, 5]
    print(f"Linked List values: {values}")
    
    result = [0] * len(values)
    stack = []
    
    for i, val in enumerate(values):
        print(f"\nStep {i+1}: Processing node with value {val} at index {i}")
        print(f"  Stack before: {[f'idx{idx}({values[idx]})' for idx in stack]}")
        
        # Pop indices with smaller values
        popped = []
        while stack and values[stack[-1]] < val:
            prev_idx = stack.pop()
            result[prev_idx] = val
            popped.append(f'idx{prev_idx}({values[prev_idx]})')
        
        if popped:
            print(f"  Popped: {popped} -> set next greater to {val}")
        
        stack.append(i)
        print(f"  Stack after: {[f'idx{idx}({values[idx]})' for idx in stack]}")
        print(f"  Result so far: {result}")
    
    print(f"\nFinal result: {result}")


def demonstrate_linked_list_processing():
    """Demonstrate linked list processing"""
    print("\n=== Linked List Processing Demonstration ===")
    
    values = [1, 7, 5, 1, 9, 2, 5, 1]
    head = create_linked_list(values)
    
    print(f"Linked List: {values}")
    print("Processing each node:")
    
    # Show linked list structure
    current = head
    index = 0
    while current:
        print(f"  Node {index}: value={current.val}, address={id(current)}")
        current = current.next
        index += 1
    
    print("\nFinding next greater elements:")
    
    solver = NextGreaterNodeInLinkedList()
    result = solver.nextLargerNodes_stack_approach(head)
    
    for i, (val, next_greater) in enumerate(zip(values, result)):
        if next_greater == 0:
            print(f"  Node {i} (value {val}): No next greater element")
        else:
            print(f"  Node {i} (value {val}): Next greater = {next_greater}")


def visualize_next_greater_concept():
    """Visualize next greater concept for linked list"""
    print("\n=== Next Greater Concept Visualization ===")
    
    values = [2, 1, 5]
    print(f"Linked List: {' -> '.join(map(str, values))}")
    print()
    
    for i, val in enumerate(values):
        print(f"Node {i} (value {val}):")
        
        # Find next greater
        next_greater = 0
        for j in range(i + 1, len(values)):
            if values[j] > val:
                next_greater = values[j]
                break
        
        # Show visualization
        vis = ['·'] * len(values)
        vis[i] = '█'  # Current node
        
        if next_greater != 0:
            next_pos = values.index(next_greater, i + 1)
            vis[next_pos] = '▲'  # Next greater node
            
            # Show connection
            for k in range(i + 1, next_pos):
                vis[k] = '-'
        
        vis_str = ' '.join(vis)
        print(f"  {vis_str}")
        print(f"  Next greater: {next_greater if next_greater != 0 else 'None'}")
        print()


def benchmark_next_greater_node():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Stack Approach", NextGreaterNodeInLinkedList().nextLargerNodes_stack_approach),
        ("Single Pass", NextGreaterNodeInLinkedList().nextLargerNodes_single_pass),
        ("Brute Force", NextGreaterNodeInLinkedList().nextLargerNodes_brute_force),
        ("Reverse Process", NextGreaterNodeInLinkedList().nextLargerNodes_reverse_process),
    ]
    
    # Test with different linked list sizes
    sizes = [100, 1000, 5000]
    
    print("\n=== Next Greater Node Performance Benchmark ===")
    
    for size in sizes:
        print(f"\n--- Linked List Size: {size} ---")
        
        # Generate random values
        values = [random.randint(1, 1000) for _ in range(size)]
        head = create_linked_list(values)
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(head)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = NextGreaterNodeInLinkedList()
    
    edge_cases = [
        ([], [], "Empty linked list"),
        ([1], [0], "Single node"),
        ([1, 1], [0, 0], "Two same values"),
        ([1, 2], [2, 0], "Two increasing values"),
        ([2, 1], [0, 0], "Two decreasing values"),
        ([1, 3, 2], [3, 0, 0], "Peak in middle"),
        ([3, 1, 2], [0, 2, 0], "Valley in middle"),
        ([1000, 999, 998], [0, 0, 0], "Large decreasing"),
        ([1, 1000, 1], [1000, 0, 0], "Large peak"),
    ]
    
    for values, expected, description in edge_cases:
        try:
            head = create_linked_list(values)
            result = solver.nextLargerNodes_stack_approach(head)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | {values} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_linked_list_vs_array():
    """Compare linked list vs array processing"""
    print("\n=== Linked List vs Array Processing Comparison ===")
    
    values = [2, 1, 5, 3, 4]
    
    print(f"Values: {values}")
    print()
    
    # Array-based next greater element
    def next_greater_array(arr):
        result = [0] * len(arr)
        stack = []
        
        for i, val in enumerate(arr):
            while stack and arr[stack[-1]] < val:
                prev_idx = stack.pop()
                result[prev_idx] = val
            stack.append(i)
        
        return result
    
    # Linked list-based
    head = create_linked_list(values)
    solver = NextGreaterNodeInLinkedList()
    
    array_result = next_greater_array(values)
    linked_list_result = solver.nextLargerNodes_stack_approach(head)
    
    print(f"Array approach result:       {array_result}")
    print(f"Linked list approach result: {linked_list_result}")
    print(f"Results match: {'✓' if array_result == linked_list_result else '✗'}")
    
    print("\nKey differences:")
    print("- Array: Direct indexing, O(1) access")
    print("- Linked List: Sequential access, O(n) to reach position")
    print("- Both use same monotonic stack algorithm")
    print("- Linked list requires conversion to array or single-pass processing")


def demonstrate_single_pass_approach():
    """Demonstrate single pass approach"""
    print("\n=== Single Pass Approach Demonstration ===")
    
    values = [1, 7, 5, 1, 9]
    head = create_linked_list(values)
    
    print(f"Linked List: {values}")
    print("Processing in single pass:")
    
    stack = []
    result = []
    index = 0
    
    current = head
    while current:
        print(f"\nStep {index + 1}: Processing node {index} with value {current.val}")
        
        # Initialize result for current node
        result.append(0)
        print(f"  Initialized result[{index}] = 0")
        
        # Pop nodes with smaller values
        popped = []
        while stack and stack[-1][1] < current.val:
            prev_idx, prev_val = stack.pop()
            result[prev_idx] = current.val
            popped.append(f"idx{prev_idx}({prev_val})")
        
        if popped:
            print(f"  Updated previous nodes: {popped} -> next greater = {current.val}")
        
        # Add current node to stack
        stack.append((index, current.val))
        print(f"  Added to stack: idx{index}({current.val})")
        print(f"  Current result: {result}")
        
        current = current.next
        index += 1
    
    print(f"\nFinal result: {result}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack Approach", "O(n)", "O(n)", "Convert to array + stack processing"),
        ("Single Pass", "O(n)", "O(n)", "Process linked list once with stack"),
        ("Brute Force", "O(n²)", "O(n)", "For each node, scan remaining nodes"),
        ("Recursive", "O(n²)", "O(n)", "Recursive calls with linear search"),
        ("Reverse Process", "O(n)", "O(n)", "Convert to array + reverse processing"),
        ("Two Pass", "O(n)", "O(n)", "Two linear passes through data"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 65)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<8} | {space_comp:<8} | {notes}")


def demonstrate_monotonic_stack_property():
    """Demonstrate monotonic stack property in linked list context"""
    print("\n=== Monotonic Stack Property in Linked List ===")
    
    values = [3, 1, 4, 2, 5]
    print(f"Linked List: {values}")
    print("Stack maintains decreasing order of values:")
    
    stack = []
    result = [0] * len(values)
    
    for i, val in enumerate(values):
        print(f"\nStep {i+1}: Processing value {val}")
        print(f"  Stack before: {[f'idx{idx}({values[idx]})' for idx in stack]}")
        
        # Show what gets popped and why
        popped = []
        while stack and values[stack[-1]] < val:
            popped_idx = stack.pop()
            result[popped_idx] = val
            popped.append(f'idx{popped_idx}({values[popped_idx]})')
        
        if popped:
            print(f"  Popped: {popped} (values < {val})")
        
        stack.append(i)
        print(f"  Stack after: {[f'idx{idx}({values[idx]})' for idx in stack]}")
        
        # Verify monotonic property
        if len(stack) > 1:
            stack_values = [values[idx] for idx in stack]
            is_decreasing = all(stack_values[j] >= stack_values[j+1] for j in range(len(stack_values)-1))
            print(f"  Monotonic property: {'✓' if is_decreasing else '✗'} (values: {stack_values})")
    
    print(f"\nFinal result: {result}")


if __name__ == "__main__":
    test_next_greater_node_in_linked_list()
    demonstrate_linked_list_processing()
    demonstrate_stack_approach()
    demonstrate_single_pass_approach()
    visualize_next_greater_concept()
    demonstrate_monotonic_stack_property()
    test_edge_cases()
    compare_linked_list_vs_array()
    analyze_time_complexity()
    benchmark_next_greater_node()

"""
Next Greater Node In Linked List demonstrates monotonic stack applications
for linked list processing, including single-pass algorithms and multiple
approaches for finding next greater elements in sequential data structures.
"""
