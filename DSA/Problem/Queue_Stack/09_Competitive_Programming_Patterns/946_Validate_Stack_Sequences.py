"""
946. Validate Stack Sequences - Multiple Approaches
Difficulty: Medium

Given two integer arrays pushed and popped each with distinct values, return true if this could have been the result of a sequence of push and pop operations on an initially empty stack.
"""

from typing import List

class ValidateStackSequences:
    """Multiple approaches to validate stack sequences"""
    
    def validateStackSequence_simulation(self, pushed: List[int], popped: List[int]) -> bool:
        """
        Approach 1: Stack Simulation (Optimal)
        
        Simulate the stack operations and check if sequence is valid.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        pop_index = 0
        
        for num in pushed:
            stack.append(num)
            
            # Pop elements while they match the popped sequence
            while stack and pop_index < len(popped) and stack[-1] == popped[pop_index]:
                stack.pop()
                pop_index += 1
        
        # All elements should be popped
        return pop_index == len(popped)
    
    def validateStackSequence_two_pointers(self, pushed: List[int], popped: List[int]) -> bool:
        """
        Approach 2: Two Pointers without Extra Stack
        
        Use two pointers to track push and pop operations.
        
        Time: O(n), Space: O(1) excluding input modification
        """
        if len(pushed) != len(popped):
            return False
        
        push_idx = 0
        pop_idx = 0
        stack_size = 0
        
        # Use pushed array as stack (modify in place)
        while pop_idx < len(popped):
            # Push elements until we find the one we need to pop
            while stack_size == 0 or pushed[stack_size - 1] != popped[pop_idx]:
                if push_idx >= len(pushed):
                    return False
                
                # Push operation
                pushed[stack_size] = pushed[push_idx]
                stack_size += 1
                push_idx += 1
            
            # Pop operation
            stack_size -= 1
            pop_idx += 1
        
        return True
    
    def validateStackSequence_recursive(self, pushed: List[int], popped: List[int]) -> bool:
        """
        Approach 3: Recursive Solution
        
        Use recursion to validate sequences.
        
        Time: O(n), Space: O(n) due to recursion
        """
        def validate(push_idx: int, pop_idx: int, stack: List[int]) -> bool:
            if pop_idx == len(popped):
                return True
            
            # Try to pop if stack top matches
            if stack and stack[-1] == popped[pop_idx]:
                stack.pop()
                if validate(push_idx, pop_idx + 1, stack):
                    return True
                stack.append(popped[pop_idx])  # Backtrack
            
            # Try to push if more elements available
            if push_idx < len(pushed):
                stack.append(pushed[push_idx])
                if validate(push_idx + 1, pop_idx, stack):
                    return True
                stack.pop()  # Backtrack
            
            return False
        
        return validate(0, 0, [])
    
    def validateStackSequence_state_tracking(self, pushed: List[int], popped: List[int]) -> bool:
        """
        Approach 4: State Tracking
        
        Track the state of each element (not_pushed, in_stack, popped).
        
        Time: O(n), Space: O(n)
        """
        if len(pushed) != len(popped):
            return False
        
        # Create mapping from value to index in pushed array
        push_order = {val: i for i, val in enumerate(pushed)}
        
        stack = []
        push_idx = 0
        
        for pop_val in popped:
            if pop_val not in push_order:
                return False
            
            pop_push_idx = push_order[pop_val]
            
            # Push all elements up to and including the one we want to pop
            while push_idx <= pop_push_idx:
                stack.append(pushed[push_idx])
                push_idx += 1
            
            # Check if we can pop the required element
            if not stack or stack[-1] != pop_val:
                return False
            
            stack.pop()
        
        return True
    
    def validateStackSequence_greedy(self, pushed: List[int], popped: List[int]) -> bool:
        """
        Approach 5: Greedy Approach
        
        Greedily push and pop elements.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        pushed_set = set(pushed)
        popped_set = set(popped)
        
        # Check if both arrays have same elements
        if pushed_set != popped_set or len(pushed) != len(popped):
            return False
        
        push_idx = 0
        
        for pop_val in popped:
            # Push elements until we can pop the required one
            while not stack or stack[-1] != pop_val:
                if push_idx >= len(pushed):
                    return False
                stack.append(pushed[push_idx])
                push_idx += 1
            
            # Pop the element
            stack.pop()
        
        return True


def test_validate_stack_sequences():
    """Test validate stack sequences algorithms"""
    solver = ValidateStackSequences()
    
    test_cases = [
        ([1,2,3,4,5], [4,5,3,2,1], True, "Example 1"),
        ([1,2,3,4,5], [4,3,5,1,2], False, "Example 2"),
        ([1], [1], True, "Single element"),
        ([1,2], [2,1], True, "Two elements valid"),
        ([1,2], [1,2], True, "Sequential order"),
        ([1,2,3], [3,1,2], False, "Invalid sequence"),
        ([1,2,3], [1,3,2], True, "Valid mixed sequence"),
        ([1,2,3,4], [2,1,4,3], True, "Nested pattern"),
        ([1,2,3,4], [1,2,3,4], True, "All sequential"),
        ([1,2,3,4], [4,3,2,1], True, "Reverse order"),
        ([1,0], [1,0], True, "With zero"),
        ([2,1,0], [1,2,0], True, "Complex valid"),
        ([0,1,2], [0,2,1], False, "Invalid order"),
    ]
    
    algorithms = [
        ("Simulation", solver.validateStackSequence_simulation),
        ("Two Pointers", solver.validateStackSequence_two_pointers),
        ("Recursive", solver.validateStackSequence_recursive),
        ("State Tracking", solver.validateStackSequence_state_tracking),
        ("Greedy", solver.validateStackSequence_greedy),
    ]
    
    print("=== Testing Validate Stack Sequences ===")
    
    for pushed, popped, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Pushed: {pushed}, Popped: {popped}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                # Make copies to avoid modification
                result = alg_func(pushed[:], popped[:])
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:15} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:40]}")


def demonstrate_simulation_approach():
    """Demonstrate simulation approach step by step"""
    print("\n=== Simulation Approach Step-by-Step Demo ===")
    
    pushed = [1, 2, 3, 4, 5]
    popped = [4, 5, 3, 2, 1]
    
    print(f"Pushed: {pushed}")
    print(f"Popped: {popped}")
    print("Strategy: Simulate stack operations and validate sequence")
    
    stack = []
    pop_index = 0
    
    print(f"\nStep-by-step simulation:")
    
    for i, num in enumerate(pushed):
        print(f"\nStep {i+1}: Push {num}")
        stack.append(num)
        print(f"  Stack after push: {stack}")
        print(f"  Next to pop: {popped[pop_index] if pop_index < len(popped) else 'None'}")
        
        # Pop elements while they match
        pops_in_this_step = []
        while stack and pop_index < len(popped) and stack[-1] == popped[pop_index]:
            popped_val = stack.pop()
            pops_in_this_step.append(popped_val)
            pop_index += 1
        
        if pops_in_this_step:
            print(f"  Popped: {pops_in_this_step}")
            print(f"  Stack after pops: {stack}")
        else:
            print(f"  No pops possible")
    
    print(f"\nFinal state:")
    print(f"  Stack: {stack}")
    print(f"  Elements popped: {pop_index}/{len(popped)}")
    print(f"  Valid sequence: {pop_index == len(popped)}")


def visualize_stack_operations():
    """Visualize stack operations"""
    print("\n=== Stack Operations Visualization ===")
    
    test_cases = [
        ([1,2,3], [1,3,2], "Valid sequence"),
        ([1,2,3], [3,1,2], "Invalid sequence"),
    ]
    
    solver = ValidateStackSequences()
    
    for pushed, popped, description in test_cases:
        print(f"\n{description}: pushed={pushed}, popped={popped}")
        
        stack = []
        pop_idx = 0
        operations = []
        
        for num in pushed:
            stack.append(num)
            operations.append(f"Push {num}")
            
            while stack and pop_idx < len(popped) and stack[-1] == popped[pop_idx]:
                popped_val = stack.pop()
                operations.append(f"Pop {popped_val}")
                pop_idx += 1
        
        print(f"  Operations: {' -> '.join(operations)}")
        print(f"  Final stack: {stack}")
        print(f"  Valid: {pop_idx == len(popped)}")


def demonstrate_competitive_programming_patterns():
    """Demonstrate competitive programming patterns"""
    print("\n=== Competitive Programming Patterns ===")
    
    solver = ValidateStackSequences()
    
    # Pattern 1: Stack simulation
    print("1. Stack Simulation Pattern:")
    print("   Use actual stack to simulate the process")
    print("   Check validity by attempting operations")
    
    example1 = ([1,2,3,4], [2,1,4,3])
    result1 = solver.validateStackSequence_simulation(*example1)
    print(f"   {example1} -> {result1}")
    
    # Pattern 2: Greedy approach
    print(f"\n2. Greedy Processing:")
    print("   Process elements in order they need to be popped")
    print("   Push elements as needed to satisfy pop requirements")
    
    # Pattern 3: State tracking
    print(f"\n3. State-based Validation:")
    print("   Track state of each element: not_pushed, in_stack, popped")
    print("   Ensure valid state transitions")
    
    # Pattern 4: Two pointers optimization
    print(f"\n4. Space Optimization:")
    print("   Use input array as stack to save space")
    print("   Modify in-place for O(1) extra space")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Simulation", "O(n)", "O(n)", "Each element pushed/popped once"),
        ("Two Pointers", "O(n)", "O(1)", "In-place modification"),
        ("Recursive", "O(n)", "O(n)", "Recursion stack overhead"),
        ("State Tracking", "O(n)", "O(n)", "Hash map for ordering"),
        ("Greedy", "O(n)", "O(n)", "Similar to simulation"),
    ]
    
    print(f"{'Approach':<15} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 55)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<15} | {time_comp:<8} | {space_comp:<8} | {notes}")
    
    print(f"\nSimulation approach is most intuitive and efficient")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = ValidateStackSequences()
    
    edge_cases = [
        ([], [], True, "Empty arrays"),
        ([1], [1], True, "Single element"),
        ([1], [2], False, "Different elements"),
        ([1,2], [1,2], True, "Sequential push-pop"),
        ([1,2], [2,1], True, "Stack order"),
        ([1,2,3], [1,2,3], True, "All sequential"),
        ([1,2,3], [3,2,1], True, "All reverse"),
        ([1,2,3], [2,3,1], True, "Mixed valid"),
        ([1,2,3], [1,3,2], True, "Another valid"),
        ([1,2,3], [3,1,2], False, "Invalid order"),
        ([1,2,3], [2,1,3], True, "Nested pattern"),
        ([1,1], [1,1], False, "Duplicate elements (invalid input)"),
    ]
    
    for pushed, popped, expected, description in edge_cases:
        try:
            result = solver.validateStackSequence_simulation(pushed[:], popped[:])
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | pushed={pushed}, popped={popped} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def demonstrate_invalid_sequences():
    """Demonstrate why certain sequences are invalid"""
    print("\n=== Invalid Sequences Analysis ===")
    
    invalid_cases = [
        ([1,2,3,4,5], [4,3,5,1,2], "Cannot pop 5 after 3"),
        ([1,2,3], [3,1,2], "Cannot pop 1 before 2"),
        ([1,2,3,4], [2,4,1,3], "Cannot pop 4 before 3"),
    ]
    
    solver = ValidateStackSequences()
    
    for pushed, popped, reason in invalid_cases:
        print(f"\nInvalid sequence: pushed={pushed}, popped={popped}")
        print(f"Reason: {reason}")
        
        # Show where it fails
        stack = []
        pop_idx = 0
        
        for i, num in enumerate(pushed):
            stack.append(num)
            print(f"  After push {num}: stack={stack}")
            
            while stack and pop_idx < len(popped) and stack[-1] == popped[pop_idx]:
                popped_val = stack.pop()
                print(f"    Pop {popped_val}: stack={stack}")
                pop_idx += 1
            
            if pop_idx < len(popped):
                next_to_pop = popped[pop_idx]
                if next_to_pop not in stack and next_to_pop not in pushed[i+1:]:
                    print(f"  ERROR: Need to pop {next_to_pop} but it's not available")
                    break


if __name__ == "__main__":
    test_validate_stack_sequences()
    demonstrate_simulation_approach()
    visualize_stack_operations()
    demonstrate_competitive_programming_patterns()
    analyze_time_complexity()
    test_edge_cases()
    demonstrate_invalid_sequences()

"""
Validate Stack Sequences demonstrates competitive programming patterns
with stack simulation, sequence validation, and efficient state tracking
for verifying the validity of push/pop operation sequences.
"""
