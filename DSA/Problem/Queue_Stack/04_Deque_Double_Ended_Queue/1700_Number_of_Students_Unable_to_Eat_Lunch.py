"""
1700. Number of Students Unable to Eat Lunch - Multiple Approaches
Difficulty: Easy

The school cafeteria offers circular and square sandwiches at lunch break, referred to by numbers 0 and 1 respectively. 
All students stand in a queue. Each student either prefers square sandwiches or circular sandwiches.

The number of sandwiches in the cafeteria is equal to the number of students. The sandwiches are placed in a stack. 
At each step:
- If the student at the front of the queue prefers the sandwich on the top of the stack, they will take it and leave the queue.
- Otherwise, they will leave it and go to the end of the queue.

This continues until none of the queue students want to take the top sandwich and are thus unable to eat.

You are given two integer arrays students and sandwiches where sandwiches[i] is the type of the ith sandwich in the stack (i = 0 is the top of the stack) and students[j] is the preference of the jth student in the initial queue (j = 0 is the front of the queue). Return the number of students that are unable to eat.
"""

from typing import List
from collections import deque

class NumberOfStudentsUnableToEat:
    """Multiple approaches to solve students unable to eat lunch problem"""
    
    def countStudents_deque_simulation(self, students: List[int], sandwiches: List[int]) -> int:
        """
        Approach 1: Deque Simulation (Optimal)
        
        Simulate the process using deque for efficient front/back operations.
        
        Time: O(n²), Space: O(n)
        """
        student_queue = deque(students)
        sandwich_stack = sandwiches[:]
        sandwich_idx = 0
        
        # Track consecutive students who couldn't eat
        consecutive_unable = 0
        
        while student_queue and sandwich_idx < len(sandwich_stack):
            current_student = student_queue.popleft()
            current_sandwich = sandwich_stack[sandwich_idx]
            
            if current_student == current_sandwich:
                # Student takes the sandwich
                sandwich_idx += 1
                consecutive_unable = 0  # Reset counter
            else:
                # Student goes to back of queue
                student_queue.append(current_student)
                consecutive_unable += 1
                
                # If all remaining students have been unable to eat, break
                if consecutive_unable == len(student_queue):
                    break
        
        return len(student_queue)
    
    def countStudents_counting_approach(self, students: List[int], sandwiches: List[int]) -> int:
        """
        Approach 2: Counting Approach (Most Optimal)
        
        Count preferences and match with available sandwiches.
        
        Time: O(n), Space: O(1)
        """
        # Count student preferences
        count_0 = students.count(0)  # Students who prefer circular (0)
        count_1 = students.count(1)  # Students who prefer square (1)
        
        # Process sandwiches from top to bottom
        for sandwich in sandwiches:
            if sandwich == 0:
                if count_0 > 0:
                    count_0 -= 1
                else:
                    # No more students want circular sandwiches
                    break
            else:  # sandwich == 1
                if count_1 > 0:
                    count_1 -= 1
                else:
                    # No more students want square sandwiches
                    break
        
        return count_0 + count_1
    
    def countStudents_queue_simulation(self, students: List[int], sandwiches: List[int]) -> int:
        """
        Approach 3: Queue Simulation with List
        
        Simulate using list operations.
        
        Time: O(n²), Space: O(n)
        """
        student_queue = students[:]
        sandwich_idx = 0
        rotations = 0
        
        while student_queue and sandwich_idx < len(sandwiches):
            if student_queue[0] == sandwiches[sandwich_idx]:
                # Student takes sandwich
                student_queue.pop(0)
                sandwich_idx += 1
                rotations = 0
            else:
                # Student goes to back
                student_queue.append(student_queue.pop(0))
                rotations += 1
                
                # If we've rotated through all students without success
                if rotations == len(student_queue):
                    break
        
        return len(student_queue)
    
    def countStudents_recursive_approach(self, students: List[int], sandwiches: List[int]) -> int:
        """
        Approach 4: Recursive Simulation
        
        Use recursion to simulate the process.
        
        Time: O(n²), Space: O(n)
        """
        def simulate(student_queue: List[int], sandwich_idx: int, rotations: int) -> int:
            # Base cases
            if not student_queue or sandwich_idx >= len(sandwiches):
                return len(student_queue)
            
            if rotations == len(student_queue):
                return len(student_queue)
            
            current_student = student_queue[0]
            current_sandwich = sandwiches[sandwich_idx]
            
            if current_student == current_sandwich:
                # Student takes sandwich
                return simulate(student_queue[1:], sandwich_idx + 1, 0)
            else:
                # Student goes to back
                new_queue = student_queue[1:] + [current_student]
                return simulate(new_queue, sandwich_idx, rotations + 1)
        
        return simulate(students, 0, 0)
    
    def countStudents_optimized_counting(self, students: List[int], sandwiches: List[int]) -> int:
        """
        Approach 5: Optimized Counting with Early Termination
        
        Count and process with optimization.
        
        Time: O(n), Space: O(1)
        """
        preferences = [0, 0]  # preferences[0] = count of 0s, preferences[1] = count of 1s
        
        # Count preferences
        for student in students:
            preferences[student] += 1
        
        # Process sandwiches
        for i, sandwich in enumerate(sandwiches):
            if preferences[sandwich] > 0:
                preferences[sandwich] -= 1
            else:
                # No student wants this sandwich, remaining students can't eat
                return sum(preferences)
        
        return 0
    
    def countStudents_stack_queue_simulation(self, students: List[int], sandwiches: List[int]) -> int:
        """
        Approach 6: Stack and Queue Simulation
        
        Explicitly use stack for sandwiches and queue for students.
        
        Time: O(n²), Space: O(n)
        """
        student_queue = deque(students)
        sandwich_stack = sandwiches[::-1]  # Reverse to use as stack (pop from end)
        
        attempts = 0
        max_attempts = len(students)
        
        while student_queue and sandwich_stack and attempts < max_attempts:
            current_student = student_queue.popleft()
            current_sandwich = sandwich_stack[-1]  # Top of stack
            
            if current_student == current_sandwich:
                sandwich_stack.pop()  # Remove sandwich from stack
                attempts = 0  # Reset attempts
            else:
                student_queue.append(current_student)  # Student goes to back
                attempts += 1
        
        return len(student_queue)
    
    def countStudents_frequency_matching(self, students: List[int], sandwiches: List[int]) -> int:
        """
        Approach 7: Frequency Matching
        
        Match frequencies of preferences with sandwich types.
        
        Time: O(n), Space: O(1)
        """
        from collections import Counter
        
        student_count = Counter(students)
        
        for sandwich in sandwiches:
            if student_count[sandwich] > 0:
                student_count[sandwich] -= 1
            else:
                # No more students want this type of sandwich
                break
        
        return sum(student_count.values())


def test_number_of_students_unable_to_eat():
    """Test number of students unable to eat algorithms"""
    solver = NumberOfStudentsUnableToEat()
    
    test_cases = [
        ([1,1,0,0], [0,1,0,1], 0, "Example 1"),
        ([1,1,1,0,0,1], [1,0,0,0,1,1], 3, "Example 2"),
        ([0,0,0], [1,1,1], 3, "No matching preferences"),
        ([1,1,1], [0,0,0], 3, "No matching preferences reverse"),
        ([0], [0], 0, "Single student matching"),
        ([0], [1], 1, "Single student not matching"),
        ([0,1], [0,1], 0, "Two students perfect match"),
        ([0,1], [1,0], 0, "Two students reverse match"),
        ([1,0,1,0], [0,1,0,1], 0, "Alternating pattern"),
        ([0,0,1,1], [0,0,1,1], 0, "Grouped preferences"),
    ]
    
    algorithms = [
        ("Deque Simulation", solver.countStudents_deque_simulation),
        ("Counting Approach", solver.countStudents_counting_approach),
        ("Queue Simulation", solver.countStudents_queue_simulation),
        ("Recursive Approach", solver.countStudents_recursive_approach),
        ("Optimized Counting", solver.countStudents_optimized_counting),
        ("Stack Queue Simulation", solver.countStudents_stack_queue_simulation),
        ("Frequency Matching", solver.countStudents_frequency_matching),
    ]
    
    print("=== Testing Number of Students Unable to Eat Lunch ===")
    
    for students, sandwiches, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Students: {students}")
        print(f"Sandwiches: {sandwiches}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(students, sandwiches)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:25} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:25} | ERROR: {str(e)[:40]}")


def demonstrate_simulation_process():
    """Demonstrate the simulation process step by step"""
    print("\n=== Simulation Process Demonstration ===")
    
    students = [1, 1, 0, 0]
    sandwiches = [0, 1, 0, 1]
    
    print(f"Students queue: {students}")
    print(f"Sandwich stack: {sandwiches} (top to bottom)")
    
    student_queue = deque(students)
    sandwich_idx = 0
    step = 1
    
    print("\nSimulation steps:")
    
    while student_queue and sandwich_idx < len(sandwiches):
        current_student = student_queue[0]
        current_sandwich = sandwiches[sandwich_idx]
        
        print(f"\nStep {step}:")
        print(f"  Queue: {list(student_queue)}")
        print(f"  Current sandwich: {current_sandwich}")
        print(f"  Front student wants: {current_student}")
        
        if current_student == current_sandwich:
            student_queue.popleft()
            sandwich_idx += 1
            print(f"  ✓ Student takes sandwich and leaves")
        else:
            student = student_queue.popleft()
            student_queue.append(student)
            print(f"  ✗ Student goes to back of queue")
        
        step += 1
        
        # Prevent infinite loop for demonstration
        if step > 20:
            print("  ... (stopping demonstration to prevent infinite loop)")
            break
    
    print(f"\nFinal result: {len(student_queue)} students unable to eat")


def demonstrate_counting_approach():
    """Demonstrate the counting approach"""
    print("\n=== Counting Approach Demonstration ===")
    
    students = [1, 1, 1, 0, 0, 1]
    sandwiches = [1, 0, 0, 0, 1, 1]
    
    print(f"Students: {students}")
    print(f"Sandwiches: {sandwiches}")
    
    # Count preferences
    count_0 = students.count(0)
    count_1 = students.count(1)
    
    print(f"\nStudent preferences:")
    print(f"  Want circular (0): {count_0}")
    print(f"  Want square (1): {count_1}")
    
    print(f"\nProcessing sandwiches from top to bottom:")
    
    for i, sandwich in enumerate(sandwiches):
        print(f"\nStep {i+1}: Sandwich {sandwich}")
        
        if sandwich == 0:
            if count_0 > 0:
                count_0 -= 1
                print(f"  ✓ Student takes circular sandwich")
                print(f"  Remaining wanting circular: {count_0}")
            else:
                print(f"  ✗ No students want circular sandwiches")
                print(f"  Stopping here - remaining students can't eat")
                break
        else:  # sandwich == 1
            if count_1 > 0:
                count_1 -= 1
                print(f"  ✓ Student takes square sandwich")
                print(f"  Remaining wanting square: {count_1}")
            else:
                print(f"  ✗ No students want square sandwiches")
                print(f"  Stopping here - remaining students can't eat")
                break
    
    unable_to_eat = count_0 + count_1
    print(f"\nFinal result: {unable_to_eat} students unable to eat")
    print(f"  {count_0} wanting circular + {count_1} wanting square")


def visualize_queue_operations():
    """Visualize queue operations"""
    print("\n=== Queue Operations Visualization ===")
    
    students = [1, 0, 1, 0]
    sandwiches = [0, 1, 0, 1]
    
    print("Initial state:")
    print(f"Queue: {students} (front -> back)")
    print(f"Stack: {sandwiches} (top -> bottom)")
    
    student_queue = deque(students)
    sandwich_idx = 0
    
    print("\nOperations:")
    
    step = 1
    while student_queue and sandwich_idx < len(sandwiches) and step <= 10:
        print(f"\nStep {step}:")
        
        # Show current state
        queue_visual = " -> ".join(map(str, student_queue))
        print(f"  Queue: [{queue_visual}]")
        print(f"  Next sandwich: {sandwiches[sandwich_idx]}")
        
        current_student = student_queue[0]
        current_sandwich = sandwiches[sandwich_idx]
        
        if current_student == current_sandwich:
            student_queue.popleft()
            sandwich_idx += 1
            print(f"  Action: Student {current_student} takes sandwich {current_sandwich}")
        else:
            student = student_queue.popleft()
            student_queue.append(student)
            print(f"  Action: Student {current_student} goes to back (wants {current_student}, got {current_sandwich})")
        
        step += 1
    
    print(f"\nFinal queue length: {len(student_queue)}")


def benchmark_students_unable_to_eat():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Deque Simulation", NumberOfStudentsUnableToEat().countStudents_deque_simulation),
        ("Counting Approach", NumberOfStudentsUnableToEat().countStudents_counting_approach),
        ("Queue Simulation", NumberOfStudentsUnableToEat().countStudents_queue_simulation),
        ("Optimized Counting", NumberOfStudentsUnableToEat().countStudents_optimized_counting),
        ("Frequency Matching", NumberOfStudentsUnableToEat().countStudents_frequency_matching),
    ]
    
    # Test with different sizes
    sizes = [100, 1000, 5000]
    
    print("\n=== Students Unable to Eat Performance Benchmark ===")
    
    for size in sizes:
        print(f"\n--- Array Size: {size} ---")
        
        # Generate random test data
        students = [random.randint(0, 1) for _ in range(size)]
        sandwiches = [random.randint(0, 1) for _ in range(size)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(students, sandwiches)
                end_time = time.time()
                print(f"{alg_name:25} | Time: {end_time - start_time:.4f}s | Result: {result}")
            except Exception as e:
                print(f"{alg_name:25} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = NumberOfStudentsUnableToEat()
    
    edge_cases = [
        ([], [], 0, "Empty arrays"),
        ([0], [0], 0, "Single matching"),
        ([0], [1], 1, "Single non-matching"),
        ([0, 0], [1, 1], 2, "All mismatched"),
        ([1, 1], [0, 0], 2, "All mismatched reverse"),
        ([0, 1, 0, 1], [0, 1, 0, 1], 0, "Perfect order"),
        ([0, 1, 0, 1], [1, 0, 1, 0], 0, "Reverse order but solvable"),
        ([0, 0, 0, 1], [1, 0, 0, 0], 1, "Mostly matching"),
        ([1, 1, 1, 0], [0, 1, 1, 1], 1, "Mostly matching reverse"),
    ]
    
    for students, sandwiches, expected, description in edge_cases:
        try:
            result = solver.countStudents_counting_approach(students, sandwiches)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | Students: {students}, Sandwiches: {sandwiches} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_cases = [
        ([1, 1, 0, 0], [0, 1, 0, 1]),
        ([1, 1, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1]),
        ([0, 0, 1, 1], [1, 1, 0, 0]),
        ([1, 0, 1, 0], [0, 1, 0, 1]),
    ]
    
    solver = NumberOfStudentsUnableToEat()
    
    approaches = [
        ("Deque Sim", solver.countStudents_deque_simulation),
        ("Counting", solver.countStudents_counting_approach),
        ("Queue Sim", solver.countStudents_queue_simulation),
        ("Recursive", solver.countStudents_recursive_approach),
        ("Frequency", solver.countStudents_frequency_matching),
    ]
    
    for i, (students, sandwiches) in enumerate(test_cases):
        print(f"\nTest case {i+1}: Students={students}, Sandwiches={sandwiches}")
        
        results = {}
        
        for name, func in approaches:
            try:
                result = func(students, sandwiches)
                results[name] = result
                print(f"{name:15} | Result: {result}")
            except Exception as e:
                print(f"{name:15} | ERROR: {str(e)[:40]}")
        
        # Check consistency
        if results:
            first_result = list(results.values())[0]
            all_same = all(result == first_result for result in results.values())
            print(f"All approaches agree: {'✓' if all_same else '✗'}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Deque Simulation", "O(n²)", "O(n)", "Worst case: all students rotate"),
        ("Counting Approach", "O(n)", "O(1)", "Count preferences, process sandwiches"),
        ("Queue Simulation", "O(n²)", "O(n)", "List operations for queue simulation"),
        ("Recursive Approach", "O(n²)", "O(n)", "Recursive calls with list copying"),
        ("Optimized Counting", "O(n)", "O(1)", "Single pass counting optimization"),
        ("Stack Queue Simulation", "O(n²)", "O(n)", "Explicit stack and queue operations"),
        ("Frequency Matching", "O(n)", "O(1)", "Counter-based frequency matching"),
    ]
    
    print(f"{'Approach':<25} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 75)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<25} | {time_comp:<8} | {space_comp:<8} | {notes}")


def demonstrate_deque_advantages():
    """Demonstrate advantages of using deque"""
    print("\n=== Deque Advantages Demonstration ===")
    
    print("Deque vs List for queue operations:")
    print("1. deque.popleft() - O(1)")
    print("2. list.pop(0) - O(n)")
    print("3. deque.append() - O(1)")
    print("4. list.append() - O(1)")
    
    print("\nDeque operations in this problem:")
    
    from collections import deque
    
    students = [1, 0, 1, 0]
    
    # Using deque
    print(f"\nUsing deque:")
    dq = deque(students)
    print(f"Initial: {list(dq)}")
    
    front = dq.popleft()  # O(1)
    print(f"After popleft(): {list(dq)}, removed: {front}")
    
    dq.append(front)  # O(1)
    print(f"After append(): {list(dq)}")
    
    # Using list (less efficient)
    print(f"\nUsing list (less efficient):")
    lst = students[:]
    print(f"Initial: {lst}")
    
    front = lst.pop(0)  # O(n) - shifts all elements
    print(f"After pop(0): {lst}, removed: {front}")
    
    lst.append(front)  # O(1)
    print(f"After append(): {lst}")
    
    print("\nConclusion: Deque is more efficient for queue operations!")


if __name__ == "__main__":
    test_number_of_students_unable_to_eat()
    demonstrate_simulation_process()
    demonstrate_counting_approach()
    visualize_queue_operations()
    demonstrate_deque_advantages()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_students_unable_to_eat()

"""
Number of Students Unable to Eat Lunch demonstrates deque applications
for simulation problems, including queue operations, counting optimizations,
and multiple approaches for process simulation with early termination.
"""
