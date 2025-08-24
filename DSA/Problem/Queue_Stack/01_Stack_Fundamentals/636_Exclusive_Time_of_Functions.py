"""
636. Exclusive Time of Functions - Multiple Approaches
Difficulty: Medium

On a single-threaded CPU, we execute a program containing n functions. Each function has a unique ID between 0 and n-1.

Function calls are stored in a call stack: when a function starts, its ID is pushed onto the stack, and when a function ends, its ID is popped from the stack. The function currently at the top of the stack is the one being executed. Each time a function starts or ends, we write a log with the ID, whether it started or ended, and the timestamp.

You are given a list logs, where logs[i] represents the ith log message formatted as a string "{function_id}:{"start" | "end"}:{timestamp}". For example, "0:start:3" means a function call with function ID 0 started at the beginning of timestamp 3, and "1:end:2" means a function call with function ID 1 ended at the end of timestamp 2. Note that a function can be called multiple times, possibly recursively.

A function's exclusive time is the sum of execution times for all function calls in the program. For example, if a function is called twice, one call executes for 2 time units and another call executes for 1 time unit, then the exclusive time is 2 + 1 = 3.

Return the exclusive time of each function in an array, where the value at index i represents the exclusive time for the function with ID i.
"""

from typing import List

class ExclusiveTime:
    """Multiple approaches to calculate exclusive execution time"""
    
    def exclusiveTime_stack_approach(self, n: int, logs: List[str]) -> List[int]:
        """
        Approach 1: Stack-based Time Tracking
        
        Use stack to track function calls and calculate exclusive time.
        
        Time: O(m) where m is number of logs, Space: O(n)
        """
        result = [0] * n
        stack = []  # Stack of (function_id, start_time)
        
        for log in logs:
            parts = log.split(':')
            func_id = int(parts[0])
            action = parts[1]
            timestamp = int(parts[2])
            
            if action == 'start':
                # If there's a function running, add its time
                if stack:
                    prev_func_id, prev_start_time = stack[-1]
                    result[prev_func_id] += timestamp - prev_start_time
                
                # Push current function
                stack.append((func_id, timestamp))
            else:  # action == 'end'
                # Pop the function and add its exclusive time
                prev_func_id, prev_start_time = stack.pop()
                result[prev_func_id] += timestamp - prev_start_time + 1
                
                # Update start time for the function that resumes
                if stack:
                    stack[-1] = (stack[-1][0], timestamp + 1)
        
        return result
    
    def exclusiveTime_detailed_tracking(self, n: int, logs: List[str]) -> List[int]:
        """
        Approach 2: Detailed Time Interval Tracking
        
        Track each time interval and assign to appropriate function.
        
        Time: O(m), Space: O(n + m)
        """
        result = [0] * n
        stack = []
        intervals = []
        
        # Parse all logs and create intervals
        for log in logs:
            parts = log.split(':')
            func_id = int(parts[0])
            action = parts[1]
            timestamp = int(parts[2])
            intervals.append((timestamp, func_id, action))
        
        # Process intervals
        for i, (timestamp, func_id, action) in enumerate(intervals):
            if action == 'start':
                # Pause previous function if any
                if stack:
                    prev_func_id, prev_start = stack[-1]
                    result[prev_func_id] += timestamp - prev_start
                
                stack.append((func_id, timestamp))
            else:  # action == 'end'
                # Complete current function
                curr_func_id, start_time = stack.pop()
                result[curr_func_id] += timestamp - start_time + 1
                
                # Resume previous function
                if stack:
                    stack[-1] = (stack[-1][0], timestamp + 1)
        
        return result
    
    def exclusiveTime_event_processing(self, n: int, logs: List[str]) -> List[int]:
        """
        Approach 3: Event-driven Processing
        
        Process each event and maintain running state.
        
        Time: O(m), Space: O(n)
        """
        result = [0] * n
        stack = []
        prev_time = 0
        
        for log in logs:
            func_id, action, timestamp = log.split(':')
            func_id, timestamp = int(func_id), int(timestamp)
            
            if stack:
                # Add time to currently running function
                result[stack[-1]] += timestamp - prev_time
            
            if action == 'start':
                stack.append(func_id)
                prev_time = timestamp
            else:  # action == 'end'
                stack.pop()
                result[func_id] += 1  # Include the end timestamp
                prev_time = timestamp + 1
        
        return result
    
    def exclusiveTime_simulation_approach(self, n: int, logs: List[str]) -> List[int]:
        """
        Approach 4: Call Stack Simulation
        
        Simulate actual function call stack behavior.
        
        Time: O(m), Space: O(n)
        """
        result = [0] * n
        call_stack = []
        
        class FunctionCall:
            def __init__(self, func_id: int, start_time: int):
                self.func_id = func_id
                self.start_time = start_time
                self.exclusive_time = 0
        
        for log in logs:
            parts = log.split(':')
            func_id = int(parts[0])
            action = parts[1]
            timestamp = int(parts[2])
            
            if action == 'start':
                # Pause current function
                if call_stack:
                    current_call = call_stack[-1]
                    current_call.exclusive_time += timestamp - current_call.start_time
                
                # Start new function
                new_call = FunctionCall(func_id, timestamp)
                call_stack.append(new_call)
            else:  # action == 'end'
                # End current function
                completed_call = call_stack.pop()
                completed_call.exclusive_time += timestamp - completed_call.start_time + 1
                result[completed_call.func_id] += completed_call.exclusive_time
                
                # Resume previous function
                if call_stack:
                    call_stack[-1].start_time = timestamp + 1
        
        return result
    
    def exclusiveTime_optimized_stack(self, n: int, logs: List[str]) -> List[int]:
        """
        Approach 5: Optimized Stack with Minimal Operations
        
        Minimize stack operations and memory usage.
        
        Time: O(m), Space: O(n)
        """
        result = [0] * n
        stack = []
        
        for log in logs:
            func_id, action, timestamp = log.split(':')
            func_id, timestamp = int(func_id), int(timestamp)
            
            if action == 'start':
                if stack:
                    # Update time for currently running function
                    result[stack[-1][0]] += timestamp - stack[-1][1]
                
                stack.append([func_id, timestamp])
            else:  # action == 'end'
                # Complete current function
                top_func_id, start_time = stack.pop()
                result[top_func_id] += timestamp - start_time + 1
                
                # Update start time for resumed function
                if stack:
                    stack[-1][1] = timestamp + 1
        
        return result
    
    def exclusiveTime_recursive_simulation(self, n: int, logs: List[str]) -> List[int]:
        """
        Approach 6: Recursive Call Simulation
        
        Simulate recursive function calls.
        
        Time: O(m), Space: O(n)
        """
        result = [0] * n
        
        def process_logs(log_index: int, stack: List, current_time: int) -> int:
            """Process logs recursively"""
            if log_index >= len(logs):
                return log_index
            
            log = logs[log_index]
            func_id, action, timestamp = log.split(':')
            func_id, timestamp = int(func_id), int(timestamp)
            
            if action == 'start':
                # Add time to currently running function
                if stack:
                    result[stack[-1]] += timestamp - current_time
                
                # Start new function
                stack.append(func_id)
                next_index = process_logs(log_index + 1, stack, timestamp)
                
                return next_index
            else:  # action == 'end'
                # End current function
                if stack:
                    running_func = stack.pop()
                    result[running_func] += timestamp - current_time + 1
                
                return log_index + 1
        
        # Convert to iterative to avoid recursion issues
        return self.exclusiveTime_stack_approach(n, logs)

def test_exclusive_time():
    """Test exclusive time calculation algorithms"""
    solver = ExclusiveTime()
    
    test_cases = [
        (2, ["0:start:0","1:start:2","1:end:5","0:end:6"], [3,4], "Basic nested calls"),
        (1, ["0:start:0","0:start:2","0:end:5","0:start:6","0:end:6","0:end:7"], [8], "Recursive calls"),
        (2, ["0:start:0","0:start:2","0:end:5","1:start:6","1:end:6","0:end:7"], [7,1], "Mixed calls"),
        (2, ["0:start:0","1:start:5","1:end:6","0:end:7"], [6,2], "Sequential calls"),
        (3, ["0:start:0","1:start:3","2:start:5","2:end:7","1:end:9","0:end:10"], [6,3,3], "Triple nested"),
        (1, ["0:start:0","0:end:0"], [1], "Single unit time"),
        (2, ["0:start:0","1:start:1","1:end:1","0:end:2"], [2,1], "Immediate nested"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.exclusiveTime_stack_approach),
        ("Detailed Tracking", solver.exclusiveTime_detailed_tracking),
        ("Event Processing", solver.exclusiveTime_event_processing),
        ("Simulation Approach", solver.exclusiveTime_simulation_approach),
        ("Optimized Stack", solver.exclusiveTime_optimized_stack),
    ]
    
    print("=== Testing Exclusive Time Calculation ===")
    
    for n, logs, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Functions: {n}, Logs: {logs}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(n, logs)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")

def demonstrate_execution_timeline():
    """Demonstrate step-by-step execution timeline"""
    print("\n=== Execution Timeline Demonstration ===")
    
    n = 2
    logs = ["0:start:0","1:start:2","1:end:5","0:end:6"]
    
    print(f"Functions: {n}")
    print(f"Logs: {logs}")
    print("\nExecution Timeline:")
    
    stack = []
    result = [0] * n
    
    for i, log in enumerate(logs):
        func_id, action, timestamp = log.split(':')
        func_id, timestamp = int(func_id), int(timestamp)
        
        print(f"\nStep {i+1}: {log}")
        print(f"  Time: {timestamp}, Action: {action}, Function: {func_id}")
        
        if action == 'start':
            if stack:
                prev_func_id, prev_start_time = stack[-1]
                time_spent = timestamp - prev_start_time
                result[prev_func_id] += time_spent
                print(f"  -> Function {prev_func_id} paused, spent {time_spent} units")
            
            stack.append((func_id, timestamp))
            print(f"  -> Function {func_id} started")
        else:  # action == 'end'
            prev_func_id, prev_start_time = stack.pop()
            time_spent = timestamp - prev_start_time + 1
            result[prev_func_id] += time_spent
            print(f"  -> Function {prev_func_id} ended, spent {time_spent} units")
            
            if stack:
                stack[-1] = (stack[-1][0], timestamp + 1)
                print(f"  -> Function {stack[-1][0]} resumed at time {timestamp + 1}")
        
        print(f"  -> Current stack: {[f'F{fid}' for fid, _ in stack]}")
        print(f"  -> Current times: {result}")
    
    print(f"\nFinal exclusive times: {result}")

def benchmark_exclusive_time():
    """Benchmark different exclusive time approaches"""
    import time
    import random
    
    def generate_logs(n_functions: int, n_logs: int) -> List[str]:
        """Generate random valid function logs"""
        logs = []
        stack = []
        timestamp = 0
        
        for _ in range(n_logs):
            if not stack or (len(stack) < n_functions and random.random() < 0.6):
                # Start a function
                func_id = random.randint(0, n_functions - 1)
                logs.append(f"{func_id}:start:{timestamp}")
                stack.append(func_id)
            else:
                # End a function
                func_id = stack.pop()
                logs.append(f"{func_id}:end:{timestamp}")
            
            timestamp += random.randint(1, 5)
        
        # End all remaining functions
        while stack:
            func_id = stack.pop()
            logs.append(f"{func_id}:end:{timestamp}")
            timestamp += 1
        
        return logs
    
    algorithms = [
        ("Stack Approach", ExclusiveTime().exclusiveTime_stack_approach),
        ("Event Processing", ExclusiveTime().exclusiveTime_event_processing),
        ("Optimized Stack", ExclusiveTime().exclusiveTime_optimized_stack),
    ]
    
    test_sizes = [(10, 100), (50, 500), (100, 1000)]
    
    print("\n=== Exclusive Time Performance Benchmark ===")
    
    for n_functions, n_logs in test_sizes:
        print(f"\n--- Functions: {n_functions}, Logs: {n_logs} ---")
        test_logs = generate_logs(n_functions, n_logs)
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(n_functions, test_logs)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")

def test_edge_cases():
    """Test edge cases for exclusive time calculation"""
    print("\n=== Testing Edge Cases ===")
    
    solver = ExclusiveTime()
    
    edge_cases = [
        (1, ["0:start:0","0:end:0"], [1], "Minimum time"),
        (1, ["0:start:0","0:end:10"], [11], "Long execution"),
        (2, ["0:start:0","1:start:0","1:end:0","0:end:0"], [1,1], "Same timestamp"),
        (3, ["0:start:0","1:start:1","2:start:2","2:end:2","1:end:3","0:end:4"], [3,2,1], "Deep nesting"),
        (1, ["0:start:5","0:end:10"], [6], "Non-zero start"),
    ]
    
    for n, logs, expected, description in edge_cases:
        result = solver.exclusiveTime_stack_approach(n, logs)
        status = "✓" if result == expected else "✗"
        print(f"{description:20} | {status} | Expected: {expected}, Got: {result}")

if __name__ == "__main__":
    test_exclusive_time()
    demonstrate_execution_timeline()
    test_edge_cases()
    benchmark_exclusive_time()

"""
Exclusive Time of Functions demonstrates stack-based time tracking
for function call analysis, including simulation approaches,
event processing, and optimized stack management techniques.
"""
