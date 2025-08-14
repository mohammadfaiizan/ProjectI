"""
Greedy Algorithm Interview Problems - Most Asked Questions
=========================================================

Topics: Common interview questions with detailed solutions and explanations
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix, Uber, LinkedIn
Difficulty: Easy to Hard
Time Complexity: O(n) to O(n log n) depending on problem
Space Complexity: O(1) to O(n) for problem-specific storage
"""

from typing import List, Tuple, Optional, Dict, Any, Set
import heapq
from collections import Counter, defaultdict
import math

class GreedyInterviewProblems:
    
    def __init__(self):
        """Initialize with interview problem tracking"""
        self.solution_steps = []
        self.problem_stats = {}
    
    # ==========================================
    # 1. CLASSIC GREEDY INTERVIEW PROBLEMS
    # ==========================================
    
    def assign_cookies(self, greed: List[int], cookies: List[int]) -> int:
        """
        Assign Cookies
        
        Company: Amazon, Google
        Difficulty: Easy
        Time: O(n log n + m log m), Space: O(1)
        
        Problem: Maximize number of content children
        Greedy Strategy: Sort both arrays, assign smallest satisfying cookie
        
        LeetCode 455 - Perfect starter greedy problem
        """
        print("=== ASSIGN COOKIES ===")
        print("Problem: Maximize number of content children")
        print("Greedy Strategy: Assign smallest cookie that satisfies each child")
        print()
        
        print(f"Children greed factors: {greed}")
        print(f"Available cookies: {cookies}")
        print()
        
        # Sort both arrays
        greed.sort()
        cookies.sort()
        
        print(f"Sorted greed factors: {greed}")
        print(f"Sorted cookies: {cookies}")
        print()
        
        child = 0
        content_children = 0
        
        print("Greedy assignment process:")
        for i, cookie_size in enumerate(cookies):
            if child < len(greed):
                print(f"Cookie {i+1} (size {cookie_size}): ", end="")
                
                if cookie_size >= greed[child]:
                    print(f"satisfies child {child+1} (greed {greed[child]})")
                    content_children += 1
                    child += 1
                else:
                    print(f"too small for child {child+1} (greed {greed[child]})")
            else:
                print(f"Cookie {i+1} (size {cookie_size}): all children satisfied")
        
        print(f"\nMaximum content children: {content_children}")
        
        # Show which children remain unsatisfied
        if child < len(greed):
            unsatisfied = greed[child:]
            print(f"Unsatisfied children with greed factors: {unsatisfied}")
        
        return content_children
    
    def lemonade_change(self, bills: List[int]) -> bool:
        """
        Lemonade Change
        
        Company: Amazon, Facebook
        Difficulty: Easy
        Time: O(n), Space: O(1)
        
        Problem: Determine if correct change can be given to all customers
        Greedy Strategy: Always give largest bills as change first
        
        LeetCode 860 - Tests greedy choice validation
        """
        print("=== LEMONADE CHANGE ===")
        print("Problem: Can we give correct change to all customers?")
        print("Greedy Strategy: Use largest bills for change first")
        print("Lemonade costs $5, customers pay $5, $10, or $20")
        print()
        
        print(f"Customer bills: {bills}")
        print()
        
        # Count available bills
        five_count = 0
        ten_count = 0
        
        print("Processing customers:")
        for i, bill in enumerate(bills):
            print(f"Customer {i+1}: pays ${bill}")
            
            if bill == 5:
                five_count += 1
                print(f"   No change needed")
                print(f"   Cash register: ${5}×{five_count}, ${10}×{ten_count}")
            
            elif bill == 10:
                # Need to give $5 change
                if five_count >= 1:
                    five_count -= 1
                    ten_count += 1
                    print(f"   Give ${5} change")
                    print(f"   Cash register: ${5}×{five_count}, ${10}×{ten_count}")
                else:
                    print(f"   ✗ Cannot give ${5} change (no ${5} bills)")
                    return False
            
            elif bill == 20:
                # Need to give $15 change
                # Strategy 1: Give one $10 and one $5
                if ten_count >= 1 and five_count >= 1:
                    ten_count -= 1
                    five_count -= 1
                    print(f"   Give ${10} + ${5} change")
                    print(f"   Cash register: ${5}×{five_count}, ${10}×{ten_count}")
                
                # Strategy 2: Give three $5 bills
                elif five_count >= 3:
                    five_count -= 3
                    print(f"   Give ${5} + ${5} + ${5} change")
                    print(f"   Cash register: ${5}×{five_count}, ${10}×{ten_count}")
                
                else:
                    print(f"   ✗ Cannot give ${15} change")
                    print(f"   Available: ${5}×{five_count}, ${10}×{ten_count}")
                    return False
            
            print()
        
        print("✓ Successfully served all customers!")
        return True
    
    def non_overlapping_intervals(self, intervals: List[List[int]]) -> int:
        """
        Non-overlapping Intervals
        
        Company: Google, Amazon, Microsoft
        Difficulty: Medium
        Time: O(n log n), Space: O(1)
        
        Problem: Minimum intervals to remove to make all non-overlapping
        Greedy Strategy: Sort by end time, remove intervals that start before current end
        
        LeetCode 435 - Classic interval greedy problem
        """
        print("=== NON-OVERLAPPING INTERVALS ===")
        print("Problem: Minimum removals to make all intervals non-overlapping")
        print("Greedy Strategy: Keep intervals that end earliest")
        print()
        
        if not intervals:
            return 0
        
        print(f"Input intervals: {intervals}")
        
        # Sort by end time
        intervals.sort(key=lambda x: x[1])
        
        print(f"Sorted by end time: {intervals}")
        print()
        
        kept_count = 1
        last_end = intervals[0][1]
        removed_intervals = []
        
        print("Greedy selection process:")
        print(f"Keep interval 1: {intervals[0]}")
        
        for i in range(1, len(intervals)):
            start, end = intervals[i]
            
            print(f"\nConsider interval {i+1}: [{start}, {end}]")
            print(f"   Last kept interval ends at: {last_end}")
            
            if start >= last_end:
                # No overlap, keep this interval
                kept_count += 1
                last_end = end
                print(f"   ✓ Keep (no overlap)")
                print(f"   Kept intervals: {kept_count}")
            else:
                # Overlap detected, remove this interval
                removed_intervals.append([start, end])
                print(f"   ✗ Remove (overlaps with previous)")
                print(f"   Overlap: starts at {start} < last_end {last_end}")
        
        removals = len(intervals) - kept_count
        
        print(f"\nResult:")
        print(f"   Total intervals: {len(intervals)}")
        print(f"   Kept intervals: {kept_count}")
        print(f"   Removed intervals: {removals}")
        print(f"   Removed: {removed_intervals}")
        
        return removals
    
    # ==========================================
    # 2. ADVANCED GREEDY PROBLEMS
    # ==========================================
    
    def minimum_platforms(self, arrivals: List[int], departures: List[int]) -> int:
        """
        Minimum Railway Platforms
        
        Company: Amazon, Google, Railway systems
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        Problem: Find minimum platforms needed for train schedule
        Greedy Strategy: Process events chronologically, track platform usage
        """
        print("=== MINIMUM RAILWAY PLATFORMS ===")
        print("Problem: Find minimum platforms needed")
        print("Greedy Strategy: Process arrival/departure events chronologically")
        print()
        
        if len(arrivals) != len(departures):
            print("Error: Mismatched arrivals and departures!")
            return -1
        
        print("Train schedule:")
        for i in range(len(arrivals)):
            print(f"   Train {i+1}: arrives {arrivals[i]}, departs {departures[i]}")
        print()
        
        # Create events: (time, type) where type: 1=arrival, -1=departure
        events = []
        for i in range(len(arrivals)):
            events.append((arrivals[i], 1))    # Arrival
            events.append((departures[i], -1)) # Departure
        
        # Sort events: departures before arrivals at same time
        events.sort(key=lambda x: (x[0], x[1]))
        
        print("Events timeline (time, type):")
        for time, event_type in events:
            event_name = "arrival" if event_type == 1 else "departure"
            print(f"   Time {time}: {event_name}")
        print()
        
        current_platforms = 0
        max_platforms = 0
        
        print("Platform usage simulation:")
        for i, (time, event_type) in enumerate(events):
            current_platforms += event_type
            max_platforms = max(max_platforms, current_platforms)
            
            event_name = "arrival" if event_type == 1 else "departure"
            print(f"Event {i+1}: Time {time} - {event_name}")
            print(f"   Platforms in use: {current_platforms}")
            print(f"   Maximum so far: {max_platforms}")
            print()
        
        print(f"Minimum platforms required: {max_platforms}")
        return max_platforms
    
    def task_scheduler(self, tasks: List[str], n: int) -> int:
        """
        Task Scheduler
        
        Company: Facebook, Amazon, Google
        Difficulty: Medium
        Time: O(m), Space: O(1) where m is execution time
        
        Problem: Schedule tasks with cooling period between same tasks
        Greedy Strategy: Execute most frequent task, then fill cooling slots
        
        LeetCode 621 - Advanced greedy with mathematical insight
        """
        print("=== TASK SCHEDULER ===")
        print("Problem: Schedule tasks with cooling period")
        print("Greedy Strategy: Execute most frequent tasks first")
        print()
        
        print(f"Tasks: {tasks}")
        print(f"Cooling period (n): {n}")
        
        # Count task frequencies
        task_count = Counter(tasks)
        max_freq = max(task_count.values())
        max_freq_count = sum(1 for freq in task_count.values() if freq == max_freq)
        
        print(f"\nTask frequencies: {dict(task_count)}")
        print(f"Maximum frequency: {max_freq}")
        print(f"Tasks with max frequency: {max_freq_count}")
        print()
        
        # Mathematical approach
        # Minimum time = max(len(tasks), (max_freq - 1) * (n + 1) + max_freq_count)
        min_time_formula = (max_freq - 1) * (n + 1) + max_freq_count
        min_time = max(len(tasks), min_time_formula)
        
        print("Mathematical analysis:")
        print(f"   Total tasks: {len(tasks)}")
        print(f"   Formula result: (max_freq - 1) × (n + 1) + max_freq_count")
        print(f"                  = ({max_freq} - 1) × ({n} + 1) + {max_freq_count}")
        print(f"                  = {max_freq - 1} × {n + 1} + {max_freq_count}")
        print(f"                  = {(max_freq - 1) * (n + 1)} + {max_freq_count}")
        print(f"                  = {min_time_formula}")
        print(f"   Minimum time: max({len(tasks)}, {min_time_formula}) = {min_time}")
        print()
        
        # Detailed simulation
        print("Detailed execution simulation:")
        
        # Use heap for task frequencies
        heap = [-count for count in task_count.values()]
        heapq.heapify(heap)
        
        time = 0
        execution_log = []
        
        while heap:
            temp = []
            cycle_time = 0
            
            print(f"Cycle starting at time {time}:")
            
            # Execute tasks for n+1 slots
            for i in range(n + 1):
                if heap:
                    freq = -heapq.heappop(heap)
                    execution_log.append(f"Task_{freq}")
                    
                    print(f"   Slot {i}: Execute task (freq was {freq})")
                    
                    if freq > 1:
                        temp.append(-(freq - 1))
                    
                    cycle_time += 1
                else:
                    if temp:  # Only idle if more tasks remain
                        execution_log.append("idle")
                        print(f"   Slot {i}: Idle")
                        cycle_time += 1
            
            # Add tasks back to heap
            for freq in temp:
                heapq.heappush(heap, freq)
            
            time += cycle_time
            print(f"   Cycle completed, total time: {time}")
            print()
        
        print(f"Execution sequence: {execution_log[:20]}{'...' if len(execution_log) > 20 else ''}")
        print(f"Total execution time: {time}")
        
        return time
    
    def valid_parenthesis_string(self, s: str) -> bool:
        """
        Valid Parenthesis String
        
        Company: Facebook, Amazon, Google
        Difficulty: Medium
        Time: O(n), Space: O(1)
        
        Problem: Check if string with *, (, ) can form valid parentheses
        Greedy Strategy: Track range of possible open parentheses count
        
        LeetCode 678 - Greedy with range tracking
        """
        print("=== VALID PARENTHESIS STRING ===")
        print("Problem: Validate parentheses with wildcards")
        print("Greedy Strategy: Track min and max possible open parentheses")
        print("Rules: * can be (, ), or empty")
        print()
        
        print(f"Input string: '{s}'")
        print()
        
        # Track range of possible open parentheses count
        min_open = 0  # Minimum possible open parentheses
        max_open = 0  # Maximum possible open parentheses
        
        print("Character-by-character analysis:")
        for i, char in enumerate(s):
            print(f"Position {i}: '{char}'")
            
            if char == '(':
                min_open += 1
                max_open += 1
                print(f"   Open parenthesis: min_open={min_open}, max_open={max_open}")
                
            elif char == ')':
                min_open = max(min_open - 1, 0)  # Can't go negative
                max_open -= 1
                print(f"   Close parenthesis: min_open={min_open}, max_open={max_open}")
                
                if max_open < 0:
                    print(f"   ✗ Too many closing parentheses")
                    return False
                
            else:  # char == '*'
                min_open = max(min_open - 1, 0)  # * as ')'
                max_open += 1                    # * as '('
                print(f"   Wildcard: min_open={min_open}, max_open={max_open}")
                print(f"   (Treating * as ')' for min, '(' for max)")
            
            print(f"   Range of possible open count: [{min_open}, {max_open}]")
            print()
        
        # Check if we can have exactly 0 open parentheses
        valid = min_open <= 0 <= max_open
        
        print(f"Final range: [{min_open}, {max_open}]")
        print(f"Can achieve 0 open parentheses: {valid}")
        
        if valid:
            print("✓ Valid parenthesis string")
        else:
            print("✗ Invalid parenthesis string")
        
        return valid
    
    # ==========================================
    # 3. GREEDY + DATA STRUCTURES
    # ==========================================
    
    def find_median_from_data_stream(self):
        """
        Find Median from Data Stream (Greedy Heap Management)
        
        Company: Google, Facebook, Amazon
        Difficulty: Hard
        Time: O(log n) per insertion, Space: O(n)
        
        Problem: Maintain median in streaming data
        Greedy Strategy: Balance two heaps to keep median accessible
        
        LeetCode 295 - Combines greedy with data structures
        """
        
        class MedianFinder:
            def __init__(self):
                self.max_heap = []  # Left half (smaller elements)
                self.min_heap = []  # Right half (larger elements)
                
                print("MedianFinder initialized")
                print("Strategy: max_heap (smaller half) + min_heap (larger half)")
                print("Invariant: |max_heap| - |min_heap| ∈ {0, 1}")
            
            def addNum(self, num: int) -> None:
                print(f"\nAdding number: {num}")
                
                # Greedy decision: which heap to add to
                if not self.max_heap or num <= -self.max_heap[0]:
                    heapq.heappush(self.max_heap, -num)
                    print(f"   Added to max_heap (smaller half)")
                else:
                    heapq.heappush(self.min_heap, num)
                    print(f"   Added to min_heap (larger half)")
                
                # Greedy rebalancing
                self._balance()
                self._display_state()
            
            def findMedian(self) -> float:
                if len(self.max_heap) == len(self.min_heap):
                    if not self.max_heap:
                        return 0.0
                    return (-self.max_heap[0] + self.min_heap[0]) / 2.0
                else:
                    return float(-self.max_heap[0])
            
            def _balance(self):
                if len(self.max_heap) > len(self.min_heap) + 1:
                    val = -heapq.heappop(self.max_heap)
                    heapq.heappush(self.min_heap, val)
                    print(f"   Rebalanced: moved {val} from max_heap to min_heap")
                elif len(self.min_heap) > len(self.max_heap):
                    val = heapq.heappop(self.min_heap)
                    heapq.heappush(self.max_heap, -val)
                    print(f"   Rebalanced: moved {val} from min_heap to max_heap")
            
            def _display_state(self):
                smaller = [-x for x in self.max_heap]
                larger = list(self.min_heap)
                print(f"   Max heap (smaller): {smaller}")
                print(f"   Min heap (larger): {larger}")
                print(f"   Sizes: {len(self.max_heap)}, {len(self.min_heap)}")
                print(f"   Current median: {self.findMedian()}")
        
        print("=== FIND MEDIAN FROM DATA STREAM ===")
        median_finder = MedianFinder()
        
        # Test with stream of numbers
        numbers = [1, 2, 3, 4, 5]
        for num in numbers:
            median_finder.addNum(num)
        
        return median_finder
    
    # ==========================================
    # 4. INTERVIEW STRATEGY PROBLEMS
    # ==========================================
    
    def two_city_scheduling(self, costs: List[List[int]]) -> int:
        """
        Two City Scheduling
        
        Company: Amazon, Google
        Difficulty: Medium
        Time: O(n log n), Space: O(1)
        
        Problem: Send n people to city A and n people to city B with minimum cost
        Greedy Strategy: Sort by cost difference between cities
        
        LeetCode 1029 - Tests understanding of opportunity cost
        """
        print("=== TWO CITY SCHEDULING ===")
        print("Problem: Send equal people to two cities with minimum cost")
        print("Greedy Strategy: Sort by opportunity cost (cost difference)")
        print()
        
        n = len(costs) // 2
        print(f"Total people: {len(costs)}")
        print(f"People per city: {n}")
        print()
        
        print("People and travel costs [cityA, cityB]:")
        for i, (cost_a, cost_b) in enumerate(costs):
            diff = cost_a - cost_b
            print(f"   Person {i+1}: cityA=${cost_a}, cityB=${cost_b}, diff=${diff}")
        print()
        
        # Sort by cost difference (cityA - cityB)
        # Negative difference means cityA is cheaper (send to A)
        # Positive difference means cityB is cheaper (send to B)
        indexed_costs = [(cost_a - cost_b, i, cost_a, cost_b) for i, (cost_a, cost_b) in enumerate(costs)]
        indexed_costs.sort()
        
        print("Sorted by cost difference (cityA - cityB):")
        for diff, person_id, cost_a, cost_b in indexed_costs:
            preference = "A" if diff < 0 else "B" if diff > 0 else "Either"
            print(f"   Person {person_id+1}: diff=${diff}, prefer city {preference}")
        print()
        
        total_cost = 0
        city_a_count = 0
        city_b_count = 0
        assignments = []
        
        print("Greedy assignment:")
        for i, (diff, person_id, cost_a, cost_b) in enumerate(indexed_costs):
            print(f"Person {person_id+1}: ", end="")
            
            if city_a_count < n and (city_b_count >= n or diff <= 0):
                # Send to city A
                assignments.append((person_id+1, 'A', cost_a))
                total_cost += cost_a
                city_a_count += 1
                print(f"→ City A (cost ${cost_a})")
            else:
                # Send to city B
                assignments.append((person_id+1, 'B', cost_b))
                total_cost += cost_b
                city_b_count += 1
                print(f"→ City B (cost ${cost_b})")
            
            print(f"   Running total: ${total_cost}")
            print(f"   City counts: A={city_a_count}, B={city_b_count}")
            print()
        
        print("Final assignment:")
        city_a_people = [p for p, city, cost in assignments if city == 'A']
        city_b_people = [p for p, city, cost in assignments if city == 'B']
        
        print(f"   City A: {city_a_people}")
        print(f"   City B: {city_b_people}")
        print(f"   Total minimum cost: ${total_cost}")
        
        return total_cost


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_greedy_interview_problems():
    """Demonstrate key greedy interview problems"""
    print("=== GREEDY INTERVIEW PROBLEMS DEMONSTRATION ===\n")
    
    problems = GreedyInterviewProblems()
    
    # 1. Classic Problems
    print("1. CLASSIC GREEDY INTERVIEW PROBLEMS")
    
    print("a) Assign Cookies:")
    problems.assign_cookies([1, 2, 3], [1, 1])
    print("\n" + "-"*40 + "\n")
    
    print("b) Lemonade Change:")
    problems.lemonade_change([5, 5, 5, 10, 20])
    print("\n" + "-"*40 + "\n")
    
    print("c) Non-overlapping Intervals:")
    problems.non_overlapping_intervals([[1,2],[2,3],[3,4],[1,3]])
    print("\n" + "="*60 + "\n")
    
    # 2. Advanced Problems
    print("2. ADVANCED GREEDY PROBLEMS")
    
    print("a) Minimum Railway Platforms:")
    arrivals = [900, 940, 950, 1100, 1500, 1800]
    departures = [910, 1200, 1120, 1130, 1900, 2000]
    problems.minimum_platforms(arrivals, departures)
    print("\n" + "-"*40 + "\n")
    
    print("b) Task Scheduler:")
    problems.task_scheduler(['A','A','A','B','B','B'], 2)
    print("\n" + "-"*40 + "\n")
    
    print("c) Valid Parenthesis String:")
    problems.valid_parenthesis_string("(*)")
    print("\n" + "="*60 + "\n")
    
    # 3. Greedy + Data Structures
    print("3. GREEDY + DATA STRUCTURES")
    
    print("Find Median from Data Stream:")
    problems.find_median_from_data_stream()
    print("\n" + "="*60 + "\n")
    
    # 4. Interview Strategy
    print("4. INTERVIEW STRATEGY PROBLEMS")
    
    print("Two City Scheduling:")
    costs = [[10,20],[30,200],[400,50],[30,20]]
    problems.two_city_scheduling(costs)


if __name__ == "__main__":
    demonstrate_greedy_interview_problems()
    
    print("\n=== GREEDY INTERVIEW SUCCESS STRATEGY ===")
    
    print("\n🎯 INTERVIEW PREPARATION ROADMAP:")
    print("Week 1: Master basic greedy concepts and classic problems")
    print("Week 2: Practice interval scheduling and activity selection")
    print("Week 3: Solve advanced optimization and mathematical problems")
    print("Week 4: Tackle hybrid problems (greedy + other techniques)")
    
    print("\n📚 MUST-PRACTICE PROBLEMS BY DIFFICULTY:")
    print("Easy:")
    print("  • Assign Cookies - Foundation greedy matching")
    print("  • Lemonade Change - Greedy resource management")
    print("  • Best Time to Buy/Sell Stock - Simple optimization")
    print()
    print("Medium:")
    print("  • Non-overlapping Intervals - Classic interval greedy")
    print("  • Task Scheduler - Frequency + greedy optimization")
    print("  • Valid Parenthesis String - Range tracking greedy")
    print("  • Gas Station - Circular array greedy")
    print()
    print("Hard:")
    print("  • Find Median from Data Stream - Greedy heap management")
    print("  • Create Maximum Number - Complex greedy construction")
    print("  • Smallest Range Covering Elements - Multi-array greedy")
    
    print("\n⚡ QUICK PROBLEM IDENTIFICATION:")
    print("• 'Maximum/minimum' with local choices → Greedy candidate")
    print("• 'Optimal selection/assignment' → Consider greedy approach")
    print("• 'Activity/interval scheduling' → Classic greedy domain")
    print("• 'Resource allocation with constraints' → Greedy optimization")
    print("• 'String/array reconstruction' → Greedy with data structures")
    
    print("\n🏆 INTERVIEW DAY TIPS:")
    print("• Always explain WHY greedy works for the specific problem")
    print("• Provide counterexamples to show when greedy fails")
    print("• Discuss time/space complexity before coding")
    print("• Start with brute force, then optimize to greedy")
    print("• Test your solution with edge cases")
    print("• Consider alternative approaches (DP, divide & conquer)")
    
    print("\n📊 COMPLEXITY EXPECTATIONS:")
    print("• Most greedy problems: O(n log n) due to sorting")
    print("• Some optimization problems: O(n) with clever techniques")
    print("• Space complexity usually: O(1) to O(n)")
    print("• Greedy often more efficient than DP for applicable problems")
    
    print("\n🎓 ADVANCED INTERVIEW CONCEPTS:")
    print("• Prove greedy choice property using exchange argument")
    print("• Discuss when greedy provides approximation vs optimal solution")
    print("• Explain relationship between greedy and matroids")
    print("• Compare greedy with dynamic programming trade-offs")
    print("• Understand online vs offline algorithm implications")
    
    print("\n💡 COMMON INTERVIEW PATTERNS:")
    print("• Sort then greedily select (most common)")
    print("• Two pointers with greedy choices")
    print("• Heap/priority queue for dynamic selection")
    print("• Stack for maintaining optimal subsequences")
    print("• Range tracking for constraint satisfaction")
    
    print("\n🚨 INTERVIEW PITFALLS TO AVOID:")
    print("• Assuming greedy works without proof/justification")
    print("• Using wrong sorting criteria for greedy choice")
    print("• Missing edge cases in constraint handling")
    print("• Not considering when greedy gives approximation only")
    print("• Overcomplicating problems that have simple greedy solutions")
