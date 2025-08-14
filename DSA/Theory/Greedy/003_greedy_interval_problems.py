"""
Greedy Interval and Scheduling Problems
======================================

Topics: Interval scheduling, meeting rooms, task scheduling, resource allocation
Companies: Google, Amazon, Microsoft, Facebook, Apple, Uber
Difficulty: Medium to Hard
Time Complexity: O(n log n) for sorting, O(n) for greedy selection
Space Complexity: O(n) for sorting and solution storage
"""

from typing import List, Tuple, Optional, Dict, Any
import heapq
from collections import defaultdict

class GreedyIntervalProblems:
    
    def __init__(self):
        """Initialize with problem tracking"""
        self.solution_steps = []
        self.problem_count = 0
    
    # ==========================================
    # 1. INTERVAL SCHEDULING MAXIMIZATION
    # ==========================================
    
    def interval_scheduling_maximization(self, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Interval Scheduling Maximization
        
        Company: Google, Amazon, Microsoft
        Difficulty: Medium
        Time: O(n log n), Space: O(1)
        
        Problem: Select maximum number of non-overlapping intervals
        Greedy Strategy: Always choose interval that ends earliest
        
        Args:
            intervals: List of (start, end) tuples
        
        Returns:
            List of selected intervals
        """
        print("=== INTERVAL SCHEDULING MAXIMIZATION ===")
        print("Problem: Select maximum number of non-overlapping intervals")
        print("Greedy Strategy: Choose interval that ends earliest")
        print()
        
        if not intervals:
            return []
        
        print(f"Input intervals: {intervals}")
        print()
        
        # Sort by end time
        sorted_intervals = sorted(intervals, key=lambda x: x[1])
        
        print("Step 1: Sort by end time")
        for i, (start, end) in enumerate(sorted_intervals):
            print(f"   {i+1}. [{start}, {end}]")
        print()
        
        # Greedy selection
        selected = []
        last_end = float('-inf')
        
        print("Step 2: Greedy selection")
        for i, (start, end) in enumerate(sorted_intervals):
            print(f"Consider interval [{start}, {end}]")
            
            if start >= last_end:
                selected.append((start, end))
                last_end = end
                print(f"   ✓ Selected (start {start} >= last_end {last_end if last_end != end else 'none'})")
                print(f"   Current selection: {selected}")
            else:
                print(f"   ✗ Rejected (start {start} < last_end {last_end})")
            print()
        
        print(f"Maximum non-overlapping intervals: {len(selected)}")
        print(f"Selected intervals: {selected}")
        
        return selected
    
    def interval_scheduling_weighted(self, intervals: List[Tuple[int, int, int]]) -> Tuple[List[Tuple[int, int, int]], int]:
        """
        Weighted Interval Scheduling (Greedy Approximation)
        
        Company: Google, Facebook
        Difficulty: Hard
        Time: O(n log n), Space: O(n)
        
        Problem: Select intervals to maximize total weight
        Greedy Strategy: Sort by weight/duration ratio
        
        Note: This is an approximation algorithm, not optimal
        """
        print("=== WEIGHTED INTERVAL SCHEDULING (GREEDY APPROXIMATION) ===")
        print("Problem: Select intervals to maximize total weight")
        print("Greedy Strategy: Sort by weight/duration ratio")
        print("Note: This gives approximation, not optimal solution")
        print()
        
        if not intervals:
            return [], 0
        
        print("Input intervals (start, end, weight):")
        for i, (start, end, weight) in enumerate(intervals):
            duration = end - start
            ratio = weight / duration if duration > 0 else float('inf')
            print(f"   {i+1}. [{start}, {end}], weight={weight}, duration={duration}, ratio={ratio:.2f}")
        print()
        
        # Sort by weight/duration ratio
        intervals_with_ratio = []
        for start, end, weight in intervals:
            duration = end - start
            ratio = weight / duration if duration > 0 else float('inf')
            intervals_with_ratio.append((start, end, weight, ratio))
        
        intervals_with_ratio.sort(key=lambda x: x[3], reverse=True)
        
        print("Sorted by weight/duration ratio:")
        for i, (start, end, weight, ratio) in enumerate(intervals_with_ratio):
            print(f"   {i+1}. [{start}, {end}], weight={weight}, ratio={ratio:.2f}")
        print()
        
        # Greedy selection
        selected = []
        total_weight = 0
        last_end = float('-inf')
        
        print("Greedy selection process:")
        for i, (start, end, weight, ratio) in enumerate(intervals_with_ratio):
            print(f"Consider interval [{start}, {end}], weight={weight}")
            
            if start >= last_end:
                selected.append((start, end, weight))
                total_weight += weight
                last_end = end
                print(f"   ✓ Selected (no overlap)")
                print(f"   Total weight: {total_weight}")
            else:
                print(f"   ✗ Rejected (overlaps with previous)")
            print()
        
        print(f"Greedy approximation result:")
        print(f"   Selected intervals: {[(s, e, w) for s, e, w in selected]}")
        print(f"   Total weight: {total_weight}")
        
        return selected, total_weight
    
    # ==========================================
    # 2. MEETING ROOMS PROBLEMS
    # ==========================================
    
    def meeting_rooms_i(self, intervals: List[List[int]]) -> bool:
        """
        Meeting Rooms I - Can Attend All Meetings
        
        Company: Facebook, Amazon
        Difficulty: Easy
        Time: O(n log n), Space: O(1)
        
        Problem: Determine if person can attend all meetings
        Greedy Strategy: Sort by start time and check overlaps
        """
        print("=== MEETING ROOMS I ===")
        print("Problem: Can a person attend all meetings?")
        print("Strategy: Sort by start time and check for overlaps")
        print()
        
        if not intervals:
            print("No meetings scheduled - can attend all!")
            return True
        
        print(f"Meeting intervals: {intervals}")
        
        # Sort by start time
        sorted_intervals = sorted(intervals)
        
        print(f"Sorted by start time: {sorted_intervals}")
        print()
        
        print("Checking for overlaps:")
        for i in range(len(sorted_intervals) - 1):
            current_start, current_end = sorted_intervals[i]
            next_start, next_end = sorted_intervals[i + 1]
            
            print(f"Meeting {i+1}: [{current_start}, {current_end}]")
            print(f"Meeting {i+2}: [{next_start}, {next_end}]")
            
            if current_end > next_start:
                print(f"   ✗ Overlap detected! Meeting ends at {current_end}, next starts at {next_start}")
                return False
            else:
                print(f"   ✓ No overlap (end {current_end} <= start {next_start})")
            print()
        
        print("✓ All meetings can be attended!")
        return True
    
    def meeting_rooms_ii(self, intervals: List[List[int]]) -> int:
        """
        Meeting Rooms II - Minimum Conference Rooms
        
        Company: Google, Facebook, Amazon
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        Problem: Find minimum number of conference rooms needed
        Greedy Strategy: Use heap to track room end times
        """
        print("=== MEETING ROOMS II ===")
        print("Problem: Find minimum number of conference rooms needed")
        print("Greedy Strategy: Use heap to track room end times")
        print()
        
        if not intervals:
            return 0
        
        print(f"Meeting intervals: {intervals}")
        
        # Sort by start time
        sorted_intervals = sorted(intervals)
        print(f"Sorted by start time: {sorted_intervals}")
        print()
        
        # Min heap to track room end times
        room_end_times = []
        max_rooms = 0
        
        print("Processing meetings:")
        for i, (start, end) in enumerate(sorted_intervals):
            print(f"Meeting {i+1}: [{start}, {end}]")
            
            # Remove rooms that are now free
            while room_end_times and room_end_times[0] <= start:
                freed_time = heapq.heappop(room_end_times)
                print(f"   Room freed at time {freed_time}")
            
            # Assign room for current meeting
            heapq.heappush(room_end_times, end)
            current_rooms = len(room_end_times)
            max_rooms = max(max_rooms, current_rooms)
            
            print(f"   Assigned room (ends at {end})")
            print(f"   Active rooms: {current_rooms}")
            print(f"   Room end times: {sorted(room_end_times)}")
            print()
        
        print(f"Minimum conference rooms needed: {max_rooms}")
        return max_rooms
    
    def meeting_rooms_iii_max_meetings(self, intervals: List[List[int]], rooms: int) -> int:
        """
        Meeting Rooms III - Maximum Meetings with Limited Rooms
        
        Company: Amazon, Google
        Difficulty: Hard
        Time: O(n log n + n log rooms), Space: O(rooms)
        
        Problem: Maximize meetings attended with limited rooms
        Greedy Strategy: Sort by end time, use earliest available room
        """
        print("=== MEETING ROOMS III ===")
        print("Problem: Maximize meetings with limited rooms")
        print("Greedy Strategy: Sort by end time, use earliest available room")
        print()
        
        if not intervals or rooms <= 0:
            return 0
        
        print(f"Available rooms: {rooms}")
        print(f"Meeting intervals: {intervals}")
        
        # Sort by end time (greedy choice)
        sorted_intervals = sorted(intervals, key=lambda x: x[1])
        print(f"Sorted by end time: {sorted_intervals}")
        print()
        
        # Track when each room becomes available
        room_available_times = [0] * rooms
        scheduled_meetings = []
        
        print("Scheduling process:")
        for i, (start, end) in enumerate(sorted_intervals):
            print(f"Meeting {i+1}: [{start}, {end}]")
            
            # Find earliest available room
            earliest_room = -1
            earliest_time = float('inf')
            
            for room_id in range(rooms):
                if room_available_times[room_id] <= start:
                    # Room is available
                    if room_available_times[room_id] < earliest_time:
                        earliest_time = room_available_times[room_id]
                        earliest_room = room_id
            
            if earliest_room != -1:
                # Schedule meeting in this room
                room_available_times[earliest_room] = end
                scheduled_meetings.append((start, end))
                print(f"   ✓ Scheduled in room {earliest_room + 1}")
                print(f"   Room {earliest_room + 1} available at: {end}")
            else:
                print(f"   ✗ No available room")
            
            print(f"   Room availability: {room_available_times}")
            print()
        
        print(f"Maximum meetings scheduled: {len(scheduled_meetings)}")
        print(f"Scheduled meetings: {scheduled_meetings}")
        
        return len(scheduled_meetings)
    
    # ==========================================
    # 3. TASK SCHEDULING PROBLEMS
    # ==========================================
    
    def task_scheduling_minimize_lateness(self, tasks: List[Tuple[str, int, int]]) -> Tuple[List[str], int]:
        """
        Task Scheduling to Minimize Maximum Lateness
        
        Company: Google, Microsoft
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        Problem: Schedule tasks to minimize maximum lateness
        Greedy Strategy: Earliest deadline first (EDF)
        
        Args:
            tasks: List of (task_name, duration, deadline)
        """
        print("=== MINIMIZE MAXIMUM LATENESS ===")
        print("Problem: Schedule tasks to minimize maximum lateness")
        print("Greedy Strategy: Earliest Deadline First (EDF)")
        print()
        
        print("Tasks (name, duration, deadline):")
        for name, duration, deadline in tasks:
            print(f"   {name}: duration={duration}, deadline={deadline}")
        print()
        
        # Sort by deadline (EDF strategy)
        sorted_tasks = sorted(tasks, key=lambda x: x[2])
        
        print("Sorted by deadline (EDF order):")
        for name, duration, deadline in sorted_tasks:
            print(f"   {name}: duration={duration}, deadline={deadline}")
        print()
        
        # Schedule tasks and calculate lateness
        current_time = 0
        schedule = []
        max_lateness = 0
        
        print("Scheduling process:")
        for i, (name, duration, deadline) in enumerate(sorted_tasks):
            start_time = current_time
            finish_time = current_time + duration
            lateness = max(0, finish_time - deadline)
            max_lateness = max(max_lateness, lateness)
            
            schedule.append(name)
            current_time = finish_time
            
            print(f"Task {i+1}: {name}")
            print(f"   Schedule: {start_time} -> {finish_time}")
            print(f"   Deadline: {deadline}")
            print(f"   Lateness: {lateness}")
            print(f"   Max lateness so far: {max_lateness}")
            print()
        
        print("Final schedule:")
        print(f"   Task order: {schedule}")
        print(f"   Maximum lateness: {max_lateness}")
        
        return schedule, max_lateness
    
    def task_scheduling_with_setup_times(self, tasks: List[Tuple[str, int]], setup_matrix: List[List[int]]) -> Tuple[List[str], int]:
        """
        Task Scheduling with Setup Times (Greedy Approximation)
        
        Company: Manufacturing systems, Google
        Difficulty: Hard
        Time: O(n²), Space: O(n)
        
        Problem: Schedule tasks minimizing total time including setup
        Greedy Strategy: Always choose next task with minimum setup time
        
        Args:
            tasks: List of (task_name, processing_time)
            setup_matrix: setup_matrix[i][j] = setup time from task i to task j
        """
        print("=== TASK SCHEDULING WITH SETUP TIMES ===")
        print("Problem: Minimize total time including setup between tasks")
        print("Greedy Strategy: Choose next task with minimum setup time")
        print()
        
        if not tasks:
            return [], 0
        
        print("Tasks (name, processing_time):")
        for i, (name, time) in enumerate(tasks):
            print(f"   {i}: {name} (processing: {time})")
        print()
        
        print("Setup time matrix:")
        print("    ", end="")
        for i in range(len(tasks)):
            print(f"{i:4}", end="")
        print()
        
        for i in range(len(setup_matrix)):
            print(f"{i:2}: ", end="")
            for j in range(len(setup_matrix[i])):
                print(f"{setup_matrix[i][j]:4}", end="")
            print()
        print()
        
        # Greedy scheduling
        n = len(tasks)
        visited = [False] * n
        schedule = []
        total_time = 0
        current_task = -1  # Start with no task (dummy task)
        
        print("Greedy scheduling process:")
        
        for step in range(n):
            best_task = -1
            min_total_cost = float('inf')
            
            print(f"Step {step + 1}:")
            
            # Find unvisited task with minimum setup + processing time
            for i in range(n):
                if not visited[i]:
                    setup_cost = 0 if current_task == -1 else setup_matrix[current_task][i]
                    processing_cost = tasks[i][1]
                    total_cost = setup_cost + processing_cost
                    
                    print(f"   Task {i} ({tasks[i][0]}): setup={setup_cost}, process={processing_cost}, total={total_cost}")
                    
                    if total_cost < min_total_cost:
                        min_total_cost = total_cost
                        best_task = i
            
            # Schedule the best task
            visited[best_task] = True
            schedule.append(tasks[best_task][0])
            
            setup_time = 0 if current_task == -1 else setup_matrix[current_task][best_task]
            processing_time = tasks[best_task][1]
            
            total_time += setup_time + processing_time
            current_task = best_task
            
            print(f"   ✓ Selected task {best_task} ({tasks[best_task][0]})")
            print(f"   Setup time: {setup_time}")
            print(f"   Processing time: {processing_time}")
            print(f"   Total time so far: {total_time}")
            print()
        
        print("Final schedule:")
        print(f"   Task order: {schedule}")
        print(f"   Total completion time: {total_time}")
        
        return schedule, total_time
    
    # ==========================================
    # 4. RESOURCE ALLOCATION PROBLEMS
    # ==========================================
    
    def car_pooling(self, trips: List[List[int]], capacity: int) -> bool:
        """
        Car Pooling Problem
        
        Company: Uber, Lyft, Amazon
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        Problem: Determine if all trips can be completed with car capacity
        Greedy Strategy: Process events (pickup/dropoff) in chronological order
        """
        print("=== CAR POOLING PROBLEM ===")
        print("Problem: Can all trips be completed with given car capacity?")
        print("Strategy: Process pickup/dropoff events chronologically")
        print()
        
        print(f"Car capacity: {capacity}")
        print("Trips [passengers, start_location, end_location]:")
        for i, trip in enumerate(trips):
            print(f"   Trip {i+1}: {trip}")
        print()
        
        # Create events for pickup and dropoff
        events = []
        for passengers, start, end in trips:
            events.append((start, passengers))    # Pickup
            events.append((end, -passengers))     # Dropoff
        
        # Sort events by location
        events.sort()
        
        print("Events (location, passenger_change):")
        for location, change in events:
            event_type = "pickup" if change > 0 else "dropoff"
            print(f"   Location {location}: {event_type} {abs(change)} passengers")
        print()
        
        # Process events and track current passengers
        current_passengers = 0
        
        print("Processing events:")
        for i, (location, change) in enumerate(events):
            current_passengers += change
            
            print(f"Event {i+1}: Location {location}")
            print(f"   Passenger change: {change:+d}")
            print(f"   Current passengers: {current_passengers}")
            
            if current_passengers > capacity:
                print(f"   ✗ Capacity exceeded! ({current_passengers} > {capacity})")
                return False
            elif current_passengers < 0:
                print(f"   ✗ Invalid state! Negative passengers")
                return False
            else:
                print(f"   ✓ Within capacity")
            print()
        
        print("✓ All trips can be completed!")
        return True
    
    def minimum_platforms(self, arrivals: List[int], departures: List[int]) -> int:
        """
        Minimum Railway Platforms Problem
        
        Company: Google, Amazon, Railway systems
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        Problem: Find minimum platforms needed for train schedule
        Greedy Strategy: Process arrival/departure events chronologically
        """
        print("=== MINIMUM RAILWAY PLATFORMS ===")
        print("Problem: Find minimum platforms needed for train schedule")
        print("Strategy: Process arrival/departure events chronologically")
        print()
        
        if len(arrivals) != len(departures):
            print("Error: Mismatched arrivals and departures!")
            return -1
        
        print("Train schedule:")
        for i in range(len(arrivals)):
            print(f"   Train {i+1}: arrives {arrivals[i]}, departs {departures[i]}")
        print()
        
        # Create events
        events = []
        for i in range(len(arrivals)):
            events.append((arrivals[i], 1))    # Arrival
            events.append((departures[i], -1)) # Departure
        
        # Sort events (departures before arrivals at same time)
        events.sort(key=lambda x: (x[0], x[1]))
        
        print("Events (time, type):")
        for time, event_type in events:
            event_name = "arrival" if event_type == 1 else "departure"
            print(f"   Time {time}: {event_name}")
        print()
        
        # Process events
        current_platforms = 0
        max_platforms = 0
        
        print("Processing events:")
        for i, (time, event_type) in enumerate(events):
            current_platforms += event_type
            max_platforms = max(max_platforms, current_platforms)
            
            event_name = "arrival" if event_type == 1 else "departure"
            print(f"Event {i+1}: Time {time} - {event_name}")
            print(f"   Current platforms needed: {current_platforms}")
            print(f"   Maximum so far: {max_platforms}")
            print()
        
        print(f"Minimum platforms required: {max_platforms}")
        return max_platforms


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_greedy_interval_problems():
    """Demonstrate all greedy interval and scheduling problems"""
    print("=== GREEDY INTERVAL PROBLEMS DEMONSTRATION ===\n")
    
    problems = GreedyIntervalProblems()
    
    # 1. Interval Scheduling
    print("1. INTERVAL SCHEDULING PROBLEMS")
    
    print("a) Interval Scheduling Maximization:")
    intervals = [(1, 3), (2, 4), (3, 6), (5, 7), (8, 9), (6, 10)]
    problems.interval_scheduling_maximization(intervals)
    print("\n" + "-"*40 + "\n")
    
    print("b) Weighted Interval Scheduling (Greedy Approximation):")
    weighted_intervals = [(1, 3, 50), (2, 5, 20), (3, 6, 10), (4, 7, 70), (6, 9, 60)]
    problems.interval_scheduling_weighted(weighted_intervals)
    print("\n" + "="*60 + "\n")
    
    # 2. Meeting Rooms
    print("2. MEETING ROOMS PROBLEMS")
    
    print("a) Meeting Rooms I:")
    meetings1 = [[0, 30], [5, 10], [15, 20]]
    problems.meeting_rooms_i(meetings1)
    print("\n" + "-"*20 + "\n")
    
    meetings2 = [[7, 10], [2, 4]]
    problems.meeting_rooms_i(meetings2)
    print("\n" + "-"*40 + "\n")
    
    print("b) Meeting Rooms II:")
    meetings = [[0, 30], [5, 10], [15, 20]]
    problems.meeting_rooms_ii(meetings)
    print("\n" + "-"*40 + "\n")
    
    print("c) Meeting Rooms III:")
    meetings3 = [[1, 5], [8, 9], [8, 9]]
    problems.meeting_rooms_iii_max_meetings(meetings3, 2)
    print("\n" + "="*60 + "\n")
    
    # 3. Task Scheduling
    print("3. TASK SCHEDULING PROBLEMS")
    
    print("a) Minimize Maximum Lateness:")
    tasks = [("T1", 3, 6), ("T2", 2, 8), ("T3", 1, 9), ("T4", 4, 9), ("T5", 3, 14), ("T6", 2, 15)]
    problems.task_scheduling_minimize_lateness(tasks)
    print("\n" + "-"*40 + "\n")
    
    print("b) Task Scheduling with Setup Times:")
    tasks_setup = [("A", 4), ("B", 3), ("C", 2)]
    setup_matrix = [
        [0, 2, 3],  # From A to A,B,C
        [1, 0, 4],  # From B to A,B,C  
        [2, 1, 0]   # From C to A,B,C
    ]
    problems.task_scheduling_with_setup_times(tasks_setup, setup_matrix)
    print("\n" + "="*60 + "\n")
    
    # 4. Resource Allocation
    print("4. RESOURCE ALLOCATION PROBLEMS")
    
    print("a) Car Pooling:")
    trips = [[2, 1, 5], [3, 3, 7]]
    problems.car_pooling(trips, 4)
    print("\n" + "-"*40 + "\n")
    
    print("b) Minimum Railway Platforms:")
    arrivals = [900, 940, 950, 1100, 1500, 1800]
    departures = [910, 1200, 1120, 1130, 1900, 2000]
    problems.minimum_platforms(arrivals, departures)


if __name__ == "__main__":
    demonstrate_greedy_interval_problems()
    
    print("\n=== INTERVAL PROBLEMS MASTERY GUIDE ===")
    
    print("\n🎯 PROBLEM PATTERNS:")
    print("• Interval Scheduling: Earliest end time first")
    print("• Meeting Rooms: Use heap to track resource availability")
    print("• Task Scheduling: Earliest deadline first (EDF)")
    print("• Resource Allocation: Event processing in chronological order")
    print("• Weighted Problems: Consider value/cost ratios")
    
    print("\n📊 COMPLEXITY ANALYSIS:")
    print("• Most problems: O(n log n) for sorting + O(n) for processing")
    print("• Meeting Rooms II: O(n log n) for heap operations")
    print("• Setup time problems: O(n²) for greedy selection")
    print("• Event processing: O(n log n) for sorting events")
    
    print("\n⚡ KEY STRATEGIES:")
    print("• Sort by appropriate criterion (end time, deadline, etc.)")
    print("• Use heaps/priority queues for resource tracking")
    print("• Convert to event processing problems when possible")
    print("• Consider greedy approximations for NP-hard problems")
    print("• Apply exchange argument for optimality proofs")
    
    print("\n🔧 IMPLEMENTATION TIPS:")
    print("• Always sort first according to greedy criterion")
    print("• Use appropriate data structures (heaps, sets)")
    print("• Handle edge cases (empty input, single interval)")
    print("• Consider tie-breaking rules")
    print("• Verify solutions with conflict checking")
    
    print("\n🏆 REAL-WORLD APPLICATIONS:")
    print("• Operating Systems: CPU scheduling, resource allocation")
    print("• Transportation: Route planning, vehicle scheduling")
    print("• Manufacturing: Production scheduling, machine allocation")
    print("• Healthcare: Surgery scheduling, room allocation")
    print("• Cloud Computing: Task scheduling, resource management")
    
    print("\n🎓 ADVANCED CONCEPTS:")
    print("• Weighted interval scheduling (DP vs greedy approximation)")
    print("• Online algorithms for interval problems")
    print("• Competitive analysis for scheduling")
    print("• Approximation ratios for greedy algorithms")
    print("• Multi-resource scheduling problems")
