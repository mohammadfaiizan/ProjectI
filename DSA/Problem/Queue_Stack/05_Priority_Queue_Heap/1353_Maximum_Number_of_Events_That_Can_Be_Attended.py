"""
1353. Maximum Number of Events That Can Be Attended - Multiple Approaches
Difficulty: Medium

You are given an array of events where events[i] = [startDayi, endDayi]. Every event i starts at startDayi and ends at endDayi.

You can attend an event i at any day d where startTimei <= d <= endTimei. You can only attend one event at any time d.

Return the maximum number of events you can attend.
"""

from typing import List
import heapq

class MaximumNumberOfEventsThatCanBeAttended:
    """Multiple approaches to find maximum events that can be attended"""
    
    def maxEvents_greedy_heap(self, events: List[List[int]]) -> int:
        """
        Approach 1: Greedy with Min Heap (Optimal)
        
        Use greedy strategy with min heap to track earliest ending events.
        
        Time: O(n log n + d log n), Space: O(n) where d is max day
        """
        events.sort()  # Sort by start day
        
        heap = []  # Min heap of end days
        attended = 0
        event_idx = 0
        day = 1
        
        while event_idx < len(events) or heap:
            # Add all events that start today
            while event_idx < len(events) and events[event_idx][0] <= day:
                heapq.heappush(heap, events[event_idx][1])
                event_idx += 1
            
            # Remove expired events
            while heap and heap[0] < day:
                heapq.heappop(heap)
            
            # Attend the event that ends earliest
            if heap:
                heapq.heappop(heap)
                attended += 1
            
            day += 1
        
        return attended
    
    def maxEvents_greedy_sorting(self, events: List[List[int]]) -> int:
        """
        Approach 2: Greedy with Sorting
        
        Sort events by end day and attend greedily.
        
        Time: O(n log n + n * d), Space: O(d)
        """
        # Sort by end day, then by start day
        events.sort(key=lambda x: (x[1], x[0]))
        
        attended = set()  # Set of days when events are attended
        count = 0
        
        for start, end in events:
            # Find the earliest available day in [start, end]
            for day in range(start, end + 1):
                if day not in attended:
                    attended.add(day)
                    count += 1
                    break
        
        return count
    
    def maxEvents_interval_scheduling(self, events: List[List[int]]) -> int:
        """
        Approach 3: Interval Scheduling
        
        Classic interval scheduling with earliest deadline first.
        
        Time: O(n log n), Space: O(n)
        """
        # Sort by end day
        events.sort(key=lambda x: x[1])
        
        attended = 0
        last_day = 0
        
        for start, end in events:
            # If we can attend this event after the last attended event
            if start > last_day:
                attended += 1
                last_day = start
            elif end > last_day:
                attended += 1
                last_day = last_day + 1 if last_day >= start else start
        
        return attended
    
    def maxEvents_dp_approach(self, events: List[List[int]]) -> int:
        """
        Approach 4: Dynamic Programming
        
        Use DP to track maximum events for each day.
        
        Time: O(n * d), Space: O(d)
        """
        if not events:
            return 0
        
        max_day = max(event[1] for event in events)
        
        # dp[day] = maximum events that can be attended up to day
        dp = [0] * (max_day + 2)
        
        # Sort events by start day
        events.sort()
        
        for day in range(1, max_day + 1):
            dp[day] = dp[day - 1]  # Don't attend any event today
            
            # Try attending each event that can be attended today
            for start, end in events:
                if start <= day <= end:
                    # Find the latest day before start when we could have attended
                    prev_max = dp[start - 1] if start > 0 else 0
                    dp[day] = max(dp[day], prev_max + 1)
        
        return dp[max_day]
    
    def maxEvents_brute_force(self, events: List[List[int]]) -> int:
        """
        Approach 5: Brute Force
        
        Try all possible combinations of events.
        
        Time: O(2^n), Space: O(n)
        """
        def can_attend(selected_events):
            """Check if selected events can be attended without conflict"""
            days_used = set()
            
            for start, end in selected_events:
                # Find an available day in [start, end]
                found = False
                for day in range(start, end + 1):
                    if day not in days_used:
                        days_used.add(day)
                        found = True
                        break
                
                if not found:
                    return False
            
            return True
        
        max_events = 0
        n = len(events)
        
        # Try all possible subsets
        for mask in range(1 << n):
            selected = []
            for i in range(n):
                if mask & (1 << i):
                    selected.append(events[i])
            
            if can_attend(selected):
                max_events = max(max_events, len(selected))
        
        return max_events


def test_maximum_number_of_events():
    """Test maximum number of events algorithms"""
    solver = MaximumNumberOfEventsThatCanBeAttended()
    
    test_cases = [
        ([[1,2],[2,3],[3,4]], 3, "Example 1"),
        ([[1,2],[2,3],[3,4],[1,2]], 4, "Example 2"),
        ([[1,4],[4,4],[2,2],[3,4],[1,1]], 4, "Example 3"),
        ([[1,1]], 1, "Single event"),
        ([[1,2],[1,2]], 2, "Two overlapping events"),
        ([[1,3],[2,4],[3,5]], 3, "Chain of events"),
        ([[1,1],[2,2],[3,3]], 3, "Non-overlapping events"),
        ([[1,5],[2,3],[4,6]], 3, "Mixed overlaps"),
    ]
    
    algorithms = [
        ("Greedy Heap", solver.maxEvents_greedy_heap),
        ("Greedy Sorting", solver.maxEvents_greedy_sorting),
        ("Interval Scheduling", solver.maxEvents_interval_scheduling),
        ("DP Approach", solver.maxEvents_dp_approach),
        # ("Brute Force", solver.maxEvents_brute_force),  # Too slow for large inputs
    ]
    
    print("=== Testing Maximum Number of Events That Can Be Attended ===")
    
    for events, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Events: {events}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(events[:])  # Pass copy
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_greedy_heap_approach():
    """Demonstrate greedy heap approach step by step"""
    print("\n=== Greedy Heap Approach Step-by-Step Demo ===")
    
    events = [[1,2],[2,3],[3,4],[1,2]]
    print(f"Events: {events}")
    
    # Sort by start day
    events.sort()
    print(f"Sorted by start day: {events}")
    
    heap = []
    attended = 0
    event_idx = 0
    day = 1
    
    print(f"\nSimulation:")
    
    while event_idx < len(events) or heap:
        print(f"\nDay {day}:")
        
        # Add events that start today
        added_today = []
        while event_idx < len(events) and events[event_idx][0] <= day:
            end_day = events[event_idx][1]
            heapq.heappush(heap, end_day)
            added_today.append(events[event_idx])
            event_idx += 1
        
        if added_today:
            print(f"  Events starting today or earlier: {added_today}")
        
        # Remove expired events
        expired = []
        while heap and heap[0] < day:
            expired.append(heapq.heappop(heap))
        
        if expired:
            print(f"  Expired events (end days): {expired}")
        
        print(f"  Available events (end days): {sorted(heap)}")
        
        # Attend earliest ending event
        if heap:
            earliest_end = heapq.heappop(heap)
            attended += 1
            print(f"  Attended event ending on day {earliest_end}")
            print(f"  Total attended so far: {attended}")
        else:
            print(f"  No events to attend today")
        
        day += 1
        
        # Stop if no more events to process
        if event_idx >= len(events) and not heap:
            break
    
    print(f"\nFinal result: {attended} events attended")


if __name__ == "__main__":
    test_maximum_number_of_events()
    demonstrate_greedy_heap_approach()

"""
Maximum Number of Events That Can Be Attended demonstrates heap applications
for interval scheduling problems, including greedy optimization strategies
and multiple approaches for event attendance maximization.
"""
