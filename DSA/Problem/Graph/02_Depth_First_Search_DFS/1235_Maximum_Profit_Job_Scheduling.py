"""
1235. Maximum Profit in Job Scheduling - Multiple Approaches
Difficulty: Hard

We have n jobs, where every job is scheduled to be done from startTime[i] to endTime[i], obtaining a profit of profit[i].

You're given the startTime, endTime and profit arrays, return the maximum profit you can take such that there are no two jobs in the subset with overlapping time.
"""

from typing import List, Tuple
import bisect

class MaxProfitJobScheduling:
    """Multiple approaches for maximum profit job scheduling"""
    
    def jobScheduling_dp_binary_search(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        """
        Approach 1: Dynamic Programming with Binary Search
        
        Sort jobs by end time, use DP with binary search for optimization.
        
        Time: O(n log n), Space: O(n)
        """
        n = len(startTime)
        
        # Create jobs list and sort by end time
        jobs = list(zip(endTime, startTime, profit))
        jobs.sort()
        
        # dp[i] = maximum profit using jobs 0 to i
        dp = [0] * n
        dp[0] = jobs[0][2]  # First job's profit
        
        for i in range(1, n):
            # Option 1: Don't take current job
            profit_without_current = dp[i - 1]
            
            # Option 2: Take current job
            current_profit = jobs[i][2]
            
            # Find latest job that doesn't overlap with current job
            current_start = jobs[i][1]
            
            # Binary search for latest non-overlapping job
            left, right = 0, i - 1
            latest_non_overlapping = -1
            
            while left <= right:
                mid = (left + right) // 2
                if jobs[mid][0] <= current_start:  # jobs[mid] ends before current starts
                    latest_non_overlapping = mid
                    left = mid + 1
                else:
                    right = mid - 1
            
            profit_with_current = current_profit
            if latest_non_overlapping != -1:
                profit_with_current += dp[latest_non_overlapping]
            
            dp[i] = max(profit_without_current, profit_with_current)
        
        return dp[n - 1]
    
    def jobScheduling_recursive_memoization(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        """
        Approach 2: Recursive DP with Memoization
        
        Use recursion with memoization for cleaner code.
        
        Time: O(n log n), Space: O(n)
        """
        n = len(startTime)
        
        # Create jobs and sort by start time
        jobs = list(zip(startTime, endTime, profit))
        jobs.sort()
        
        memo = {}
        
        def find_next_job(current_end: int, start_idx: int) -> int:
            """Find next job that starts after current_end"""
            left, right = start_idx, n - 1
            result = n  # If no job found, return n
            
            while left <= right:
                mid = (left + right) // 2
                if jobs[mid][0] >= current_end:
                    result = mid
                    right = mid - 1
                else:
                    left = mid + 1
            
            return result
        
        def dp(index: int) -> int:
            """Maximum profit starting from job index"""
            if index >= n:
                return 0
            
            if index in memo:
                return memo[index]
            
            # Option 1: Skip current job
            skip_profit = dp(index + 1)
            
            # Option 2: Take current job
            current_start, current_end, current_profit = jobs[index]
            next_job_index = find_next_job(current_end, index + 1)
            take_profit = current_profit + dp(next_job_index)
            
            memo[index] = max(skip_profit, take_profit)
            return memo[index]
        
        return dp(0)
    
    def jobScheduling_iterative_dp(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        """
        Approach 3: Iterative DP with Optimized Search
        
        Bottom-up DP with efficient job searching.
        
        Time: O(n log n), Space: O(n)
        """
        n = len(startTime)
        
        # Create and sort jobs by end time
        jobs = [(endTime[i], startTime[i], profit[i]) for i in range(n)]
        jobs.sort()
        
        # Extract sorted end times for binary search
        end_times = [job[0] for job in jobs]
        
        dp = [0] * n
        dp[0] = jobs[0][2]
        
        for i in range(1, n):
            # Profit without taking current job
            without_current = dp[i - 1]
            
            # Profit with taking current job
            current_profit = jobs[i][2]
            
            # Find latest job that ends before current job starts
            current_start = jobs[i][1]
            latest_compatible = bisect.bisect_right(end_times, current_start) - 1
            
            with_current = current_profit
            if latest_compatible >= 0:
                with_current += dp[latest_compatible]
            
            dp[i] = max(without_current, with_current)
        
        return dp[n - 1]
    
    def jobScheduling_segment_tree(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        """
        Approach 4: Segment Tree for Range Maximum Query
        
        Use segment tree for efficient range maximum queries.
        
        Time: O(n log n), Space: O(n)
        """
        n = len(startTime)
        
        # Coordinate compression
        times = sorted(set(startTime + endTime))
        time_to_idx = {time: i for i, time in enumerate(times)}
        
        # Segment tree for range maximum query
        class SegmentTree:
            def __init__(self, size):
                self.size = size
                self.tree = [0] * (4 * size)
            
            def update(self, node, start, end, idx, val):
                if start == end:
                    self.tree[node] = max(self.tree[node], val)
                else:
                    mid = (start + end) // 2
                    if idx <= mid:
                        self.update(2 * node, start, mid, idx, val)
                    else:
                        self.update(2 * node + 1, mid + 1, end, idx, val)
                    self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])
            
            def query(self, node, start, end, l, r):
                if r < start or end < l:
                    return 0
                if l <= start and end <= r:
                    return self.tree[node]
                mid = (start + end) // 2
                return max(self.query(2 * node, start, mid, l, r),
                          self.query(2 * node + 1, mid + 1, end, l, r))
        
        # Create jobs and sort by end time
        jobs = [(endTime[i], startTime[i], profit[i]) for i in range(n)]
        jobs.sort()
        
        seg_tree = SegmentTree(len(times))
        max_profit = 0
        
        for end_time, start_time, job_profit in jobs:
            start_idx = time_to_idx[start_time]
            end_idx = time_to_idx[end_time]
            
            # Query maximum profit for jobs ending before current start time
            prev_max = seg_tree.query(1, 0, len(times) - 1, 0, start_idx)
            current_max = prev_max + job_profit
            
            # Update segment tree with current job's end time
            seg_tree.update(1, 0, len(times) - 1, end_idx, current_max)
            max_profit = max(max_profit, current_max)
        
        return max_profit
    
    def jobScheduling_greedy_intervals(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        """
        Approach 5: Greedy with Interval Scheduling
        
        Greedy approach with careful interval selection.
        
        Time: O(n²), Space: O(n)
        """
        n = len(startTime)
        
        # Create jobs with indices
        jobs = [(startTime[i], endTime[i], profit[i], i) for i in range(n)]
        jobs.sort(key=lambda x: x[1])  # Sort by end time
        
        # dp[i] = maximum profit considering jobs 0 to i
        dp = [0] * n
        dp[0] = jobs[0][2]
        
        for i in range(1, n):
            # Option 1: Don't include current job
            exclude_current = dp[i - 1]
            
            # Option 2: Include current job
            include_current = jobs[i][2]
            
            # Find latest non-overlapping job
            for j in range(i - 1, -1, -1):
                if jobs[j][1] <= jobs[i][0]:  # Non-overlapping
                    include_current += dp[j]
                    break
            
            dp[i] = max(exclude_current, include_current)
        
        return dp[n - 1]

def test_max_profit_job_scheduling():
    """Test maximum profit job scheduling algorithms"""
    solver = MaxProfitJobScheduling()
    
    test_cases = [
        ([1,2,3,3], [3,4,5,6], [50,10,40,70], 120, "Example 1"),
        ([1,2,3,4,6], [3,5,10,6,9], [20,20,100,70,60], 150, "Example 2"),
        ([1,1,1], [2,3,4], [5,6,4], 6, "Same start time"),
        ([4,2,4,8,1], [5,5,5,10,4], [1,2,8,10,4], 18, "Complex case"),
    ]
    
    algorithms = [
        ("DP + Binary Search", solver.jobScheduling_dp_binary_search),
        ("Recursive Memo", solver.jobScheduling_recursive_memoization),
        ("Iterative DP", solver.jobScheduling_iterative_dp),
        ("Greedy Intervals", solver.jobScheduling_greedy_intervals),
    ]
    
    print("=== Testing Maximum Profit Job Scheduling ===")
    
    for startTime, endTime, profit, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Start: {startTime}, End: {endTime}, Profit: {profit}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(startTime[:], endTime[:], profit[:])
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Max Profit: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_max_profit_job_scheduling()

"""
Maximum Profit Job Scheduling demonstrates advanced dynamic programming
techniques with interval scheduling and optimization strategies
for complex constraint satisfaction problems.
"""
