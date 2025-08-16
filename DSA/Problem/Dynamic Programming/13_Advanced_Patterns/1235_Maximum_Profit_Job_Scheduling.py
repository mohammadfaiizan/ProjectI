"""
LeetCode 1235: Maximum Profit in Job Scheduling
Difficulty: Hard
Category: Advanced DP - DP with Data Structures

PROBLEM DESCRIPTION:
===================
We have n jobs, where every job is scheduled to be done from startTime[i] to endTime[i], obtaining a profit of profit[i].

You're given the startTime, endTime and profit arrays, return the maximum profit you can take such that there are no two jobs in the subset with overlapping time.

Example 1:
Input: startTime = [1,2,3,3], endTime = [3,4,5,6], profit = [50,10,40,70]
Output: 120
Explanation: The subset chosen is the first and fourth job. 
Time range [1-3]+[3-6] , we get profit of 120 = 50 + 70.

Example 2:
Input: startTime = [1,2,3,4,6], endTime = [3,5,10,6,9], profit = [20,20,100,70,60]
Output: 150
Explanation: The subset chosen is the first, fourth and fifth job. 
Profit obtained 150 = 20 + 70 + 60.

Example 3:
Input: startTime = [1,1,1], endTime = [2,3,3], profit = [5,6,4]
Output: 6

Constraints:
- 1 <= startTime.length == endTime.length == profit.length <= 5 * 10^4
- 1 <= startTime[i] < endTime[i] <= 10^9
- 1 <= profit[i] <= 10^4
"""


def job_scheduling_basic_dp(startTime, endTime, profit):
    """
    BASIC DP APPROACH:
    =================
    Sort by end time and use DP with binary search.
    
    Time Complexity: O(n^2) - DP with linear search
    Space Complexity: O(n) - DP array
    """
    n = len(startTime)
    
    # Create jobs list and sort by end time
    jobs = list(zip(startTime, endTime, profit))
    jobs.sort(key=lambda x: x[1])  # Sort by end time
    
    # dp[i] = maximum profit using jobs[0:i+1]
    dp = [0] * n
    dp[0] = jobs[0][2]  # First job's profit
    
    for i in range(1, n):
        # Option 1: Don't take current job
        current_profit = dp[i - 1]
        
        # Option 2: Take current job
        job_profit = jobs[i][2]
        
        # Find latest non-overlapping job
        latest_compatible = -1
        for j in range(i - 1, -1, -1):
            if jobs[j][1] <= jobs[i][0]:  # jobs[j] ends before jobs[i] starts
                latest_compatible = j
                break
        
        if latest_compatible != -1:
            job_profit += dp[latest_compatible]
        
        dp[i] = max(current_profit, job_profit)
    
    return dp[n - 1]


def job_scheduling_binary_search(startTime, endTime, profit):
    """
    OPTIMIZED DP WITH BINARY SEARCH:
    ===============================
    Use binary search to find latest compatible job.
    
    Time Complexity: O(n log n) - sorting + binary search
    Space Complexity: O(n) - DP array
    """
    n = len(startTime)
    
    # Create jobs and sort by end time
    jobs = list(zip(startTime, endTime, profit))
    jobs.sort(key=lambda x: x[1])
    
    def binary_search_latest_compatible(i):
        """Find latest job that ends before jobs[i] starts"""
        left, right = 0, i - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if jobs[mid][1] <= jobs[i][0]:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    # DP
    dp = [0] * n
    dp[0] = jobs[0][2]
    
    for i in range(1, n):
        # Option 1: Don't take current job
        dont_take = dp[i - 1]
        
        # Option 2: Take current job
        take = jobs[i][2]
        latest_compatible = binary_search_latest_compatible(i)
        
        if latest_compatible != -1:
            take += dp[latest_compatible]
        
        dp[i] = max(dont_take, take)
    
    return dp[n - 1]


def job_scheduling_segment_tree(startTime, endTime, profit):
    """
    SEGMENT TREE APPROACH:
    =====================
    Use segment tree for range maximum queries.
    
    Time Complexity: O(n log n) - coordinate compression + segment tree
    Space Complexity: O(n) - segment tree
    """
    import bisect
    
    n = len(startTime)
    jobs = list(zip(startTime, endTime, profit))
    
    # Coordinate compression
    all_times = set()
    for start, end, _ in jobs:
        all_times.add(start)
        all_times.add(end)
    
    sorted_times = sorted(all_times)
    time_to_index = {time: i for i, time in enumerate(sorted_times)}
    
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
            left_max = self.query(2 * node, start, mid, l, r)
            right_max = self.query(2 * node + 1, mid + 1, end, l, r)
            return max(left_max, right_max)
        
        def update_point(self, idx, val):
            self.update(1, 0, self.size - 1, idx, val)
        
        def query_range(self, l, r):
            if l > r:
                return 0
            return self.query(1, 0, self.size - 1, l, r)
    
    # Sort jobs by end time
    jobs.sort(key=lambda x: x[1])
    
    seg_tree = SegmentTree(len(sorted_times))
    max_profit = 0
    
    for start, end, prof in jobs:
        start_idx = time_to_index[start]
        end_idx = time_to_index[end]
        
        # Query maximum profit for jobs ending before current start
        prev_max = seg_tree.query_range(0, start_idx - 1) if start_idx > 0 else 0
        
        # Current profit if we take this job
        current_profit = prev_max + prof
        max_profit = max(max_profit, current_profit)
        
        # Update segment tree at end time
        seg_tree.update_point(end_idx, current_profit)
    
    return max_profit


def job_scheduling_coordinate_compression_binary_indexed_tree(startTime, endTime, profit):
    """
    BINARY INDEXED TREE APPROACH:
    =============================
    Use coordinate compression + BIT for efficient range maximum queries.
    
    Time Complexity: O(n log n) - sorting + BIT operations
    Space Complexity: O(n) - BIT structure
    """
    n = len(startTime)
    jobs = list(zip(startTime, endTime, profit))
    
    # Coordinate compression
    all_times = sorted(set(startTime + endTime))
    time_to_idx = {time: i + 1 for i, time in enumerate(all_times)}  # 1-indexed for BIT
    
    class BIT:
        def __init__(self, size):
            self.size = size
            self.tree = [0] * (size + 1)
        
        def update(self, idx, val):
            while idx <= self.size:
                self.tree[idx] = max(self.tree[idx], val)
                idx += idx & (-idx)
        
        def query(self, idx):
            result = 0
            while idx > 0:
                result = max(result, self.tree[idx])
                idx -= idx & (-idx)
            return result
    
    # Sort jobs by end time
    jobs.sort(key=lambda x: x[1])
    
    bit = BIT(len(all_times))
    
    for start, end, prof in jobs:
        start_idx = time_to_idx[start]
        end_idx = time_to_idx[end]
        
        # Get maximum profit from jobs ending before current start
        prev_max = bit.query(start_idx - 1) if start_idx > 1 else 0
        
        # Update BIT with current job's contribution
        current_profit = prev_max + prof
        bit.update(end_idx, current_profit)
    
    # Return maximum profit overall
    return bit.query(len(all_times))


def job_scheduling_with_analysis(startTime, endTime, profit):
    """
    JOB SCHEDULING WITH DETAILED ANALYSIS:
    =====================================
    Solve with comprehensive analysis and multiple approaches.
    
    Time Complexity: O(n log n) - optimal approach
    Space Complexity: O(n) - analysis data
    """
    n = len(startTime)
    
    analysis = {
        'num_jobs': n,
        'time_range': (min(startTime), max(endTime)),
        'total_profit_available': sum(profit),
        'job_details': [],
        'selected_jobs': [],
        'overlap_analysis': {},
        'optimization_insights': []
    }
    
    # Analyze individual jobs
    jobs = []
    for i in range(n):
        job_info = {
            'index': i,
            'start': startTime[i],
            'end': endTime[i],
            'profit': profit[i],
            'duration': endTime[i] - startTime[i],
            'profit_per_unit_time': profit[i] / (endTime[i] - startTime[i])
        }
        jobs.append((startTime[i], endTime[i], profit[i], i))
        analysis['job_details'].append(job_info)
    
    # Sort by end time for DP
    jobs.sort(key=lambda x: x[1])
    
    def find_latest_compatible(i):
        left, right = 0, i - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if jobs[mid][1] <= jobs[i][0]:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    # DP with job tracking
    dp = [0] * n
    parent = [-1] * n
    
    dp[0] = jobs[0][2]
    
    for i in range(1, n):
        # Option 1: Don't take current job
        dont_take = dp[i - 1]
        
        # Option 2: Take current job
        take = jobs[i][2]
        latest_compatible = find_latest_compatible(i)
        
        if latest_compatible != -1:
            take += dp[latest_compatible]
        
        if take > dont_take:
            dp[i] = take
            parent[i] = latest_compatible
        else:
            dp[i] = dont_take
            parent[i] = i - 1
    
    # Reconstruct solution
    selected = []
    
    def reconstruct(i):
        if i == -1:
            return
        
        if i == 0 or (parent[i] != i - 1):
            # Job i was selected
            selected.append(jobs[i][3])  # Original index
            reconstruct(parent[i])
        else:
            # Job i was not selected
            reconstruct(i - 1)
    
    reconstruct(n - 1)
    
    # Analyze selected jobs
    selected.sort()
    for idx in selected:
        analysis['selected_jobs'].append({
            'original_index': idx,
            'start': startTime[idx],
            'end': endTime[idx],
            'profit': profit[idx]
        })
    
    # Generate insights
    analysis['optimization_insights'].append(f"Maximum profit: {dp[n - 1]}")
    analysis['optimization_insights'].append(f"Jobs selected: {len(selected)}/{n}")
    
    efficiency = dp[n - 1] / analysis['total_profit_available']
    analysis['optimization_insights'].append(f"Profit efficiency: {efficiency:.2%}")
    
    if len(selected) > 0:
        avg_duration = sum(endTime[i] - startTime[i] for i in selected) / len(selected)
        analysis['optimization_insights'].append(f"Average selected job duration: {avg_duration:.1f}")
    
    # Analyze overlaps
    overlap_count = 0
    for i in range(n):
        overlaps_with = []
        for j in range(n):
            if i != j and not (endTime[i] <= startTime[j] or endTime[j] <= startTime[i]):
                overlaps_with.append(j)
        overlap_count += len(overlaps_with)
    
    analysis['overlap_analysis'] = {
        'total_overlaps': overlap_count // 2,  # Each overlap counted twice
        'avg_overlaps_per_job': overlap_count / n / 2
    }
    
    return dp[n - 1], analysis


def job_scheduling_analysis(startTime, endTime, profit):
    """
    COMPREHENSIVE JOB SCHEDULING ANALYSIS:
    =====================================
    Analyze the problem with multiple approaches and optimizations.
    """
    print(f"Job Scheduling Analysis:")
    print(f"Number of jobs: {len(startTime)}")
    print(f"Time range: [{min(startTime)}, {max(endTime)}]")
    print(f"Total available profit: {sum(profit)}")
    
    # Show jobs
    print(f"\nJob Details:")
    for i in range(len(startTime)):
        duration = endTime[i] - startTime[i]
        profit_rate = profit[i] / duration
        print(f"  Job {i}: [{startTime[i]:3d}, {endTime[i]:3d}] profit={profit[i]:3d} "
              f"duration={duration:2d} rate={profit_rate:.1f}")
    
    # Different approaches
    basic_result = job_scheduling_basic_dp(startTime[:], endTime[:], profit[:])
    binary_search_result = job_scheduling_binary_search(startTime[:], endTime[:], profit[:])
    
    print(f"\nResults:")
    print(f"Basic DP:        {basic_result}")
    print(f"Binary Search:   {binary_search_result}")
    
    # Advanced data structure approaches for larger inputs
    if len(startTime) <= 1000:
        try:
            segment_tree_result = job_scheduling_segment_tree(startTime[:], endTime[:], profit[:])
            print(f"Segment Tree:    {segment_tree_result}")
        except:
            print("Segment Tree:    Implementation complexity")
        
        try:
            bit_result = job_scheduling_coordinate_compression_binary_indexed_tree(startTime[:], endTime[:], profit[:])
            print(f"BIT:             {bit_result}")
        except:
            print("BIT:             Implementation complexity")
    
    # Detailed analysis
    detailed_result, analysis = job_scheduling_with_analysis(startTime[:], endTime[:], profit[:])
    
    print(f"\nDetailed Analysis:")
    print(f"Optimal profit: {detailed_result}")
    
    print(f"\nSelected Jobs:")
    for job in analysis['selected_jobs']:
        print(f"  Job {job['original_index']}: [{job['start']}, {job['end']}] profit={job['profit']}")
    
    print(f"\nOptimization Insights:")
    for insight in analysis['optimization_insights']:
        print(f"  • {insight}")
    
    print(f"\nOverlap Analysis:")
    overlap = analysis['overlap_analysis']
    print(f"  • Total overlapping pairs: {overlap['total_overlaps']}")
    print(f"  • Average overlaps per job: {overlap['avg_overlaps_per_job']:.1f}")
    
    return detailed_result


def advanced_scheduling_variants():
    """
    ADVANCED SCHEDULING VARIANTS:
    ============================
    Demonstrate various scheduling problem extensions.
    """
    
    def weighted_job_scheduling_with_deadlines(jobs, deadlines):
        """Job scheduling with deadlines"""
        # jobs = [(start, end, profit), ...]
        # deadlines = [deadline for each job]
        
        n = len(jobs)
        job_data = []
        
        for i, ((start, end, profit), deadline) in enumerate(zip(jobs, deadlines)):
            if end <= deadline:  # Job is feasible
                job_data.append((start, end, profit, i))
        
        # Sort by end time
        job_data.sort(key=lambda x: x[1])
        
        if not job_data:
            return 0
        
        # Standard DP approach
        dp = [0] * len(job_data)
        dp[0] = job_data[0][2]
        
        for i in range(1, len(job_data)):
            dont_take = dp[i - 1]
            take = job_data[i][2]
            
            # Find latest compatible job
            for j in range(i - 1, -1, -1):
                if job_data[j][1] <= job_data[i][0]:
                    take += dp[j]
                    break
            
            dp[i] = max(dont_take, take)
        
        return dp[-1]
    
    def job_scheduling_with_setup_times(jobs, setup_times):
        """Job scheduling with setup times between jobs"""
        # setup_times[i][j] = setup time from job i to job j
        
        n = len(jobs)
        jobs_with_idx = [(jobs[i][0], jobs[i][1], jobs[i][2], i) for i in range(n)]
        jobs_with_idx.sort(key=lambda x: x[1])  # Sort by end time
        
        dp = [0] * n
        dp[0] = jobs_with_idx[0][2]
        
        for i in range(1, n):
            dont_take = dp[i - 1]
            take = jobs_with_idx[i][2]
            
            best_prev = 0
            curr_start = jobs_with_idx[i][0]
            curr_idx = jobs_with_idx[i][3]
            
            for j in range(i - 1, -1, -1):
                prev_end = jobs_with_idx[j][1]
                prev_idx = jobs_with_idx[j][3]
                setup_time = setup_times[prev_idx][curr_idx]
                
                if prev_end + setup_time <= curr_start:
                    best_prev = max(best_prev, dp[j])
                    break
            
            take += best_prev
            dp[i] = max(dont_take, take)
        
        return dp[-1]
    
    def machine_scheduling_multiple_machines(jobs, num_machines):
        """Schedule jobs on multiple machines"""
        # Each job can be assigned to any machine
        # Minimize makespan (maximum completion time)
        
        jobs.sort(key=lambda x: x[2], reverse=True)  # Sort by profit descending
        
        machine_end_times = [0] * num_machines
        machine_profits = [0] * num_machines
        
        for start, end, profit in jobs:
            duration = end - start
            
            # Find machine that can start this job earliest
            best_machine = 0
            earliest_start = machine_end_times[0]
            
            for m in range(1, num_machines):
                if machine_end_times[m] < earliest_start:
                    earliest_start = machine_end_times[m]
                    best_machine = m
            
            # Assign job to best machine
            actual_start = max(earliest_start, start)
            machine_end_times[best_machine] = actual_start + duration
            machine_profits[best_machine] += profit
        
        return sum(machine_profits), max(machine_end_times)
    
    # Test variants
    print("Advanced Scheduling Variants:")
    print("=" * 40)
    
    # Example jobs
    jobs = [(1, 3, 50), (2, 4, 10), (3, 5, 40), (3, 6, 70)]
    
    print(f"\nBase jobs: {jobs}")
    
    # With deadlines
    deadlines = [4, 5, 6, 7]
    deadline_result = weighted_job_scheduling_with_deadlines(jobs, deadlines)
    print(f"With deadlines {deadlines}: {deadline_result}")
    
    # With setup times
    setup_matrix = [
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0]
    ]
    setup_result = job_scheduling_with_setup_times(jobs, setup_matrix)
    print(f"With setup times: {setup_result}")
    
    # Multiple machines
    profit_sum, makespan = machine_scheduling_multiple_machines(jobs, 2)
    print(f"2 machines - Total profit: {profit_sum}, Makespan: {makespan}")


# Test cases
def test_job_scheduling():
    """Test job scheduling implementations"""
    test_cases = [
        ([1,2,3,3], [3,4,5,6], [50,10,40,70], 120),
        ([1,2,3,4,6], [3,5,10,6,9], [20,20,100,70,60], 150),
        ([1,1,1], [2,3,3], [5,6,4], 6),
        ([4,2,4,8,1], [5,5,5,10,2], [1,2,8,10,4], 18)
    ]
    
    print("Testing Job Scheduling Solutions:")
    print("=" * 70)
    
    for i, (start, end, profit, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"start = {start}")
        print(f"end = {end}")
        print(f"profit = {profit}")
        print(f"Expected: {expected}")
        
        basic = job_scheduling_basic_dp(start[:], end[:], profit[:])
        binary_search = job_scheduling_binary_search(start[:], end[:], profit[:])
        
        print(f"Basic DP:         {basic:>3} {'✓' if basic == expected else '✗'}")
        print(f"Binary Search:    {binary_search:>3} {'✓' if binary_search == expected else '✗'}")
        
        # Advanced methods for reasonable input sizes
        if len(start) <= 100:
            try:
                seg_tree = job_scheduling_segment_tree(start[:], end[:], profit[:])
                print(f"Segment Tree:     {seg_tree:>3} {'✓' if seg_tree == expected else '✗'}")
            except:
                print("Segment Tree:     Error")
            
            try:
                bit_result = job_scheduling_coordinate_compression_binary_indexed_tree(start[:], end[:], profit[:])
                print(f"BIT:              {bit_result:>3} {'✓' if bit_result == expected else '✗'}")
            except:
                print("BIT:              Error")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    job_scheduling_analysis([1,2,3,3], [3,4,5,6], [50,10,40,70])
    
    # Advanced variants
    print(f"\n" + "=" * 70)
    advanced_scheduling_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. SORTING STRATEGY: Sort by end time for optimal substructure")
    print("2. BINARY SEARCH: Find latest compatible job efficiently")
    print("3. DATA STRUCTURES: Segment tree/BIT for advanced optimizations")
    print("4. COORDINATE COMPRESSION: Handle large time ranges efficiently")
    print("5. DP OPTIMIZATION: Choose vs skip decision at each job")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Resource Scheduling: CPU, memory, network bandwidth allocation")
    print("• Meeting Scheduling: Conference room and calendar optimization")
    print("• Project Management: Task scheduling with dependencies")
    print("• Manufacturing: Machine scheduling and production planning")
    print("• Cloud Computing: Virtual machine and container scheduling")


if __name__ == "__main__":
    test_job_scheduling()


"""
MAXIMUM PROFIT JOB SCHEDULING - DP WITH ADVANCED DATA STRUCTURES:
=================================================================

This problem demonstrates sophisticated DP optimization using data structures:
- Binary search for efficient predecessor finding
- Coordinate compression for large time ranges
- Segment trees and BIT for range queries
- Complex state transitions with multiple optimization paths

KEY INSIGHTS:
============
1. **SORTING STRATEGY**: Sort by end time to enable optimal substructure
2. **BINARY SEARCH OPTIMIZATION**: Find latest compatible job in O(log n)
3. **DATA STRUCTURE INTEGRATION**: Segment tree/BIT for advanced queries
4. **COORDINATE COMPRESSION**: Handle sparse time ranges efficiently
5. **SPACE-TIME TRADEOFFS**: Multiple approaches with different complexities

ALGORITHM APPROACHES:
====================

1. **Basic DP**: O(n²) time, O(n) space
   - Linear search for compatible jobs
   - Clear but inefficient for large inputs

2. **Binary Search DP**: O(n log n) time, O(n) space
   - Binary search for latest compatible job
   - Standard optimal approach

3. **Segment Tree**: O(n log n) time, O(n) space
   - Coordinate compression + range maximum queries
   - Handles complex query patterns

4. **Binary Indexed Tree**: O(n log n) time, O(n) space
   - Efficient range maximum queries
   - Space-efficient implementation

CORE DP RECURRENCE:
==================
**State**: dp[i] = maximum profit using jobs 0..i (sorted by end time)
**Transition**: dp[i] = max(dp[i-1], profit[i] + dp[j])
where j is the latest job that ends before job i starts

**Key Optimization**: Finding j efficiently using binary search

BINARY SEARCH OPTIMIZATION:
===========================
```python
def find_latest_compatible(jobs, i):
    left, right = 0, i - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        if jobs[mid][1] <= jobs[i][0]:  # jobs[mid] ends before jobs[i] starts
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

COORDINATE COMPRESSION:
======================
**Problem**: Time values can be up to 10⁹
**Solution**: Map time values to compressed indices
```python
all_times = sorted(set(start_times + end_times))
time_to_index = {time: i for i, time in enumerate(all_times)}
```

SEGMENT TREE INTEGRATION:
=========================
**Purpose**: Range maximum queries for DP optimization
**Operations**:
- Update(time, profit): Set maximum profit achievable by time
- Query(0, time): Get maximum profit achievable by time

ADVANCED DATA STRUCTURE PATTERNS:
=================================

**Segment Tree for DP**:
- Maintains maximum profit for each time range
- Enables efficient "best profit before time t" queries
- O(log n) updates and queries

**BIT for DP**:
- Simpler implementation than segment tree
- Efficient for maximum queries on prefixes
- Lower constant factors

COMPLEXITY ANALYSIS:
===================
**Basic Approach**: O(n²) - quadratic in number of jobs
**Optimized Approach**: O(n log n) - optimal for comparison-based sorting
**Space Complexity**: O(n) for DP array + O(n) for data structures

**Practical Performance**: 
- n ≤ 1000: All approaches viable
- n ≤ 50000: Optimized approaches essential

IMPLEMENTATION CONSIDERATIONS:
=============================

**Sorting**: Critical for enabling DP transitions
**Edge Cases**: Single job, no jobs, all overlapping jobs
**Integer Overflow**: Large profit sums may require long integers
**Coordinate Compression**: Essential for sparse time ranges

REAL-WORLD APPLICATIONS:
=======================
- **Cloud Computing**: VM scheduling for maximum resource utilization
- **Manufacturing**: Machine scheduling with setup times
- **Finance**: Trade execution scheduling
- **Healthcare**: Operating room scheduling
- **Transportation**: Vehicle routing and scheduling

RELATED PROBLEMS:
================
- **Activity Selection**: Greedy variant without profits
- **Weighted Interval Scheduling**: Classic DP problem
- **Machine Scheduling**: Multiple machines extension
- **Project Scheduling**: With dependencies and resources

This problem showcases how sophisticated data structures
can transform DP algorithms from quadratic to logarithmic
complexity, enabling solution of much larger problem instances
while maintaining optimal results.
"""
