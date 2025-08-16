"""
LeetCode 1723: Find Minimum Time to Finish All Jobs
Difficulty: Hard
Category: Bitmask DP - Job Scheduling Optimization

PROBLEM DESCRIPTION:
===================
You are given an integer array jobs, where jobs[i] is the amount of time it takes to complete the ith job.

There are k workers available. You can assign jobs to workers such that some workers stay idle or a worker can be assigned multiple jobs.

Jobs cannot be split among multiple workers.

The goal is to devise an assignment such that the maximum working time of any worker is minimized.

Return the minimum possible maximum working time of any worker.

Example 1:
Input: jobs = [3,2,3], k = 3
Output: 3
Explanation: By assigning each person one job, the maximum time is 3.

Example 2:
Input: jobs = [1,2,4,7,8], k = 2
Output: 11
Explanation: Assign the jobs the following way:
Worker 1: 1, 2, 8 (working time = 1 + 2 + 8 = 11)
Worker 2: 4, 7 (working time = 4 + 7 = 11)
The maximum working time is 11.

Constraints:
- 1 <= k <= jobs.length <= 12
- 1 <= jobs[i] <= 10^7
"""


def minimum_time_to_finish_jobs_backtrack(jobs, k):
    """
    BACKTRACKING APPROACH:
    =====================
    Try all possible job assignments using backtracking.
    
    Time Complexity: O(k^n) - exponential
    Space Complexity: O(n) - recursion stack
    """
    n = len(jobs)
    jobs.sort(reverse=True)  # Optimization: assign larger jobs first
    
    min_time = float('inf')
    workers = [0] * k
    
    def backtrack(job_idx):
        nonlocal min_time
        
        if job_idx == n:
            min_time = min(min_time, max(workers))
            return
        
        # Pruning: if current max already >= min_time, skip
        if max(workers) >= min_time:
            return
        
        # Try assigning current job to each worker
        for i in range(k):
            # Optimization: if worker i is idle and worker i+1 is also idle, skip
            if i > 0 and workers[i] == workers[i-1]:
                continue
            
            workers[i] += jobs[job_idx]
            backtrack(job_idx + 1)
            workers[i] -= jobs[job_idx]
    
    backtrack(0)
    return min_time


def minimum_time_to_finish_jobs_bitmask_dp(jobs, k):
    """
    BITMASK DP APPROACH:
    ===================
    Use bitmask DP to find optimal job distribution.
    
    Time Complexity: O(3^n) - iterate through all subset pairs
    Space Complexity: O(2^n) - DP table
    """
    n = len(jobs)
    
    # Precompute subset sums
    subset_sum = [0] * (1 << n)
    for mask in range(1 << n):
        for i in range(n):
            if mask & (1 << i):
                subset_sum[mask] += jobs[i]
    
    # dp[mask] = minimum maximum time to assign jobs in mask using any number of workers
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    
    # For each number of workers from 1 to k
    for workers_used in range(1, k + 1):
        new_dp = dp[:]
        
        for mask in range(1 << n):
            if dp[mask] == float('inf'):
                continue
            
            # Try all possible subsets to assign to a new worker
            submask = mask
            while submask > 0:
                complement = mask ^ submask
                time_for_new_worker = subset_sum[submask]
                new_max_time = max(dp[complement], time_for_new_worker)
                new_dp[mask] = min(new_dp[mask], new_max_time)
                
                submask = (submask - 1) & mask
        
        dp = new_dp
    
    return dp[(1 << n) - 1]


def minimum_time_to_finish_jobs_optimized_dp(jobs, k):
    """
    OPTIMIZED BITMASK DP:
    ====================
    Use subset enumeration DP for optimal job assignment.
    
    Time Complexity: O(3^n * k) - optimal for this problem
    Space Complexity: O(2^n) - DP table
    """
    n = len(jobs)
    
    # Precompute subset sums
    subset_sum = [0] * (1 << n)
    for mask in range(1 << n):
        for i in range(n):
            if mask & (1 << i):
                subset_sum[mask] += jobs[i]
    
    # dp[mask] = minimum time to complete jobs in mask with unlimited workers
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    
    # Compute single worker times for all subsets
    for mask in range(1 << n):
        dp[mask] = subset_sum[mask]
    
    # For each additional worker
    for worker in range(2, k + 1):
        new_dp = dp[:]
        
        for mask in range(1 << n):
            # Try all ways to split this mask between workers
            submask = mask
            while submask > 0:
                remaining = mask ^ submask
                new_dp[mask] = min(new_dp[mask], 
                                 max(subset_sum[submask], dp[remaining]))
                submask = (submask - 1) & mask
        
        dp = new_dp
    
    return dp[(1 << n) - 1]


def minimum_time_to_finish_jobs_binary_search(jobs, k):
    """
    BINARY SEARCH + BACKTRACKING:
    =============================
    Binary search on answer with backtracking verification.
    
    Time Complexity: O(log(sum) * k^n) - binary search * verification
    Space Complexity: O(n) - recursion stack
    """
    def can_finish_in_time(max_time):
        """Check if all jobs can be finished within max_time"""
        workers = [0] * k
        
        def backtrack(job_idx):
            if job_idx == len(jobs):
                return True
            
            # Try assigning current job to each worker
            for i in range(k):
                if workers[i] + jobs[job_idx] <= max_time:
                    workers[i] += jobs[job_idx]
                    if backtrack(job_idx + 1):
                        return True
                    workers[i] -= jobs[job_idx]
                
                # Optimization: if this worker is idle, don't try other idle workers
                if workers[i] == 0:
                    break
            
            return False
        
        return backtrack(0)
    
    # Binary search on the answer
    left = max(jobs)  # At least one job must be done by one worker
    right = sum(jobs)  # One worker does all jobs
    
    while left < right:
        mid = (left + right) // 2
        if can_finish_in_time(mid):
            right = mid
        else:
            left = mid + 1
    
    return left


def minimum_time_to_finish_jobs_with_analysis(jobs, k):
    """
    JOB SCHEDULING WITH DETAILED ANALYSIS:
    =====================================
    Solve job scheduling and provide detailed insights.
    
    Time Complexity: O(3^n * k) - standard DP
    Space Complexity: O(2^n) - DP table + analysis
    """
    n = len(jobs)
    
    analysis = {
        'num_jobs': n,
        'num_workers': k,
        'jobs': jobs[:],
        'total_work': sum(jobs),
        'avg_work_per_worker': sum(jobs) / k,
        'max_single_job': max(jobs),
        'min_single_job': min(jobs),
        'job_distribution': {},
        'optimal_assignment': None
    }
    
    # Job distribution analysis
    from collections import Counter
    job_freq = Counter(jobs)
    analysis['job_distribution'] = dict(job_freq)
    
    # Subset sum computation
    subset_sum = [0] * (1 << n)
    for mask in range(1 << n):
        for i in range(n):
            if mask & (1 << i):
                subset_sum[mask] += jobs[i]
    
    # DP with parent tracking
    dp = [float('inf')] * (1 << n)
    parent = [None] * (1 << n)
    dp[0] = 0
    
    for mask in range(1 << n):
        dp[mask] = subset_sum[mask]
        parent[mask] = [(mask, -1)]  # Single worker takes all jobs in mask
    
    for worker in range(2, k + 1):
        new_dp = dp[:]
        new_parent = parent[:]
        
        for mask in range(1 << n):
            submask = mask
            while submask > 0:
                remaining = mask ^ submask
                new_time = max(subset_sum[submask], dp[remaining])
                
                if new_time < new_dp[mask]:
                    new_dp[mask] = new_time
                    new_parent[mask] = [(submask, worker-1)] + parent[remaining]
                
                submask = (submask - 1) & mask
        
        dp = new_dp
        parent = new_parent
    
    optimal_time = dp[(1 << n) - 1]
    
    # Reconstruct assignment
    def reconstruct_assignment():
        assignment = [[] for _ in range(k)]
        
        if parent[(1 << n) - 1]:
            for subset_mask, worker_id in parent[(1 << n) - 1]:
                if worker_id >= 0:
                    for i in range(n):
                        if subset_mask & (1 << i):
                            assignment[worker_id].append(i)
        
        return assignment
    
    analysis['optimal_time'] = optimal_time
    analysis['optimal_assignment'] = reconstruct_assignment()
    analysis['efficiency'] = analysis['total_work'] / (optimal_time * k)
    analysis['load_balance'] = optimal_time / analysis['avg_work_per_worker']
    
    return optimal_time, analysis


def minimum_time_to_finish_jobs_analysis(jobs, k):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze the job scheduling problem with detailed insights.
    """
    print(f"Find Minimum Time to Finish All Jobs Analysis:")
    print(f"Jobs: {jobs}")
    print(f"Number of jobs: {len(jobs)}")
    print(f"Number of workers: {k}")
    print(f"Total work: {sum(jobs)}")
    print(f"Average work per worker: {sum(jobs) / k:.2f}")
    print(f"Maximum single job: {max(jobs)}")
    print(f"Theoretical minimum time: {max(max(jobs), sum(jobs) // k)}")
    
    # Different approaches
    if len(jobs) <= 8:
        try:
            backtrack = minimum_time_to_finish_jobs_backtrack(jobs, k)
            print(f"Backtracking result: {backtrack}")
        except:
            print("Backtracking: Too slow")
    
    bitmask_dp = minimum_time_to_finish_jobs_bitmask_dp(jobs, k)
    optimized = minimum_time_to_finish_jobs_optimized_dp(jobs, k)
    binary_search = minimum_time_to_finish_jobs_binary_search(jobs, k)
    
    print(f"Bitmask DP result: {bitmask_dp}")
    print(f"Optimized DP result: {optimized}")
    print(f"Binary search result: {binary_search}")
    
    # Detailed analysis
    detailed_time, analysis = minimum_time_to_finish_jobs_with_analysis(jobs, k)
    
    print(f"\nDetailed Analysis:")
    print(f"Optimal time: {detailed_time}")
    print(f"Work efficiency: {analysis['efficiency']:.2%}")
    print(f"Load balance ratio: {analysis['load_balance']:.2f}")
    
    print(f"\nJob Distribution:")
    for job_time, count in analysis['job_distribution'].items():
        print(f"  Jobs with time {job_time}: {count}")
    
    if analysis['optimal_assignment']:
        print(f"\nOptimal Assignment:")
        for i, worker_jobs in enumerate(analysis['optimal_assignment']):
            if worker_jobs:
                job_times = [jobs[j] for j in worker_jobs]
                total_time = sum(job_times)
                print(f"  Worker {i}: jobs {worker_jobs} -> times {job_times} (total: {total_time})")
    
    # Workload analysis
    print(f"\nWorkload Analysis:")
    if k == 1:
        print(f"Single worker must do all jobs: {sum(jobs)}")
    elif k >= len(jobs):
        print(f"Each job can have its own worker: {max(jobs)}")
    else:
        print(f"Need to balance {len(jobs)} jobs among {k} workers")
        print(f"Some workers will get multiple jobs")
    
    return optimized


def minimum_time_to_finish_jobs_variants():
    """
    JOB SCHEDULING VARIANTS:
    =======================
    Different scenarios and modifications.
    """
    
    def minimum_time_with_worker_capacities(jobs, worker_capacities):
        """Job scheduling with different worker capacities"""
        n = len(jobs)
        k = len(worker_capacities)
        
        # This is more complex - simplified version
        workers = worker_capacities[:]
        jobs_sorted = sorted(enumerate(jobs), key=lambda x: x[1], reverse=True)
        
        assignment = [[] for _ in range(k)]
        
        for job_idx, job_time in jobs_sorted:
            # Assign to worker with minimum current load that can handle the job
            best_worker = -1
            min_total_time = float('inf')
            
            for i in range(k):
                current_load = sum(jobs[j] for j in assignment[i])
                if current_load + job_time <= worker_capacities[i]:
                    if current_load + job_time < min_total_time:
                        min_total_time = current_load + job_time
                        best_worker = i
            
            if best_worker != -1:
                assignment[best_worker].append(job_idx)
        
        # Calculate maximum time
        max_time = 0
        for i in range(k):
            worker_time = sum(jobs[j] for j in assignment[i])
            max_time = max(max_time, worker_time)
        
        return max_time, assignment
    
    def minimum_time_with_precedence(jobs, k, precedence):
        """Job scheduling with precedence constraints"""
        # This is much more complex - simplified version
        # For now, just use basic scheduling and check if precedence is satisfied
        basic_time = minimum_time_to_finish_jobs_optimized_dp(jobs, k)
        
        # In a full implementation, we would need to modify the DP to respect precedence
        return basic_time
    
    def count_optimal_assignments(jobs, k):
        """Count number of optimal job assignments"""
        optimal_time = minimum_time_to_finish_jobs_optimized_dp(jobs, k)
        
        # Count assignments that achieve optimal time
        count = 0
        workers = [0] * k
        
        def count_backtrack(job_idx):
            nonlocal count
            
            if job_idx == len(jobs):
                if max(workers) == optimal_time:
                    count += 1
                return
            
            for i in range(k):
                if workers[i] + jobs[job_idx] <= optimal_time:
                    workers[i] += jobs[job_idx]
                    count_backtrack(job_idx + 1)
                    workers[i] -= jobs[job_idx]
                
                if workers[i] == 0:
                    break
        
        if len(jobs) <= 8:  # Only for small instances
            count_backtrack(0)
        else:
            count = -1  # Too complex
        
        return count
    
    def maximum_jobs_with_time_limit(jobs, k, time_limit):
        """Maximum number of jobs that can be completed within time limit"""
        n = len(jobs)
        max_jobs = 0
        
        # Try all subsets of jobs
        for mask in range(1 << n):
            selected_jobs = [jobs[i] for i in range(n) if mask & (1 << i)]
            
            if not selected_jobs:
                continue
            
            # Check if these jobs can be completed within time limit
            min_time = minimum_time_to_finish_jobs_optimized_dp(selected_jobs, k)
            
            if min_time <= time_limit:
                max_jobs = max(max_jobs, len(selected_jobs))
        
        return max_jobs
    
    # Test variants
    test_cases = [
        ([3, 2, 3], 3),
        ([1, 2, 4, 7, 8], 2),
        ([5, 5, 4, 4, 4], 3),
        ([1, 1, 1, 1], 2)
    ]
    
    print("Job Scheduling Variants:")
    print("=" * 50)
    
    for jobs, k in test_cases:
        print(f"\nJobs: {jobs}, Workers: {k}")
        
        basic_result = minimum_time_to_finish_jobs_optimized_dp(jobs, k)
        print(f"Basic minimum time: {basic_result}")
        
        # Worker capacities variant
        worker_capacities = [basic_result + 2] * k  # Give each worker some extra capacity
        capacity_time, capacity_assignment = minimum_time_with_worker_capacities(jobs, worker_capacities)
        print(f"With worker capacities {worker_capacities}: {capacity_time}")
        
        # Count optimal assignments
        if len(jobs) <= 6:
            count = count_optimal_assignments(jobs, k)
            print(f"Number of optimal assignments: {count}")
        
        # Maximum jobs with time limit
        time_limit = basic_result + 1
        max_jobs = maximum_jobs_with_time_limit(jobs, k, time_limit)
        print(f"Max jobs within time {time_limit}: {max_jobs}")


# Test cases
def test_minimum_time_to_finish_jobs():
    """Test all implementations with various inputs"""
    test_cases = [
        ([3, 2, 3], 3, 3),
        ([1, 2, 4, 7, 8], 2, 11),
        ([5, 5, 4, 4, 4], 3, 8),
        ([1, 1, 1, 1], 2, 2),
        ([10], 1, 10),
        ([1, 2], 2, 2),
        ([9, 8, 7, 6, 5, 4, 3, 2, 1], 3, 17)
    ]
    
    print("Testing Find Minimum Time to Finish All Jobs Solutions:")
    print("=" * 70)
    
    for i, (jobs, k, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"jobs = {jobs}, k = {k}")
        print(f"Expected: {expected}")
        
        # Skip backtracking for large inputs
        if len(jobs) <= 8:
            try:
                backtrack = minimum_time_to_finish_jobs_backtrack(jobs, k)
                print(f"Backtracking:     {backtrack:>4} {'✓' if backtrack == expected else '✗'}")
            except:
                print(f"Backtracking:     Timeout")
        
        bitmask_dp = minimum_time_to_finish_jobs_bitmask_dp(jobs, k)
        optimized = minimum_time_to_finish_jobs_optimized_dp(jobs, k)
        binary_search = minimum_time_to_finish_jobs_binary_search(jobs, k)
        
        print(f"Bitmask DP:       {bitmask_dp:>4} {'✓' if bitmask_dp == expected else '✗'}")
        print(f"Optimized:        {optimized:>4} {'✓' if optimized == expected else '✗'}")
        print(f"Binary Search:    {binary_search:>4} {'✓' if binary_search == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    minimum_time_to_finish_jobs_analysis([1, 2, 4, 7, 8], 2)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    minimum_time_to_finish_jobs_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. LOAD BALANCING: Distribute jobs to minimize maximum worker time")
    print("2. SUBSET ENUMERATION: Try all ways to partition jobs among workers")
    print("3. BINARY SEARCH: Search for minimum feasible maximum time")
    print("4. OPTIMIZATION: Use pruning and sorting for better performance")
    print("5. ASSIGNMENT TRACKING: Reconstruct optimal job assignments")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Task Scheduling: Distribute computational tasks among processors")
    print("• Load Balancing: Balance server loads in distributed systems")
    print("• Resource Allocation: Optimize resource distribution in systems")
    print("• Manufacturing: Schedule jobs on parallel machines")
    print("• Project Management: Assign tasks to team members optimally")


if __name__ == "__main__":
    test_minimum_time_to_finish_jobs()


"""
FIND MINIMUM TIME TO FINISH ALL JOBS - LOAD BALANCING OPTIMIZATION:
===================================================================

This problem demonstrates advanced job scheduling and load balancing:
- Optimal distribution of tasks among workers
- Minimizing maximum completion time (makespan scheduling)
- Complex state space exploration with pruning
- Real-world application in parallel processing systems

KEY INSIGHTS:
============
1. **LOAD BALANCING**: Distribute jobs to minimize maximum worker completion time
2. **MAKESPAN MINIMIZATION**: Classic scheduling objective in parallel systems
3. **SUBSET ENUMERATION**: Explore all ways to partition jobs among workers
4. **BINARY SEARCH OPTIMIZATION**: Search for minimum feasible maximum time
5. **ASSIGNMENT RECONSTRUCTION**: Track optimal job-to-worker assignments

ALGORITHM APPROACHES:
====================

1. **Backtracking**: O(k^n) time, O(n) space
   - Try all possible job assignments recursively
   - Heavy pruning for practical performance

2. **Bitmask DP**: O(3^n) time, O(2^n) space
   - Dynamic programming over job subsets
   - Optimal for exact solutions

3. **Optimized DP**: O(3^n * k) time, O(2^n) space
   - Improved state transitions and memory usage
   - Better constant factors

4. **Binary Search + Backtracking**: O(log(sum) * k^n) time, O(n) space
   - Binary search on answer with feasibility checking
   - Good for problems with large job times

CORE SUBSET ENUMERATION DP:
===========================
```python
def minimumTimeToFinishJobs(jobs, k):
    n = len(jobs)
    
    # Precompute subset sums
    subset_sum = [0] * (1 << n)
    for mask in range(1 << n):
        for i in range(n):
            if mask & (1 << i):
                subset_sum[mask] += jobs[i]
    
    # dp[mask] = min time to complete jobs in mask
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    
    # Initialize with single worker
    for mask in range(1 << n):
        dp[mask] = subset_sum[mask]
    
    # Add workers one by one
    for worker in range(2, k + 1):
        new_dp = dp[:]
        for mask in range(1 << n):
            # Try all ways to split jobs for new worker
            submask = mask
            while submask > 0:
                remaining = mask ^ submask
                new_dp[mask] = min(new_dp[mask], 
                                 max(subset_sum[submask], dp[remaining]))
                submask = (submask - 1) & mask
        dp = new_dp
    
    return dp[(1 << n) - 1]
```

SUBSET ENUMERATION TECHNIQUE:
============================
**Subset Iteration**: `submask = (submask - 1) & mask`
- Iterates through all subsets of a given mask
- Critical for exploring all job partitions

**State Transition**: For each job subset, consider:
- Assigning subset to one worker: `subset_sum[submask]`
- Optimal assignment of remaining jobs: `dp[remaining]`
- Combined time: `max(worker_time, remaining_time)`

LOAD BALANCING ANALYSIS:
=======================
**Objectives**:
- Minimize maximum completion time (makespan)
- Balance workload distribution among workers
- Achieve high resource utilization

**Constraints**:
- Each job assigned to exactly one worker
- Jobs cannot be split or preempted
- Workers process jobs sequentially

**Optimization Metrics**:
- Work efficiency = total_work / (optimal_time × workers)
- Load balance ratio = optimal_time / average_work_per_worker

BINARY SEARCH OPTIMIZATION:
===========================
**Search Space**: [max(jobs), sum(jobs)]
- Lower bound: largest single job time
- Upper bound: one worker does all jobs

**Feasibility Check**: Given maximum time T, can all jobs be completed?
```python
def can_finish_in_time(max_time):
    workers = [0] * k
    
    def backtrack(job_idx):
        if job_idx == len(jobs):
            return True
        
        for i in range(k):
            if workers[i] + jobs[job_idx] <= max_time:
                workers[i] += jobs[job_idx]
                if backtrack(job_idx + 1):
                    return True
                workers[i] -= jobs[job_idx]
            
            if workers[i] == 0:  # Avoid duplicate idle workers
                break
        
        return False
    
    return backtrack(0)
```

PRUNING TECHNIQUES:
==================
**Job Sorting**: Process larger jobs first for better pruning
**Idle Worker Elimination**: Don't try multiple idle workers
**Bound Checking**: Skip branches that exceed current best
**Symmetry Breaking**: Avoid equivalent worker assignments

COMPLEXITY ANALYSIS:
===================
**Bitmask DP**: O(3^n) time, O(2^n) space
- 3^n comes from subset enumeration pattern
- Each mask can be partitioned in multiple ways

**Binary Search**: O(log(sum) × k^n) time
- Logarithmic search × exponential verification
- Better for problems with large job values

**Practical Limits**: n ≤ 12-15 due to exponential nature

ASSIGNMENT RECONSTRUCTION:
=========================
**Parent Tracking**: Store optimal partitioning decisions
```python
parent[mask] = [(subset_assigned_to_worker, worker_id), ...]
```

**Path Recovery**: Backtrack through parent pointers
```python
def reconstruct_assignment():
    assignment = [[] for _ in range(k)]
    for subset_mask, worker_id in parent[full_mask]:
        for job_idx in subset_to_jobs(subset_mask):
            assignment[worker_id].append(job_idx)
    return assignment
```

APPLICATIONS:
============
- **Parallel Computing**: Task distribution among processors
- **Cloud Computing**: Load balancing in distributed systems  
- **Manufacturing**: Job scheduling on parallel machines
- **Project Management**: Task assignment optimization
- **Operations Research**: Makespan minimization problems

RELATED PROBLEMS:
================
- **Multiprocessor Scheduling**: Classic scheduling theory problem
- **Bin Packing**: Similar optimization with different constraints
- **Partition Problem**: Special case with two workers
- **Load Balancing**: General resource distribution problems

VARIANTS:
========
- **Worker Capacities**: Different maximum loads per worker
- **Precedence Constraints**: Job dependencies and ordering
- **Setup Times**: Additional costs for job transitions
- **Preemptive Scheduling**: Allow job interruption and resumption

EDGE CASES:
==========
- **k = 1**: Single worker does all jobs (sum of all jobs)
- **k ≥ n**: Each job can have dedicated worker (max single job)
- **Identical Jobs**: Uniform distribution among workers
- **One Large Job**: Dominates the solution

OPTIMIZATION TECHNIQUES:
=======================
**Preprocessing**: Sort jobs by size for better pruning
**State Compression**: Efficient subset representation
**Memoization**: Cache computed states to avoid recomputation  
**Approximation**: Heuristic methods for larger instances

This problem showcases practical optimization techniques
for parallel processing and resource allocation, demonstrating
how exact algorithms can solve moderately-sized instances
while highlighting the need for approximation methods
in larger real-world systems.
"""
