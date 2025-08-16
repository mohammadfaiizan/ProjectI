"""
LeetCode 1434: Number of Ways to Wear Different Hats to Each Other
Difficulty: Hard
Category: Bitmask DP - Assignment Counting

PROBLEM DESCRIPTION:
===================
There are n people and 40 types of hats labeled from 1 to 40.

Given a list of list of integers hats, where hats[i] is a list of all hats preferred by the ith person.

Return the number of ways to assign hats to people such that no two people wear the same hat.

Each person must wear exactly one hat and each hat can be worn by at most one person.

Example 1:
Input: hats = [[3,4],[4,5],[5]]
Output: 1
Explanation: There is only one way to assign hats to three people:
First person can only choose hat 3 or 4.
Second person can only choose hat 4 or 5.
Third person can only choose hat 5.
Since the second person cannot choose hat 5 (it would be taken by the third person), the second person must choose hat 4.
First person chooses hat 3. Each person wears a different hat and we have 1 way.

Example 2:
Input: hats = [[3,5,1],[3,5],[1,2,3,4]]
Output: 4
Explanation: There are 4 ways to assign hats to three people:
way #1: first person chooses hat 3, second person chooses hat 5, third person chooses hat 1.
way #2: first person chooses hat 5, second person chooses hat 3, third person chooses hat 1.
way #3: first person chooses hat 1, second person chooses hat 3, third person chooses hat 4.
way #4: first person chooses hat 1, second person chooses hat 5, third person chooses hat 4.

Constraints:
- n == hats.length
- 1 <= n <= 10
- 1 <= hats[i].length <= 40
- 1 <= hats[i][j] <= 40
- hats[i] contains a list of unique integers.
"""


def number_of_ways_to_wear_hats_backtrack(hats):
    """
    BACKTRACKING APPROACH:
    =====================
    Try all possible hat assignments using backtracking.
    
    Time Complexity: O(n! * h^n) where h is number of distinct hats
    Space Complexity: O(n) - recursion stack
    """
    MOD = 10**9 + 7
    n = len(hats)
    
    # Get all unique hats
    all_hats = set()
    for person_hats in hats:
        all_hats.update(person_hats)
    all_hats = sorted(all_hats)
    
    # Track which hats are used
    used_hats = set()
    count = [0]
    
    def backtrack(person_idx):
        if person_idx == n:
            count[0] = (count[0] + 1) % MOD
            return
        
        # Try each hat that this person likes
        for hat in hats[person_idx]:
            if hat not in used_hats:
                used_hats.add(hat)
                backtrack(person_idx + 1)
                used_hats.remove(hat)
    
    backtrack(0)
    return count[0]


def number_of_ways_to_wear_hats_person_bitmask(hats):
    """
    PERSON-BASED BITMASK DP:
    =======================
    Use bitmask to track which people have been assigned hats.
    
    Time Complexity: O(h * 2^n * n) where h is number of hats
    Space Complexity: O(2^n) - DP table
    """
    MOD = 10**9 + 7
    n = len(hats)
    
    # Create hat-to-people mapping
    hat_to_people = {}
    for person, person_hats in enumerate(hats):
        for hat in person_hats:
            if hat not in hat_to_people:
                hat_to_people[hat] = []
            hat_to_people[hat].append(person)
    
    all_hats = sorted(hat_to_people.keys())
    
    # dp[mask] = number of ways to assign hats to people in mask
    dp = [0] * (1 << n)
    dp[0] = 1  # Base case: no people assigned yet
    
    # Process each hat
    for hat in all_hats:
        new_dp = dp[:]
        
        # For each current state
        for mask in range(1 << n):
            if dp[mask] == 0:
                continue
            
            # Try assigning this hat to each person who likes it
            for person in hat_to_people[hat]:
                if not (mask & (1 << person)):  # Person not yet assigned
                    new_mask = mask | (1 << person)
                    new_dp[new_mask] = (new_dp[new_mask] + dp[mask]) % MOD
        
        dp = new_dp
    
    return dp[(1 << n) - 1]


def number_of_ways_to_wear_hats_hat_bitmask(hats):
    """
    HAT-BASED BITMASK DP:
    ====================
    Use bitmask to track which hats have been used.
    
    Time Complexity: O(n * 2^h) where h is number of distinct hats
    Space Complexity: O(2^h) - DP table
    """
    MOD = 10**9 + 7
    n = len(hats)
    
    # Get all unique hats and create mapping
    all_hats = set()
    for person_hats in hats:
        all_hats.update(person_hats)
    all_hats = sorted(all_hats)
    hat_to_id = {hat: i for i, hat in enumerate(all_hats)}
    
    # Convert to hat IDs
    person_hat_masks = []
    for person_hats in hats:
        mask = 0
        for hat in person_hats:
            mask |= (1 << hat_to_id[hat])
        person_hat_masks.append(mask)
    
    num_hats = len(all_hats)
    memo = {}
    
    def dp(person_idx, used_hat_mask):
        if person_idx == n:
            return 1
        
        if (person_idx, used_hat_mask) in memo:
            return memo[(person_idx, used_hat_mask)]
        
        result = 0
        person_mask = person_hat_masks[person_idx]
        
        # Try each hat this person likes
        for hat_id in range(num_hats):
            if (person_mask & (1 << hat_id)) and not (used_hat_mask & (1 << hat_id)):
                new_used_mask = used_hat_mask | (1 << hat_id)
                result = (result + dp(person_idx + 1, new_used_mask)) % MOD
        
        memo[(person_idx, used_hat_mask)] = result
        return result
    
    return dp(0, 0)


def number_of_ways_to_wear_hats_optimized(hats):
    """
    OPTIMIZED BITMASK DP:
    ====================
    Use the more efficient approach based on problem constraints.
    
    Time Complexity: O(h * 2^n * n) - better for small n
    Space Complexity: O(2^n) - DP table
    """
    MOD = 10**9 + 7
    n = len(hats)
    
    # Build hat-to-people mapping
    hat_to_people = {}
    for person, person_hats in enumerate(hats):
        for hat in person_hats:
            if hat not in hat_to_people:
                hat_to_people[hat] = []
            hat_to_people[hat].append(person)
    
    # Sort hats for consistent processing
    all_hats = sorted(hat_to_people.keys())
    
    # dp[mask] = ways to assign hats to people represented by mask
    dp = [0] * (1 << n)
    dp[0] = 1
    
    # Process hats one by one
    for hat in all_hats:
        # Process in reverse order to avoid using updated values
        for mask in range((1 << n) - 1, -1, -1):
            if dp[mask] == 0:
                continue
            
            # Try giving this hat to each person who likes it
            for person in hat_to_people[hat]:
                if not (mask & (1 << person)):
                    new_mask = mask | (1 << person)
                    dp[new_mask] = (dp[new_mask] + dp[mask]) % MOD
    
    return dp[(1 << n) - 1]


def number_of_ways_to_wear_hats_with_analysis(hats):
    """
    HAT ASSIGNMENT WITH DETAILED ANALYSIS:
    =====================================
    Count ways and provide detailed analysis of the assignment problem.
    
    Time Complexity: O(h * 2^n * n) - standard approach
    Space Complexity: O(2^n) - DP table + analysis
    """
    MOD = 10**9 + 7
    n = len(hats)
    
    # Analyze the problem structure
    all_hats = set()
    for person_hats in hats:
        all_hats.update(person_hats)
    all_hats = sorted(all_hats)
    
    analysis = {
        'num_people': n,
        'num_unique_hats': len(all_hats),
        'hat_range': (min(all_hats), max(all_hats)),
        'person_preferences': {},
        'hat_popularity': {},
        'critical_hats': [],
        'flexible_people': [],
        'state_space_size': 2 ** n
    }
    
    # Analyze preferences
    for person, person_hats in enumerate(hats):
        analysis['person_preferences'][person] = {
            'likes': person_hats[:],
            'count': len(person_hats),
            'flexibility': len(person_hats) / len(all_hats)
        }
    
    # Build hat-to-people mapping and analyze popularity
    hat_to_people = {}
    for person, person_hats in enumerate(hats):
        for hat in person_hats:
            if hat not in hat_to_people:
                hat_to_people[hat] = []
            hat_to_people[hat].append(person)
    
    for hat in all_hats:
        popularity = len(hat_to_people.get(hat, []))
        analysis['hat_popularity'][hat] = popularity
        
        if popularity == 1:
            analysis['critical_hats'].append(hat)
    
    # Find flexible people (many hat choices)
    avg_choices = sum(len(person_hats) for person_hats in hats) / n
    for person, person_hats in enumerate(hats):
        if len(person_hats) > avg_choices:
            analysis['flexible_people'].append(person)
    
    # Run DP with state tracking
    dp = [0] * (1 << n)
    dp[0] = 1
    states_computed = 0
    
    for hat in all_hats:
        new_dp = dp[:]
        
        for mask in range(1 << n):
            if dp[mask] == 0:
                continue
            
            states_computed += 1
            
            for person in hat_to_people.get(hat, []):
                if not (mask & (1 << person)):
                    new_mask = mask | (1 << person)
                    new_dp[new_mask] = (new_dp[new_mask] + dp[mask]) % MOD
        
        dp = new_dp
    
    result = dp[(1 << n) - 1]
    
    analysis['result'] = result
    analysis['states_computed'] = states_computed
    analysis['computation_efficiency'] = states_computed / (len(all_hats) * (2 ** n))
    
    return result, analysis


def number_of_ways_to_wear_hats_analysis(hats):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze the hat assignment problem with detailed insights.
    """
    print(f"Number of Ways to Wear Different Hats Analysis:")
    print(f"People and their hat preferences:")
    
    for i, person_hats in enumerate(hats):
        print(f"  Person {i}: {person_hats}")
    
    n = len(hats)
    all_hats = set()
    for person_hats in hats:
        all_hats.update(person_hats)
    all_hats = sorted(all_hats)
    
    print(f"Number of people: {n}")
    print(f"Number of unique hats: {len(all_hats)}")
    print(f"Hat range: {min(all_hats)} to {max(all_hats)}")
    print(f"Total possible assignments: {len(all_hats)}^{n}")
    print(f"State space size: 2^{n} = {2**n}")
    
    # Different approaches
    if n <= 6:
        try:
            backtrack = number_of_ways_to_wear_hats_backtrack(hats)
            print(f"Backtracking result: {backtrack}")
        except:
            print("Backtracking: Too slow")
    
    person_bitmask = number_of_ways_to_wear_hats_person_bitmask(hats)
    hat_bitmask = number_of_ways_to_wear_hats_hat_bitmask(hats)
    optimized = number_of_ways_to_wear_hats_optimized(hats)
    
    print(f"Person bitmask result: {person_bitmask}")
    print(f"Hat bitmask result: {hat_bitmask}")
    print(f"Optimized result: {optimized}")
    
    # Detailed analysis
    detailed_result, analysis = number_of_ways_to_wear_hats_with_analysis(hats)
    
    print(f"\nDetailed Analysis:")
    print(f"Number of valid assignments: {detailed_result}")
    print(f"States computed: {analysis['states_computed']:,}")
    print(f"Computation efficiency: {analysis['computation_efficiency']:.2%}")
    
    print(f"\nPerson Preferences:")
    for person, info in analysis['person_preferences'].items():
        print(f"  Person {person}: {info['count']} choices, flexibility {info['flexibility']:.2%}")
    
    print(f"\nHat Popularity:")
    for hat, popularity in sorted(analysis['hat_popularity'].items()):
        print(f"  Hat {hat}: liked by {popularity} people")
    
    if analysis['critical_hats']:
        print(f"\nCritical hats (liked by only 1 person): {analysis['critical_hats']}")
    
    if analysis['flexible_people']:
        print(f"Flexible people (above average choices): {analysis['flexible_people']}")
    
    # Constraint analysis
    print(f"\nConstraint Analysis:")
    impossible = False
    for person, person_hats in enumerate(hats):
        if not person_hats:
            print(f"  Person {person} has no hat preferences - impossible!")
            impossible = True
    
    if not impossible:
        print(f"  All people have at least one hat preference")
        
        # Check if assignment is possible (matching theory)
        hat_to_people = {}
        for person, person_hats in enumerate(hats):
            for hat in person_hats:
                if hat not in hat_to_people:
                    hat_to_people[hat] = []
                hat_to_people[hat].append(person)
        
        print(f"  {len(hat_to_people)} distinct hats available for {n} people")
        
        if len(hat_to_people) < n:
            print(f"  Not enough hats for all people - some assignments impossible")
    
    return optimized


def number_of_ways_to_wear_hats_variants():
    """
    HAT ASSIGNMENT VARIANTS:
    =======================
    Different scenarios and modifications.
    """
    
    def ways_with_required_hats(hats, required_hats):
        """Count ways where specific hats must be used"""
        MOD = 10**9 + 7
        n = len(hats)
        
        # Filter preferences to only include required hats
        filtered_hats = []
        for person_hats in hats:
            filtered = [hat for hat in person_hats if hat in required_hats]
            filtered_hats.append(filtered)
        
        return number_of_ways_to_wear_hats_optimized(filtered_hats)
    
    def ways_with_forbidden_hats(hats, forbidden_hats):
        """Count ways where specific hats cannot be used"""
        # Filter out forbidden hats
        filtered_hats = []
        for person_hats in hats:
            filtered = [hat for hat in person_hats if hat not in forbidden_hats]
            filtered_hats.append(filtered)
        
        return number_of_ways_to_wear_hats_optimized(filtered_hats)
    
    def ways_with_paired_people(hats, pairs):
        """Count ways where certain people must wear similar hats"""
        # This is much more complex - simplified version
        # For now, just return basic count
        return number_of_ways_to_wear_hats_optimized(hats)
    
    def maximum_people_with_hats(hats, available_hats):
        """Maximum number of people that can get hats from available set"""
        n = len(hats)
        
        # Filter preferences
        filtered_hats = []
        for person_hats in hats:
            filtered = [hat for hat in person_hats if hat in available_hats]
            filtered_hats.append(filtered)
        
        # Try to find maximum matching
        max_people = 0
        
        # Use bitmask to try all subsets of people
        for mask in range(1 << n):
            people_subset = [i for i in range(n) if mask & (1 << i)]
            
            if not people_subset:
                continue
            
            # Check if this subset can be satisfied
            subset_hats = [filtered_hats[i] for i in people_subset]
            ways = number_of_ways_to_wear_hats_optimized(subset_hats)
            
            if ways > 0:
                max_people = max(max_people, len(people_subset))
        
        return max_people
    
    # Test variants
    test_cases = [
        [[3, 4], [4, 5], [5]],
        [[3, 5, 1], [3, 5], [1, 2, 3, 4]],
        [[1, 2], [2, 3], [3, 4]],
        [[1], [2], [3]]
    ]
    
    print("Hat Assignment Variants:")
    print("=" * 50)
    
    for hats in test_cases:
        print(f"\nHat preferences: {hats}")
        
        basic_ways = number_of_ways_to_wear_hats_optimized(hats)
        print(f"Basic assignment ways: {basic_ways}")
        
        # Required hats variant
        all_hats = set()
        for person_hats in hats:
            all_hats.update(person_hats)
        
        if len(all_hats) > 2:
            required = list(all_hats)[:2]
            required_ways = ways_with_required_hats(hats, required)
            print(f"Ways with required hats {required}: {required_ways}")
        
        # Forbidden hats variant
        if len(all_hats) > 1:
            forbidden = [list(all_hats)[0]]
            forbidden_ways = ways_with_forbidden_hats(hats, forbidden)
            print(f"Ways without forbidden hats {forbidden}: {forbidden_ways}")
        
        # Maximum people variant
        available = list(all_hats)[:-1] if len(all_hats) > 1 else list(all_hats)
        max_people = maximum_people_with_hats(hats, available)
        print(f"Max people with hats {available}: {max_people}")


# Test cases
def test_number_of_ways_to_wear_hats():
    """Test all implementations with various inputs"""
    test_cases = [
        ([[3,4],[4,5],[5]], 1),
        ([[3,5,1],[3,5],[1,2,3,4]], 4),
        ([[1,2],[2,3],[3,4]], 1),
        ([[1]], 1),
        ([[1,2],[1,2]], 2),
        ([[1,2,3],[2,3,4],[1,4]], 8)
    ]
    
    print("Testing Number of Ways to Wear Different Hats Solutions:")
    print("=" * 70)
    
    for i, (hats, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"hats = {hats}")
        print(f"Expected: {expected}")
        
        # Skip backtracking for large inputs
        if len(hats) <= 5:
            try:
                backtrack = number_of_ways_to_wear_hats_backtrack(hats)
                print(f"Backtracking:     {backtrack:>6} {'✓' if backtrack == expected else '✗'}")
            except:
                print(f"Backtracking:     Timeout")
        
        person_bitmask = number_of_ways_to_wear_hats_person_bitmask(hats)
        hat_bitmask = number_of_ways_to_wear_hats_hat_bitmask(hats)
        optimized = number_of_ways_to_wear_hats_optimized(hats)
        
        print(f"Person Bitmask:   {person_bitmask:>6} {'✓' if person_bitmask == expected else '✗'}")
        print(f"Hat Bitmask:      {hat_bitmask:>6} {'✓' if hat_bitmask == expected else '✗'}")
        print(f"Optimized:        {optimized:>6} {'✓' if optimized == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    number_of_ways_to_wear_hats_analysis([[3,5,1],[3,5],[1,2,3,4]])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    number_of_ways_to_wear_hats_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. ASSIGNMENT COUNTING: Count valid bijective mappings")
    print("2. BITMASK CHOICE: Choose based on smaller dimension (people vs hats)")
    print("3. STATE TRANSITIONS: Process hats sequentially, update people states")
    print("4. CONSTRAINT SATISFACTION: Ensure unique assignments")
    print("5. PREFERENCE ANALYSIS: Identify critical constraints and flexibility")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Resource Assignment: Assign unique resources to requesters")
    print("• Task Allocation: Distribute distinct tasks among workers")
    print("• Matching Problems: Count perfect matchings in bipartite graphs")
    print("• Combinatorics: Permutations with restrictions")
    print("• Game Theory: Count valid strategy combinations")


if __name__ == "__main__":
    test_number_of_ways_to_wear_hats()


"""
NUMBER OF WAYS TO WEAR DIFFERENT HATS - ASSIGNMENT COUNTING WITH CONSTRAINTS:
=============================================================================

This problem demonstrates advanced assignment counting with preferences:
- Counting valid bijective mappings between people and hats
- Constraint satisfaction with preference restrictions
- Bipartite matching with counting instead of optimization
- Choice of state representation based on problem dimensions

KEY INSIGHTS:
============
1. **ASSIGNMENT COUNTING**: Count valid one-to-one mappings between people and hats
2. **BITMASK CHOICE**: Choose state representation based on smaller dimension
3. **CONSTRAINT SATISFACTION**: Ensure each person gets exactly one hat, each hat assigned at most once
4. **PREFERENCE MODELING**: Handle individual preferences as hard constraints
5. **STATE TRANSITION OPTIMIZATION**: Process in order that minimizes state space

ALGORITHM APPROACHES:
====================

1. **Backtracking**: O(n! × h^n) time, O(n) space
   - Try all possible assignments recursively
   - Exponential but can work for small instances

2. **Person-Based Bitmask**: O(h × 2^n × n) time, O(2^n) space
   - Track which people have been assigned hats
   - Better when n < h (fewer people than hats)

3. **Hat-Based Bitmask**: O(n × 2^h) time, O(2^h) space  
   - Track which hats have been used
   - Better when h < n (fewer hats than people)

4. **Optimized Approach**: O(h × 2^n × n) time, O(2^n) space
   - Choose optimal representation based on constraints
   - Most practical for given problem limits

CORE PERSON-BASED BITMASK DP:
============================
```python
def numberOfWays(hats):
    MOD = 10**9 + 7
    n = len(hats)
    
    # Build hat-to-people mapping
    hat_to_people = {}
    for person, person_hats in enumerate(hats):
        for hat in person_hats:
            if hat not in hat_to_people:
                hat_to_people[hat] = []
            hat_to_people[hat].append(person)
    
    # dp[mask] = ways to assign hats to people in mask
    dp = [0] * (1 << n)
    dp[0] = 1
    
    # Process each hat sequentially
    for hat in sorted(hat_to_people.keys()):
        new_dp = dp[:]
        
        for mask in range(1 << n):
            if dp[mask] == 0:
                continue
            
            # Try assigning this hat to each person who likes it
            for person in hat_to_people[hat]:
                if not (mask & (1 << person)):  # Person not assigned yet
                    new_mask = mask | (1 << person)
                    new_dp[new_mask] = (new_dp[new_mask] + dp[mask]) % MOD
        
        dp = new_dp
    
    return dp[(1 << n) - 1]
```

BITMASK REPRESENTATION CHOICE:
=============================
**People Bitmask**: Track which people have hats
- State: `dp[people_mask]` = ways to assign hats to people in mask
- Transitions: For each hat, try assigning to available people
- Complexity: O(hats × 2^people × people)

**Hat Bitmask**: Track which hats are used
- State: `dp[person][hat_mask]` = ways to assign first person people using hats in mask
- Transitions: For each person, try each available hat
- Complexity: O(people × 2^hats)

**Optimal Choice**: Choose representation with smaller exponential factor

CONSTRAINT SATISFACTION:
=======================
**Hard Constraints**:
- Each person must get exactly one hat
- Each hat can be assigned to at most one person
- People can only get hats from their preference list

**Soft Preferences**: Individual preference lists restrict valid assignments

**Feasibility**: Assignment possible only if perfect matching exists in preference graph

STATE TRANSITION DESIGN:
========================
**Sequential Processing**: Process hats (or people) one at a time
```python
for hat in all_hats:
    # Update all states by trying to assign this hat
    for current_state in all_states:
        for person in people_who_like_hat:
            if person_available(current_state, person):
                update_state(current_state, person, hat)
```

**Update Strategy**: Create new DP array to avoid using updated values
**Modular Arithmetic**: Apply MOD at each step to prevent overflow

PREFERENCE ANALYSIS:
===================
**Critical Hats**: Hats liked by only one person (must be assigned to that person)
**Flexible People**: People with many hat options (easier to satisfy)
**Bottlenecks**: People with few options (constrain solution space)

**Popularity Metrics**:
```python
hat_popularity = {hat: len(people_who_like_it) for hat in all_hats}
person_flexibility = {person: len(liked_hats) for person, liked_hats in enumerate(hats)}
```

COMPLEXITY ANALYSIS:
===================
**Person-Based**: O(h × 2^n × n) 
- h hats to process
- 2^n possible people assignments
- n people to try per hat

**Hat-Based**: O(n × 2^h)
- n people to assign
- 2^h possible hat usage patterns

**Space**: O(2^min(n,h)) - exponential in smaller dimension

**Practical Limits**: min(n,h) ≤ 15-20 for reasonable performance

BIPARTITE MATCHING CONNECTION:
=============================
**Graph Model**: People on left, hats on right, edges for preferences
**Perfect Matching**: Assignment where everyone gets a hat
**Counting vs Optimization**: Count all perfect matchings instead of finding one

**Hall's Theorem**: Perfect matching exists iff for every subset of people, their collective preferences cover at least as many hats

ASSIGNMENT RECONSTRUCTION:
=========================
**Solution Enumeration**: For small cases, can enumerate all valid assignments
```python
def enumerate_assignments(hats):
    assignments = []
    # Use backtracking to generate all valid assignments
    # Store each complete assignment
    return assignments
```

**Random Sampling**: For large solution spaces, sample random valid assignments

APPLICATIONS:
============
- **Resource Allocation**: Assign unique resources based on preferences
- **Task Assignment**: Distribute distinct tasks among workers with skills
- **Matching Markets**: Count stable matchings in two-sided markets
- **Combinatorics**: Permutations and combinations with constraints
- **Game Theory**: Count valid strategy profiles

RELATED PROBLEMS:
================
- **Perfect Matching**: Find any valid assignment
- **Maximum Bipartite Matching**: Maximize assignments when perfect matching impossible
- **Assignment Problem**: Optimize assignment costs
- **Stable Marriage**: Consider mutual preferences

VARIANTS:
========
- **Required Assignments**: Force certain people to get specific hats
- **Forbidden Assignments**: Prevent certain person-hat combinations
- **Grouped Constraints**: People in same group must wear similar hats
- **Weighted Counting**: Count assignments with preference-based weights

EDGE CASES:
==========
- **No Valid Assignment**: When perfect matching impossible
- **Unique Assignment**: Only one way to satisfy all constraints
- **Empty Preferences**: Person likes no hats (impossible case)
- **Universal Preferences**: Person likes all hats (maximum flexibility)

OPTIMIZATION TECHNIQUES:
=======================
**Preprocessing**: Remove impossible assignments, identify forced choices
**Symmetry Breaking**: Avoid counting equivalent assignments multiple times
**Constraint Propagation**: Use critical assignments to simplify problem
**Early Termination**: Stop when assignment becomes impossible

This problem showcases the power of choosing appropriate state
representations in Bitmask DP, demonstrating how the same
problem can be solved with different exponential complexities
based on which dimension is encoded in the bitmask.
"""
