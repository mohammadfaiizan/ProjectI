"""
LeetCode 1125: Smallest Sufficient Team
Difficulty: Hard
Category: Bitmask DP - Team Formation Optimization

PROBLEM DESCRIPTION:
===================
In a project, you have a list of required skills req_skills, and a list of people. The ith person people[i] contains a list of skills that the person has.

Consider a sufficient team: a set of people such that for every required skill, there is at least one person in the team who has that skill. We can represent these skills by an index, and by people[i][j] being the index of the jth skill of the ith person.

Return any sufficient team of the smallest size, represented by the index of each person. You may return the answer in any order.

Example 1:
Input: req_skills = ["java","nodejs","reactjs"], people = [["java"],["nodejs"],["nodejs","reactjs"]]
Output: [0,2]
Explanation: Person 0 has skill "java", person 2 has skills "nodejs" and "reactjs".

Example 2:
Input: req_skills = ["algorithms","math","java","reactjs","csharp","aws"], people = [["algorithms","math","java"],["algorithms","math","reactjs"],["java","csharp","aws"],["reactjs","csharp"],["csharp","math"],["aws","java"]]
Output: [1,2]

Constraints:
- 1 <= req_skills.length <= 16
- 1 <= people.length <= 60
- 1 <= people[i].length, req_skills[i].length, people[i][j].length <= 16
- Elements of req_skills are unique.
- req_skills[i] and people[i][j] are composed of lowercase English letters.
- Every required skill is covered by at least one person.
"""


def smallest_sufficient_team_brute_force(req_skills, people):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible team combinations to find the smallest one.
    
    Time Complexity: O(2^n * m * k) where n=people, m=skills, k=avg skills per person
    Space Complexity: O(n) - recursion stack
    """
    skill_to_id = {skill: i for i, skill in enumerate(req_skills)}
    num_skills = len(req_skills)
    target_mask = (1 << num_skills) - 1
    
    # Convert people skills to bitmasks
    people_masks = []
    for person_skills in people:
        mask = 0
        for skill in person_skills:
            if skill in skill_to_id:
                mask |= (1 << skill_to_id[skill])
        people_masks.append(mask)
    
    min_team_size = float('inf')
    best_team = []
    
    def backtrack(idx, current_mask, current_team):
        nonlocal min_team_size, best_team
        
        if current_mask == target_mask:
            if len(current_team) < min_team_size:
                min_team_size = len(current_team)
                best_team = current_team[:]
            return
        
        if idx >= len(people) or len(current_team) >= min_team_size:
            return
        
        # Skip current person
        backtrack(idx + 1, current_mask, current_team)
        
        # Include current person
        new_mask = current_mask | people_masks[idx]
        current_team.append(idx)
        backtrack(idx + 1, new_mask, current_team)
        current_team.pop()
    
    backtrack(0, 0, [])
    return best_team


def smallest_sufficient_team_bitmask_dp(req_skills, people):
    """
    BITMASK DP APPROACH:
    ===================
    Use bitmask DP to find optimal team formation.
    
    Time Complexity: O(n * 2^m) where n=people, m=skills
    Space Complexity: O(2^m) - DP table
    """
    skill_to_id = {skill: i for i, skill in enumerate(req_skills)}
    num_skills = len(req_skills)
    
    # Convert people skills to bitmasks
    people_masks = []
    for person_skills in people:
        mask = 0
        for skill in person_skills:
            if skill in skill_to_id:
                mask |= (1 << skill_to_id[skill])
        people_masks.append(mask)
    
    # dp[mask] = minimum team size to achieve skill set mask
    dp = [float('inf')] * (1 << num_skills)
    parent = [-1] * (1 << num_skills)
    dp[0] = 0
    
    for mask in range(1 << num_skills):
        if dp[mask] == float('inf'):
            continue
        
        for i, person_mask in enumerate(people_masks):
            new_mask = mask | person_mask
            if dp[mask] + 1 < dp[new_mask]:
                dp[new_mask] = dp[mask] + 1
                parent[new_mask] = (mask, i)
    
    # Reconstruct solution
    target_mask = (1 << num_skills) - 1
    team = []
    current_mask = target_mask
    
    while current_mask != 0:
        if parent[current_mask] == -1:
            break
        prev_mask, person_idx = parent[current_mask]
        team.append(person_idx)
        current_mask = prev_mask
    
    return team


def smallest_sufficient_team_optimized_dp(req_skills, people):
    """
    OPTIMIZED BITMASK DP:
    ====================
    Use optimized DP with better team tracking.
    
    Time Complexity: O(n * 2^m) - optimal for this problem
    Space Complexity: O(2^m) - DP table
    """
    skill_to_id = {skill: i for i, skill in enumerate(req_skills)}
    num_skills = len(req_skills)
    
    # Convert people to skill masks
    people_masks = []
    for person_skills in people:
        mask = 0
        for skill in person_skills:
            if skill in skill_to_id:
                mask |= (1 << skill_to_id[skill])
        people_masks.append(mask)
    
    # dp[mask] = list of people indices forming minimum team for mask
    dp = {}
    dp[0] = []
    
    for mask in range(1 << num_skills):
        if mask not in dp:
            continue
        
        for i, person_mask in enumerate(people_masks):
            new_mask = mask | person_mask
            new_team = dp[mask] + [i]
            
            if new_mask not in dp or len(new_team) < len(dp[new_mask]):
                dp[new_mask] = new_team
    
    target_mask = (1 << num_skills) - 1
    return dp.get(target_mask, [])


def smallest_sufficient_team_with_analysis(req_skills, people):
    """
    BITMASK DP WITH DETAILED ANALYSIS:
    =================================
    Track the computation process with team formation insights.
    
    Time Complexity: O(n * 2^m) - standard DP
    Space Complexity: O(2^m) - DP table + analysis
    """
    skill_to_id = {skill: i for i, skill in enumerate(req_skills)}
    num_skills = len(req_skills)
    
    # Analyze people skills
    people_masks = []
    skill_coverage = [[] for _ in range(num_skills)]
    
    for i, person_skills in enumerate(people):
        mask = 0
        for skill in person_skills:
            if skill in skill_to_id:
                skill_id = skill_to_id[skill]
                mask |= (1 << skill_id)
                skill_coverage[skill_id].append(i)
        people_masks.append(mask)
    
    analysis = {
        'num_skills': num_skills,
        'num_people': len(people),
        'skill_coverage': skill_coverage,
        'people_masks': people_masks,
        'rare_skills': [],
        'versatile_people': [],
        'total_states': 2 ** num_skills
    }
    
    # Find rare skills (covered by few people)
    for skill_id in range(num_skills):
        if len(skill_coverage[skill_id]) <= 2:
            analysis['rare_skills'].append((req_skills[skill_id], len(skill_coverage[skill_id])))
    
    # Find versatile people (many skills)
    for i, mask in enumerate(people_masks):
        skill_count = bin(mask).count('1')
        if skill_count >= num_skills // 2:
            analysis['versatile_people'].append((i, skill_count))
    
    # Run DP
    dp = {}
    dp[0] = []
    
    for mask in range(1 << num_skills):
        if mask not in dp:
            continue
        
        for i, person_mask in enumerate(people_masks):
            new_mask = mask | person_mask
            new_team = dp[mask] + [i]
            
            if new_mask not in dp or len(new_team) < len(dp[new_mask]):
                dp[new_mask] = new_team
    
    target_mask = (1 << num_skills) - 1
    optimal_team = dp.get(target_mask, [])
    
    analysis['optimal_team'] = optimal_team
    analysis['optimal_size'] = len(optimal_team)
    analysis['states_computed'] = len(dp)
    
    return optimal_team, analysis


def smallest_sufficient_team_analysis(req_skills, people):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze the team formation problem with detailed insights.
    """
    print(f"Smallest Sufficient Team Analysis:")
    print(f"Required skills: {req_skills}")
    print(f"Number of required skills: {len(req_skills)}")
    print(f"Number of people: {len(people)}")
    
    # People analysis
    for i, person_skills in enumerate(people):
        print(f"Person {i}: {person_skills}")
    
    print(f"Total possible teams: 2^{len(people)} = {2**len(people):,}")
    print(f"Skill state space: 2^{len(req_skills)} = {2**len(req_skills):,}")
    
    # Different approaches
    if len(people) <= 15:
        try:
            brute_force = smallest_sufficient_team_brute_force(req_skills, people)
            print(f"Brute force result: {brute_force}")
        except:
            print("Brute force: Too slow")
    
    bitmask_dp = smallest_sufficient_team_bitmask_dp(req_skills, people)
    optimized = smallest_sufficient_team_optimized_dp(req_skills, people)
    
    print(f"Bitmask DP result: {bitmask_dp}")
    print(f"Optimized DP result: {optimized}")
    
    # Detailed analysis
    team_with_analysis, analysis = smallest_sufficient_team_with_analysis(req_skills, people)
    
    print(f"\nDetailed Analysis:")
    print(f"Optimal team: {team_with_analysis}")
    print(f"Optimal team size: {analysis['optimal_size']}")
    print(f"States computed: {analysis['states_computed']}/{analysis['total_states']}")
    
    print(f"\nSkill Coverage Analysis:")
    for i, (skill, people_list) in enumerate(zip(req_skills, analysis['skill_coverage'])):
        print(f"  {skill}: covered by people {people_list} ({len(people_list)} people)")
    
    if analysis['rare_skills']:
        print(f"\nRare Skills (few people have them):")
        for skill, count in analysis['rare_skills']:
            print(f"  {skill}: {count} people")
    
    if analysis['versatile_people']:
        print(f"\nVersatile People (many skills):")
        for person_idx, skill_count in analysis['versatile_people']:
            print(f"  Person {person_idx}: {skill_count} skills - {people[person_idx]}")
    
    # Verify solution
    if team_with_analysis:
        covered_skills = set()
        for person_idx in team_with_analysis:
            covered_skills.update(people[person_idx])
        
        missing_skills = set(req_skills) - covered_skills
        print(f"\nSolution Verification:")
        print(f"Team covers all skills: {'✓' if not missing_skills else '✗'}")
        if missing_skills:
            print(f"Missing skills: {missing_skills}")
    
    return optimized


def smallest_sufficient_team_variants():
    """
    TEAM FORMATION VARIANTS:
    =======================
    Different scenarios and modifications.
    """
    
    def largest_sufficient_team_with_budget(req_skills, people, budget):
        """Find largest team within budget constraint"""
        # Simplified version - assume each person costs 1 unit
        if budget <= 0:
            return []
        
        skill_to_id = {skill: i for i, skill in enumerate(req_skills)}
        num_skills = len(req_skills)
        
        people_masks = []
        for person_skills in people:
            mask = 0
            for skill in person_skills:
                if skill in skill_to_id:
                    mask |= (1 << skill_to_id[skill])
            people_masks.append(mask)
        
        # Try to find largest team within budget
        from itertools import combinations
        
        target_mask = (1 << num_skills) - 1
        
        for team_size in range(min(budget, len(people)), 0, -1):
            for team_indices in combinations(range(len(people)), team_size):
                team_mask = 0
                for idx in team_indices:
                    team_mask |= people_masks[idx]
                
                if team_mask == target_mask:
                    return list(team_indices)
        
        return []
    
    def count_sufficient_teams_of_size_k(req_skills, people, k):
        """Count number of sufficient teams of exactly size k"""
        skill_to_id = {skill: i for i, skill in enumerate(req_skills)}
        num_skills = len(req_skills)
        target_mask = (1 << num_skills) - 1
        
        people_masks = []
        for person_skills in people:
            mask = 0
            for skill in person_skills:
                if skill in skill_to_id:
                    mask |= (1 << skill_to_id[skill])
            people_masks.append(mask)
        
        from itertools import combinations
        
        count = 0
        for team_indices in combinations(range(len(people)), k):
            team_mask = 0
            for idx in team_indices:
                team_mask |= people_masks[idx]
            
            if team_mask == target_mask:
                count += 1
        
        return count
    
    def team_with_skill_priorities(req_skills, people, priorities):
        """Find team considering skill priorities"""
        # Higher priority skills should be covered by more people
        skill_to_id = {skill: i for i, skill in enumerate(req_skills)}
        num_skills = len(req_skills)
        
        # Modified DP considering priorities
        people_masks = []
        for person_skills in people:
            mask = 0
            for skill in person_skills:
                if skill in skill_to_id:
                    mask |= (1 << skill_to_id[skill])
            people_masks.append(mask)
        
        # Use weighted scoring
        dp = {}
        dp[0] = (0, [])  # (score, team)
        
        for mask in range(1 << num_skills):
            if mask not in dp:
                continue
            
            for i, person_mask in enumerate(people_masks):
                new_mask = mask | person_mask
                
                # Calculate score based on priorities
                score = 0
                for skill_id in range(num_skills):
                    if new_mask & (1 << skill_id):
                        score += priorities.get(req_skills[skill_id], 1)
                
                new_team = dp[mask][1] + [i]
                
                if new_mask not in dp or (len(new_team), -score) < (len(dp[new_mask][1]), -dp[new_mask][0]):
                    dp[new_mask] = (score, new_team)
        
        target_mask = (1 << num_skills) - 1
        return dp.get(target_mask, (0, []))[1]
    
    def minimum_teams_to_cover_all_people(req_skills, people):
        """Find minimum number of teams to include all people"""
        # This is more complex - simplified approximation
        team = smallest_sufficient_team_optimized_dp(req_skills, people)
        remaining_people = [i for i in range(len(people)) if i not in team]
        
        teams = [team]
        while remaining_people:
            # Greedily form next team
            next_team = remaining_people[:min(3, len(remaining_people))]
            teams.append(next_team)
            remaining_people = remaining_people[len(next_team):]
        
        return teams
    
    # Test variants
    test_cases = [
        (["java","nodejs","reactjs"], [["java"],["nodejs"],["nodejs","reactjs"]]),
        (["algorithms","math"], [["algorithms"],["math"],["algorithms","math"]]),
        (["a","b","c"], [["a","b"],["b","c"],["a","c"]])
    ]
    
    print("Team Formation Variants:")
    print("=" * 50)
    
    for req_skills, people in test_cases:
        print(f"\nRequired skills: {req_skills}")
        print(f"People: {people}")
        
        basic_team = smallest_sufficient_team_optimized_dp(req_skills, people)
        print(f"Basic smallest team: {basic_team}")
        
        # Budget constraint
        budget = len(basic_team) + 1
        budget_team = largest_sufficient_team_with_budget(req_skills, people, budget)
        print(f"Largest team with budget {budget}: {budget_team}")
        
        # Count teams of specific size
        if len(people) <= 8:
            for k in range(len(basic_team), min(len(people), 4) + 1):
                count = count_sufficient_teams_of_size_k(req_skills, people, k)
                print(f"Sufficient teams of size {k}: {count}")
        
        # Priority-based team
        priorities = {skill: i+1 for i, skill in enumerate(req_skills)}
        priority_team = team_with_skill_priorities(req_skills, people, priorities)
        print(f"Priority-based team: {priority_team}")


# Test cases
def test_smallest_sufficient_team():
    """Test all implementations with various inputs"""
    test_cases = [
        (["java","nodejs","reactjs"], [["java"],["nodejs"],["nodejs","reactjs"]], [0,2]),
        (["algorithms","math","java","reactjs","csharp","aws"], 
         [["algorithms","math","java"],["algorithms","math","reactjs"],["java","csharp","aws"],["reactjs","csharp"],["csharp","math"],["aws","java"]], 
         [1,2]),
        (["a"], [["a"]], [0]),
        (["a","b"], [["a"],["b"]], [0,1]),
        (["a","b"], [["a","b"]], [0]),
        (["a","b","c"], [["a","b"],["b","c"],["a","c"]], [0,1])
    ]
    
    print("Testing Smallest Sufficient Team Solutions:")
    print("=" * 70)
    
    for i, (req_skills, people, expected_size) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"req_skills = {req_skills}")
        print(f"people = {people}")
        print(f"Expected team size: {len(expected_size) if isinstance(expected_size, list) else expected_size}")
        
        # Skip brute force for large inputs
        if len(people) <= 10:
            try:
                brute_force = smallest_sufficient_team_brute_force(req_skills, people)
                print(f"Brute Force:      {brute_force} (size: {len(brute_force)})")
            except:
                print(f"Brute Force:      Timeout")
        
        bitmask_dp = smallest_sufficient_team_bitmask_dp(req_skills, people)
        optimized = smallest_sufficient_team_optimized_dp(req_skills, people)
        
        print(f"Bitmask DP:       {bitmask_dp} (size: {len(bitmask_dp)})")
        print(f"Optimized:        {optimized} (size: {len(optimized)})")
        
        # Verify solutions
        def verify_team(team, req_skills, people):
            if not team:
                return False
            
            covered_skills = set()
            for person_idx in team:
                if 0 <= person_idx < len(people):
                    covered_skills.update(people[person_idx])
            
            return set(req_skills).issubset(covered_skills)
        
        bitmask_valid = verify_team(bitmask_dp, req_skills, people)
        optimized_valid = verify_team(optimized, req_skills, people)
        
        print(f"Bitmask valid:    {'✓' if bitmask_valid else '✗'}")
        print(f"Optimized valid:  {'✓' if optimized_valid else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    smallest_sufficient_team_analysis(["java","nodejs","reactjs"], [["java"],["nodejs"],["nodejs","reactjs"]])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    smallest_sufficient_team_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. SKILL BITMASKS: Represent skill sets as bit patterns")
    print("2. TEAM OPTIMIZATION: Find minimum set cover using DP")
    print("3. STATE COMPRESSION: Use skills coverage as DP states")
    print("4. GREEDY PROPERTIES: Locally optimal choices don't guarantee global optimum")
    print("5. COVERAGE ANALYSIS: Identify rare skills and versatile people")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Human Resources: Optimal team formation for projects")
    print("• Resource Allocation: Minimize resources while meeting requirements")
    print("• Set Cover Problems: Classic optimization problem variant")
    print("• Project Management: Skill-based team assembly")
    print("• Computer Science: Combinatorial optimization and constraint satisfaction")


if __name__ == "__main__":
    test_smallest_sufficient_team()


"""
SMALLEST SUFFICIENT TEAM - SET COVER WITH BITMASK OPTIMIZATION:
===============================================================

This problem demonstrates Bitmask DP for set cover optimization:
- Minimum set cover with skill requirements
- Team formation under coverage constraints
- Optimization with exponential state spaces
- Real-world application of combinatorial optimization

KEY INSIGHTS:
============
1. **SKILL BITMASK REPRESENTATION**: Use bits to represent which skills are covered
2. **SET COVER OPTIMIZATION**: Find minimum number of people to cover all required skills
3. **STATE COMPRESSION**: Use skill coverage mask as DP state
4. **OPTIMAL SUBSTRUCTURE**: Optimal team built from optimal sub-teams
5. **COVERAGE ANALYSIS**: Identify critical skills and versatile team members

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(2^n × m × k) time, O(n) space
   - Try all possible team combinations
   - Exponential in number of people

2. **Bitmask DP**: O(n × 2^m) time, O(2^m) space
   - State: skill coverage bitmask
   - Transition: add one person at a time

3. **Optimized DP**: O(n × 2^m) time, O(2^m) space
   - Efficient team tracking and reconstruction
   - Memory-optimized state management

4. **Advanced Analysis**: O(n × 2^m) time, O(2^m) space
   - Include coverage analysis and optimization insights
   - Detailed performance metrics

CORE BITMASK DP ALGORITHM:
=========================
```python
def smallestSufficientTeam(req_skills, people):
    skill_to_id = {skill: i for i, skill in enumerate(req_skills)}
    num_skills = len(req_skills)
    
    # Convert people skills to bitmasks
    people_masks = []
    for person_skills in people:
        mask = 0
        for skill in person_skills:
            if skill in skill_to_id:
                mask |= (1 << skill_to_id[skill])
        people_masks.append(mask)
    
    # dp[mask] = minimum team to achieve skill coverage mask
    dp = {}
    dp[0] = []
    
    for mask in range(1 << num_skills):
        if mask not in dp:
            continue
        
        for i, person_mask in enumerate(people_masks):
            new_mask = mask | person_mask
            new_team = dp[mask] + [i]
            
            if new_mask not in dp or len(new_team) < len(dp[new_mask]):
                dp[new_mask] = new_team
    
    target_mask = (1 << num_skills) - 1
    return dp.get(target_mask, [])
```

SKILL BITMASK OPERATIONS:
========================
**Skill Mapping**: `skill_to_id = {skill: i for i, skill in enumerate(req_skills)}`
**Person Skill Mask**: `mask |= (1 << skill_id)` for each skill
**Coverage Union**: `new_mask = current_mask | person_mask`
**Complete Coverage**: `mask == (1 << num_skills) - 1`

**Skill Analysis**:
```python
def analyze_skills(req_skills, people):
    skill_coverage = [[] for _ in range(len(req_skills))]
    
    for person_idx, skills in enumerate(people):
        for skill in skills:
            if skill in skill_to_id:
                skill_id = skill_to_id[skill]
                skill_coverage[skill_id].append(person_idx)
    
    # Identify rare skills (bottlenecks)
    rare_skills = [(skill, len(coverage)) 
                   for skill, coverage in zip(req_skills, skill_coverage)
                   if len(coverage) <= 2]
```

SET COVER OPTIMIZATION:
======================
**Problem Formulation**: Given universe U (required skills) and collection S (people with skills), find minimum subcollection that covers U

**DP State**: `dp[mask]` = minimum team size to cover skills represented by mask

**Transitions**: For each person, update all reachable skill coverage states
```python
for mask in reachable_states:
    for person in people:
        new_coverage = mask | person_skills[person]
        if team_size[mask] + 1 < team_size[new_coverage]:
            update_optimal_team(new_coverage, mask, person)
```

TEAM RECONSTRUCTION:
===================
**Parent Tracking**: Store previous state and decision for each optimal transition
```python
parent[new_mask] = (prev_mask, person_added)
```

**Path Recovery**: Backtrack from complete coverage to empty state
```python
def reconstruct_team():
    team = []
    mask = target_mask
    
    while mask != 0:
        prev_mask, person = parent[mask]
        team.append(person)
        mask = prev_mask
    
    return team
```

OPTIMIZATION STRATEGIES:
=======================
**Preprocessing**: Convert skills to integer IDs for efficient bit operations
**State Pruning**: Only process reachable states
**Early Termination**: Stop when target coverage achieved
**Memory Management**: Use efficient data structures for team storage

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(n × 2^m)
- n people to consider
- 2^m possible skill coverage states
- Each transition is O(1) with proper preprocessing

**Space Complexity**: O(2^m)
- Store optimal team for each skill coverage state
- Exponential in number of required skills

**Practical Limits**: m ≤ 16-20 due to exponential state space

COVERAGE ANALYSIS TECHNIQUES:
============================
**Skill Rarity**: Identify skills covered by few people (bottlenecks)
**Person Versatility**: Find people with many required skills
**Critical Path**: Skills that must be covered by specific people
**Redundancy**: Alternative people who can cover same skill combinations

**Bottleneck Detection**:
```python
bottleneck_skills = [skill for skill in req_skills 
                     if len(people_with_skill[skill]) == 1]
```

APPLICATIONS:
============
- **Human Resources**: Optimal team formation for projects
- **Resource Planning**: Minimize resources while meeting all requirements
- **Skill Management**: Strategic hiring and team composition
- **Project Assignment**: Allocate people to projects efficiently
- **Combinatorial Optimization**: Set cover and related problems

RELATED PROBLEMS:
================
- **Set Cover Problem**: Classic NP-hard optimization problem
- **Vertex Cover**: Graph theory variant
- **Hitting Set**: Dual formulation of set cover
- **Facility Location**: Similar optimization objectives

VARIANTS:
========
- **Weighted Teams**: Consider person costs or preferences
- **Team Size Constraints**: Maximum/minimum team size limits
- **Skill Priorities**: Some skills more important than others
- **Multiple Teams**: Partition people into multiple sufficient teams

EDGE CASES:
==========
- **No Solution**: Required skills not coverable by available people
- **Single Person**: One person has all required skills
- **Disjoint Skills**: Each person covers unique skill subset
- **Redundant People**: Multiple people with identical skill sets

OPTIMIZATION TECHNIQUES:
=======================
**Greedy Initialization**: Start with greedy solution as upper bound
**Branch and Bound**: Prune search space using bounds
**Heuristic Methods**: Approximate solutions for large instances
**Preprocessing**: Remove dominated people and redundant skills

This problem showcases how Bitmask DP can solve real-world
optimization problems by efficiently exploring exponential
solution spaces while maintaining optimal substructure
and enabling practical team formation decisions.
"""
