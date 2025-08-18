"""
997. Find the Town Judge
Difficulty: Easy

Problem:
In a town, there are n people labeled from 1 to n. There is a rumor that one of these 
people is secretly the town judge.

If the town judge exists, then:
1. The town judge trusts nobody.
2. Everybody (except for the town judge) trusts the town judge.
3. There is exactly one person that satisfies properties 1 and 2.

You are given an array trust where trust[i] = [ai, bi] representing that the person 
labeled ai trusts the person labeled bi.

Return the label of the town judge if the town judge exists and can be identified, 
or return -1 otherwise.

Examples:
Input: n = 2, trust = [[1,2]]
Output: 2

Input: n = 3, trust = [[1,3],[2,3]]
Output: 3

Input: n = 3, trust = [[1,3],[2,3],[3,1]]
Output: -1

Constraints:
- 1 <= n <= 1000
- 0 <= trust.length <= 10^4
- trust[i].length == 2
- All the pairs of trust are unique
- ai != bi
- 1 <= ai, bi <= n
"""

from typing import List

class Solution:
    def findJudge_approach1_indegree_outdegree(self, n: int, trust: List[List[int]]) -> int:
        """
        Approach 1: In-degree and Out-degree counting
        
        Think of this as a directed graph where trust[i] = [a, b] means
        there's an edge from a to b (a trusts b).
        
        Judge properties:
        - In-degree = n-1 (everyone trusts them)
        - Out-degree = 0 (they trust nobody)
        
        Time: O(T + N) where T = len(trust)
        Space: O(N)
        """
        if n == 1:
            return 1  # Only one person, must be the judge
        
        # Count in-degree and out-degree for each person
        in_degree = [0] * (n + 1)   # How many people trust them
        out_degree = [0] * (n + 1)  # How many people they trust
        
        for a, b in trust:
            out_degree[a] += 1
            in_degree[b] += 1
        
        # Find person with in_degree = n-1 and out_degree = 0
        for person in range(1, n + 1):
            if in_degree[person] == n - 1 and out_degree[person] == 0:
                return person
        
        return -1
    
    def findJudge_approach2_net_trust(self, n: int, trust: List[List[int]]) -> int:
        """
        Approach 2: Net trust counting (Optimized)
        
        Instead of tracking in_degree and out_degree separately,
        we can use a single array where:
        - trust_count[person] = in_degree - out_degree
        
        For the judge: trust_count[judge] = (n-1) - 0 = n-1
        
        Time: O(T + N) where T = len(trust)
        Space: O(N)
        """
        if n == 1:
            return 1
        
        trust_count = [0] * (n + 1)
        
        for a, b in trust:
            trust_count[a] -= 1  # a trusts someone (out-degree)
            trust_count[b] += 1  # b is trusted by someone (in-degree)
        
        # Judge has trust_count = n-1
        for person in range(1, n + 1):
            if trust_count[person] == n - 1:
                return person
        
        return -1
    
    def findJudge_approach3_set_operations(self, n: int, trust: List[List[int]]) -> int:
        """
        Approach 3: Set operations approach
        
        Use sets to track:
        - Who trusts others (cannot be judge)
        - Who is trusted by others (potential judge)
        
        Time: O(T + N) where T = len(trust)
        Space: O(N)
        """
        if n == 1:
            return 1
        
        trusters = set()  # People who trust others
        trusted_by = {}   # person -> count of people who trust them
        
        for a, b in trust:
            trusters.add(a)
            trusted_by[b] = trusted_by.get(b, 0) + 1
        
        # Judge: not in trusters and trusted by n-1 people
        for person in range(1, n + 1):
            if person not in trusters and trusted_by.get(person, 0) == n - 1:
                return person
        
        return -1

def test_find_judge():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, trust, expected)
        (2, [[1, 2]], 2),
        (3, [[1, 3], [2, 3]], 3),
        (3, [[1, 3], [2, 3], [3, 1]], -1),
        (3, [[1, 2], [2, 3]], -1),
        (4, [[1, 3], [1, 4], [2, 3], [2, 4], [4, 3]], 3),
        (1, [], 1),  # Edge case: single person
        (2, [], -1), # Edge case: no trust relationships
    ]
    
    approaches = [
        ("In-degree/Out-degree", solution.findJudge_approach1_indegree_outdegree),
        ("Net Trust", solution.findJudge_approach2_net_trust),
        ("Set Operations", solution.findJudge_approach3_set_operations),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, trust, expected) in enumerate(test_cases):
            result = func(n, trust)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Input: n={n}, trust={trust}")
            print(f"         Expected: {expected}, Got: {result}")

if __name__ == "__main__":
    test_find_judge()

"""
Graph Theory Concepts:
1. Directed Graph Representation
2. In-degree and Out-degree
3. Graph Properties Analysis
4. Node Centrality (Judge = highest in-degree, zero out-degree)

Key Insights:
- This is fundamentally a graph problem disguised as a logic puzzle
- The judge is the node with specific degree properties
- Multiple solution approaches show different graph analysis techniques
- Time complexity is optimal O(T + N) for all approaches

Real-world Applications:
- Social network analysis (finding influencers)
- Recommendation systems (trust relationships)
- Authority ranking in networks
- Fraud detection (unusual trust patterns)
"""
