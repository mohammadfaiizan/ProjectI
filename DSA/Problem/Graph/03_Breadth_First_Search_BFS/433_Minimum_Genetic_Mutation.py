"""
433. Minimum Genetic Mutation - Multiple Approaches
Difficulty: Medium

A gene string can be represented by an 8-character string, with choices from 'A', 'C', 'G', and 'T'.

Suppose we need to investigate a mutation from a gene string start to a gene string end where one mutation is defined as one single character changed in the gene string.

For example, "AACCGGTT" --> "AACCGGTA" is one mutation.

There is also a gene bank bank that records all the valid gene mutations. A gene mutation is valid only if it is present in the gene bank.

Given the two gene strings start and end and the gene bank bank, return the minimum number of mutations needed to mutate from start to end. If there is no such a mutation, return -1.

Note that the start point is assumed to be valid, so it might not be included in the gene bank.
"""

from typing import List, Set
from collections import deque

class MinimumGeneticMutation:
    """Multiple approaches to find minimum genetic mutations"""
    
    def minMutation_bfs_standard(self, start: str, end: str, bank: List[str]) -> int:
        """
        Approach 1: Standard BFS
        
        Use BFS to find shortest transformation path.
        
        Time: O(N*M*4) where N=bank size, M=gene length
        Space: O(N)
        """
        if end not in bank:
            return -1
        
        bank_set = set(bank)
        if start == end:
            return 0
        
        queue = deque([(start, 0)])
        visited = {start}
        genes = ['A', 'C', 'G', 'T']
        
        while queue:
            current_gene, mutations = queue.popleft()
            
            # Try all possible single mutations
            for i in range(len(current_gene)):
                for gene in genes:
                    if gene != current_gene[i]:
                        new_gene = current_gene[:i] + gene + current_gene[i+1:]
                        
                        if new_gene == end:
                            return mutations + 1
                        
                        if new_gene in bank_set and new_gene not in visited:
                            visited.add(new_gene)
                            queue.append((new_gene, mutations + 1))
        
        return -1
    
    def minMutation_bidirectional_bfs(self, start: str, end: str, bank: List[str]) -> int:
        """
        Approach 2: Bidirectional BFS
        
        Search from both start and end simultaneously.
        
        Time: O(N*M*4), Space: O(N)
        """
        if end not in bank:
            return -1
        
        if start == end:
            return 0
        
        bank_set = set(bank)
        genes = ['A', 'C', 'G', 'T']
        
        # Forward and backward search sets
        forward = {start}
        backward = {end}
        visited = set()
        mutations = 0
        
        while forward and backward:
            # Always expand the smaller set
            if len(forward) > len(backward):
                forward, backward = backward, forward
            
            mutations += 1
            next_forward = set()
            
            for current_gene in forward:
                for i in range(len(current_gene)):
                    for gene in genes:
                        if gene != current_gene[i]:
                            new_gene = current_gene[:i] + gene + current_gene[i+1:]
                            
                            if new_gene in backward:
                                return mutations
                            
                            if (new_gene in bank_set and 
                                new_gene not in visited):
                                visited.add(new_gene)
                                next_forward.add(new_gene)
            
            forward = next_forward
        
        return -1
    
    def minMutation_dfs_memoization(self, start: str, end: str, bank: List[str]) -> int:
        """
        Approach 3: DFS with Memoization
        
        Use DFS with memoization for path finding.
        
        Time: O(N*M*4), Space: O(N)
        """
        if end not in bank:
            return -1
        
        bank_set = set(bank)
        memo = {}
        genes = ['A', 'C', 'G', 'T']
        
        def dfs(current: str, target: str, visited: Set[str]) -> int:
            if current == target:
                return 0
            
            if current in memo:
                return memo[current]
            
            min_mutations = float('inf')
            
            for i in range(len(current)):
                for gene in genes:
                    if gene != current[i]:
                        new_gene = current[:i] + gene + current[i+1:]
                        
                        if (new_gene in bank_set and 
                            new_gene not in visited):
                            visited.add(new_gene)
                            result = dfs(new_gene, target, visited)
                            if result != -1:
                                min_mutations = min(min_mutations, result + 1)
                            visited.remove(new_gene)
            
            result = min_mutations if min_mutations != float('inf') else -1
            memo[current] = result
            return result
        
        return dfs(start, end, {start})
    
    def minMutation_precomputed_graph(self, start: str, end: str, bank: List[str]) -> int:
        """
        Approach 4: Precomputed Graph + BFS
        
        Build adjacency graph first, then BFS.
        
        Time: O(N^2*M + N), Space: O(N^2)
        """
        if end not in bank:
            return -1
        
        if start == end:
            return 0
        
        # Add start to bank for graph construction
        all_genes = [start] + bank
        n = len(all_genes)
        
        # Build adjacency graph
        def is_one_mutation(gene1: str, gene2: str) -> bool:
            diff_count = sum(1 for i in range(len(gene1)) if gene1[i] != gene2[i])
            return diff_count == 1
        
        graph = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if is_one_mutation(all_genes[i], all_genes[j]):
                    graph[i].append(j)
                    graph[j].append(i)
        
        # Find indices
        start_idx = 0  # start is at index 0
        end_idx = -1
        
        for i, gene in enumerate(all_genes):
            if gene == end:
                end_idx = i
                break
        
        if end_idx == -1:
            return -1
        
        # BFS on graph
        queue = deque([(start_idx, 0)])
        visited = {start_idx}
        
        while queue:
            current_idx, mutations = queue.popleft()
            
            if current_idx == end_idx:
                return mutations
            
            for neighbor_idx in graph[current_idx]:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    queue.append((neighbor_idx, mutations + 1))
        
        return -1
    
    def minMutation_optimized_generation(self, start: str, end: str, bank: List[str]) -> int:
        """
        Approach 5: Optimized Neighbor Generation
        
        Optimize neighbor generation using set operations.
        
        Time: O(N*M*4), Space: O(N)
        """
        if end not in bank:
            return -1
        
        if start == end:
            return 0
        
        bank_set = set(bank)
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            current_gene, mutations = queue.popleft()
            
            # Generate all possible neighbors efficiently
            neighbors = []
            for i in range(len(current_gene)):
                for gene in 'ACGT':
                    if gene != current_gene[i]:
                        neighbor = current_gene[:i] + gene + current_gene[i+1:]
                        neighbors.append(neighbor)
            
            for neighbor in neighbors:
                if neighbor == end:
                    return mutations + 1
                
                if neighbor in bank_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, mutations + 1))
        
        return -1

def test_minimum_genetic_mutation():
    """Test minimum genetic mutation algorithms"""
    solver = MinimumGeneticMutation()
    
    test_cases = [
        ("AACCGGTT", "AACCGGTA", ["AACCGGTA"], 1, "Single mutation"),
        ("AACCGGTT", "AAACGGTA", ["AACCGGTA","AACCGCTA","AAACGGTA"], 2, "Two mutations"),
        ("AAAAACCC", "AACCCCCC", ["AAAACCCC","AAACCCCC","AACCCCCC"], 3, "Three mutations"),
        ("AACCGGTT", "AACCGGTA", [], -1, "No valid path"),
        ("AACCGGTT", "AACCGGTT", ["AACCGGTA"], 0, "Same start and end"),
    ]
    
    algorithms = [
        ("Standard BFS", solver.minMutation_bfs_standard),
        ("Bidirectional BFS", solver.minMutation_bidirectional_bfs),
        ("DFS Memoization", solver.minMutation_dfs_memoization),
        ("Precomputed Graph", solver.minMutation_precomputed_graph),
        ("Optimized Generation", solver.minMutation_optimized_generation),
    ]
    
    print("=== Testing Minimum Genetic Mutation ===")
    
    for start, end, bank, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Start: {start}, End: {end}, Bank: {bank}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(start, end, bank)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Mutations: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_minimum_genetic_mutation()

"""
Minimum Genetic Mutation demonstrates BFS applications
in string transformation problems with constraint validation
and bidirectional search optimization techniques.
"""
