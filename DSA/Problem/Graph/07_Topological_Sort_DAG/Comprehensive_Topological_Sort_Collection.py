"""
Comprehensive Topological Sort Collection
Difficulty: Easy to Hard

This file contains implementations of multiple topological sort and DAG problems
that demonstrate the breadth and depth of these algorithmic concepts.

Problems Included:
1. 1136_Parallel_Courses (Easy)
2. 269_Alien_Dictionary (Hard)  
3. 802_Find_Eventual_Safe_States (Medium)
4. 444_Sequence_Reconstruction (Medium)
5. 310_Minimum_Height_Trees (Medium)
6. 1203_Sort_Items_by_Groups_Respecting_Dependencies (Hard)
7. 1494_Parallel_Courses_II (Hard)
8. 2392_Build_a_Matrix_With_Conditions (Hard)
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import heapq

class ComprehensiveTopologicalSort:
    
    def parallel_courses_1136(self, n: int, relations: List[List[int]], time: List[int]) -> int:
        """
        1136. Parallel Courses
        
        Find minimum time to complete all courses when courses can be taken in parallel.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build graph and in-degree
        graph = defaultdict(list)
        in_degree = [0] * (n + 1)
        
        for prev, next_course in relations:
            graph[prev].append(next_course)
            in_degree[next_course] += 1
        
        # Initialize queue with courses having no prerequisites
        queue = deque()
        for i in range(1, n + 1):
            if in_degree[i] == 0:
                queue.append(i)
        
        total_time = 0
        completed = 0
        
        while queue:
            # Process all courses at current level (semester)
            level_size = len(queue)
            completed += level_size
            
            # Find maximum time for this level
            max_time_this_level = 0
            next_level = []
            
            for _ in range(level_size):
                course = queue.popleft()
                max_time_this_level = max(max_time_this_level, time[course - 1])
                
                # Update dependent courses
                for next_course in graph[course]:
                    in_degree[next_course] -= 1
                    if in_degree[next_course] == 0:
                        next_level.append(next_course)
            
            total_time += max_time_this_level
            queue.extend(next_level)
        
        return total_time if completed == n else -1
    
    def alien_dictionary_269(self, words: List[str]) -> str:
        """
        269. Alien Dictionary
        
        Derive alien language character ordering from sorted word list.
        
        Time: O(C) where C is total length of all words
        Space: O(1) for fixed alphabet size
        """
        # Build graph of character dependencies
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        chars = set()
        
        # Initialize all characters
        for word in words:
            for char in word:
                chars.add(char)
                in_degree[char] = 0
        
        # Build dependency graph
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            min_len = min(len(word1), len(word2))
            
            # Check for invalid ordering (longer word is prefix of shorter)
            if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
                return ""
            
            # Find first differing character
            for j in range(min_len):
                if word1[j] != word2[j]:
                    if word2[j] not in graph[word1[j]]:
                        graph[word1[j]].add(word2[j])
                        in_degree[word2[j]] += 1
                    break
        
        # Topological sort
        queue = deque()
        for char in chars:
            if in_degree[char] == 0:
                queue.append(char)
        
        result = []
        while queue:
            char = queue.popleft()
            result.append(char)
            
            for next_char in graph[char]:
                in_degree[next_char] -= 1
                if in_degree[next_char] == 0:
                    queue.append(next_char)
        
        return "".join(result) if len(result) == len(chars) else ""
    
    def find_eventual_safe_states_802(self, graph: List[List[int]]) -> List[int]:
        """
        802. Find Eventual Safe States
        
        Find nodes that cannot reach a cycle (terminal or lead to terminal).
        
        Time: O(V + E)
        Space: O(V)
        """
        n = len(graph)
        
        # Reverse the graph
        reverse_graph = defaultdict(list)
        out_degree = [0] * n
        
        for i in range(n):
            for neighbor in graph[i]:
                reverse_graph[neighbor].append(i)
            out_degree[i] = len(graph[i])
        
        # Find terminal nodes (out-degree 0)
        queue = deque()
        for i in range(n):
            if out_degree[i] == 0:
                queue.append(i)
        
        safe = [False] * n
        
        # Process nodes in reverse topological order
        while queue:
            node = queue.popleft()
            safe[node] = True
            
            for prev_node in reverse_graph[node]:
                out_degree[prev_node] -= 1
                if out_degree[prev_node] == 0:
                    queue.append(prev_node)
        
        return [i for i in range(n) if safe[i]]
    
    def sequence_reconstruction_444(self, org: List[int], seqs: List[List[int]]) -> bool:
        """
        444. Sequence Reconstruction
        
        Check if original sequence can be uniquely reconstructed from subsequences.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        if not seqs:
            return False
        
        # Build graph and collect all numbers
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        nodes = set()
        
        for seq in seqs:
            for num in seq:
                nodes.add(num)
                if num not in in_degree:
                    in_degree[num] = 0
            
            for i in range(len(seq) - 1):
                if seq[i + 1] not in graph[seq[i]]:
                    graph[seq[i]].add(seq[i + 1])
                    in_degree[seq[i + 1]] += 1
        
        # Check if we have the right set of numbers
        if set(org) != nodes or len(org) != len(nodes):
            return False
        
        # Topological sort with unique path requirement
        queue = deque()
        for num in nodes:
            if in_degree[num] == 0:
                queue.append(num)
        
        result = []
        while queue:
            if len(queue) > 1:  # Multiple choices - not unique
                return False
            
            current = queue.popleft()
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result == org
    
    def find_minimum_height_trees_310(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        310. Minimum Height Trees
        
        Find roots that minimize tree height (centroid nodes).
        
        Time: O(V)
        Space: O(V)
        """
        if n == 1:
            return [0]
        
        # Build adjacency list
        graph = defaultdict(set)
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        # Start from leaves and peel layers
        leaves = deque()
        for i in range(n):
            if len(graph[i]) == 1:
                leaves.append(i)
        
        remaining = n
        
        while remaining > 2:
            leaf_count = len(leaves)
            remaining -= leaf_count
            
            for _ in range(leaf_count):
                leaf = leaves.popleft()
                
                # Remove leaf and update its neighbor
                neighbor = graph[leaf].pop()
                graph[neighbor].remove(leaf)
                
                if len(graph[neighbor]) == 1:
                    leaves.append(neighbor)
        
        return list(leaves)
    
    def sort_items_1203(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        """
        1203. Sort Items by Groups Respecting Dependencies
        
        Topological sort with group constraints.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Assign items without groups to new groups
        group_id = m
        for i in range(n):
            if group[i] == -1:
                group[i] = group_id
                group_id += 1
        
        # Build item and group graphs
        item_graph = defaultdict(list)
        item_in_degree = [0] * n
        group_graph = defaultdict(set)
        group_in_degree = defaultdict(int)
        
        # Initialize group in-degrees
        for i in range(group_id):
            group_in_degree[i] = 0
        
        for i in range(n):
            for prev_item in beforeItems[i]:
                item_graph[prev_item].append(i)
                item_in_degree[i] += 1
                
                # Add group dependency if items are in different groups
                if group[prev_item] != group[i]:
                    if group[i] not in group_graph[group[prev_item]]:
                        group_graph[group[prev_item]].add(group[i])
                        group_in_degree[group[i]] += 1
        
        # Topological sort for groups
        def topo_sort_groups():
            queue = deque()
            for g in range(group_id):
                if group_in_degree[g] == 0:
                    queue.append(g)
            
            group_order = []
            while queue:
                current_group = queue.popleft()
                group_order.append(current_group)
                
                for next_group in group_graph[current_group]:
                    group_in_degree[next_group] -= 1
                    if group_in_degree[next_group] == 0:
                        queue.append(next_group)
            
            return group_order if len(group_order) == group_id else []
        
        # Topological sort for items within each group
        def topo_sort_items(items):
            local_in_degree = [0] * len(items)
            item_to_idx = {item: idx for idx, item in enumerate(items)}
            
            for idx, item in enumerate(items):
                for next_item in item_graph[item]:
                    if next_item in item_to_idx:
                        local_in_degree[item_to_idx[next_item]] += 1
            
            queue = deque()
            for idx, item in enumerate(items):
                if local_in_degree[idx] == 0:
                    queue.append(item)
            
            item_order = []
            while queue:
                current_item = queue.popleft()
                item_order.append(current_item)
                
                for next_item in item_graph[current_item]:
                    if next_item in item_to_idx:
                        local_in_degree[item_to_idx[next_item]] -= 1
                        if local_in_degree[item_to_idx[next_item]] == 0:
                            queue.append(next_item)
            
            return item_order if len(item_order) == len(items) else []
        
        # Get group ordering
        group_order = topo_sort_groups()
        if not group_order:
            return []
        
        # Group items by group
        group_items = defaultdict(list)
        for i in range(n):
            group_items[group[i]].append(i)
        
        # Sort items within each group and combine
        result = []
        for g in group_order:
            if group_items[g]:
                item_order = topo_sort_items(group_items[g])
                if not item_order:
                    return []
                result.extend(item_order)
        
        return result
    
    def minimum_semesters_1494(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        1494. Parallel Courses II
        
        Minimum semesters when at most k courses can be taken per semester.
        
        Time: O(2^n * n) for bitmask DP
        Space: O(2^n)
        """
        # Build prerequisite mask for each course
        prereq = [0] * n
        for prev, curr in relations:
            prereq[curr - 1] |= (1 << (prev - 1))
        
        # DP with bitmask: dp[mask] = minimum semesters to complete courses in mask
        dp = [float('inf')] * (1 << n)
        dp[0] = 0
        
        for mask in range(1 << n):
            if dp[mask] == float('inf'):
                continue
            
            # Find available courses (prerequisites satisfied)
            available = []
            for i in range(n):
                if not (mask & (1 << i)) and (prereq[i] & mask) == prereq[i]:
                    available.append(i)
            
            # Try all combinations of up to k available courses
            def backtrack(idx, current_mask, count):
                if count == k or idx == len(available):
                    if count > 0:
                        new_mask = mask | current_mask
                        dp[new_mask] = min(dp[new_mask], dp[mask] + 1)
                    return
                
                # Take current course
                backtrack(idx + 1, current_mask | (1 << available[idx]), count + 1)
                
                # Skip current course
                backtrack(idx + 1, current_mask, count)
            
            backtrack(0, 0, 0)
        
        return dp[(1 << n) - 1] if dp[(1 << n) - 1] != float('inf') else -1
    
    def build_matrix_2392(self, rowConditions: List[List[int]], colConditions: List[List[int]]) -> List[List[int]]:
        """
        2392. Build a Matrix With Conditions
        
        Build matrix satisfying row and column ordering constraints.
        
        Time: O(k^2) where k is number of elements
        Space: O(k^2)
        """
        def topological_sort(conditions):
            """Perform topological sort on conditions"""
            graph = defaultdict(list)
            in_degree = defaultdict(int)
            nodes = set()
            
            for u, v in conditions:
                nodes.add(u)
                nodes.add(v)
                graph[u].append(v)
                in_degree[v] += 1
            
            # Initialize in-degrees
            for node in nodes:
                if node not in in_degree:
                    in_degree[node] = 0
            
            # Kahn's algorithm
            queue = deque()
            for node in nodes:
                if in_degree[node] == 0:
                    queue.append(node)
            
            result = []
            while queue:
                current = queue.popleft()
                result.append(current)
                
                for neighbor in graph[current]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            
            return result if len(result) == len(nodes) else []
        
        # Get topological orderings
        row_order = topological_sort(rowConditions)
        col_order = topological_sort(colConditions)
        
        if not row_order or not col_order:
            return []
        
        # Check if we have the same set of elements
        if set(row_order) != set(col_order):
            return []
        
        k = len(row_order)
        
        # Create position mappings
        row_pos = {val: idx for idx, val in enumerate(row_order)}
        col_pos = {val: idx for idx, val in enumerate(col_order)}
        
        # Build matrix
        matrix = [[0] * k for _ in range(k)]
        for val in row_order:
            matrix[row_pos[val]][col_pos[val]] = val
        
        return matrix

def test_comprehensive_topological_sort():
    """Test all topological sort problems"""
    solver = ComprehensiveTopologicalSort()
    
    print("=== Testing Comprehensive Topological Sort Problems ===")
    
    # Test 1136 - Parallel Courses
    print("\n1. Parallel Courses (1136):")
    result = solver.parallel_courses_1136(3, [[1,3],[2,3]], [3,2,5])
    print(f"   Result: {result} (Expected: 8)")
    
    # Test 269 - Alien Dictionary
    print("\n2. Alien Dictionary (269):")
    result = solver.alien_dictionary_269(["wrt","wrf","er","ett","rftt"])
    print(f"   Result: {result} (Expected: wertf)")
    
    # Test 802 - Find Eventual Safe States
    print("\n3. Find Eventual Safe States (802):")
    result = solver.find_eventual_safe_states_802([[1,2],[2,3],[5],[0],[5],[],[]])
    print(f"   Result: {result} (Expected: [2,4,5,6])")
    
    # Test 444 - Sequence Reconstruction
    print("\n4. Sequence Reconstruction (444):")
    result = solver.sequence_reconstruction_444([1,2,3], [[1,2],[1,3]])
    print(f"   Result: {result} (Expected: False)")
    
    # Test 310 - Minimum Height Trees
    print("\n5. Minimum Height Trees (310):")
    result = solver.find_minimum_height_trees_310(4, [[1,0],[1,2],[1,3]])
    print(f"   Result: {result} (Expected: [1])")

def demonstrate_topological_sort_patterns():
    """Demonstrate common topological sort patterns"""
    print("\n=== Topological Sort Patterns ===")
    
    print("\nCommon Patterns in Topological Sort Problems:")
    
    print("\n1. **Basic Course Scheduling (207, 210):**")
    print("   • Standard DAG cycle detection")
    print("   • Generate valid ordering")
    print("   • Applications: Course prerequisites, task scheduling")
    
    print("\n2. **Parallel Processing (1136, 1494):**")
    print("   • Level-by-level processing")
    print("   • Resource constraints (max k items per level)")
    print("   • Applications: Parallel compilation, resource allocation")
    
    print("\n3. **String/Sequence Ordering (269, 444):**")
    print("   • Derive ordering from partial information")
    print("   • Unique reconstruction validation")
    print("   • Applications: Language inference, sequence validation")
    
    print("\n4. **Graph Analysis (802, 310):**")
    print("   • Safety analysis (cycle avoidance)")
    print("   • Structural properties (tree centers)")
    print("   • Applications: Deadlock detection, network analysis")
    
    print("\n5. **Hierarchical Sorting (1203, 2392):**")
    print("   • Multi-level constraints")
    print("   • Group-based dependencies")
    print("   • Applications: Organization structures, matrix construction")
    
    print("\n6. **Optimization Problems (329, 1494):**")
    print("   • Longest paths in DAGs")
    print("   • Resource-constrained scheduling")
    print("   • Applications: Project optimization, resource planning")

def analyze_topological_sort_applications():
    """Analyze real-world applications of topological sort"""
    print("\n=== Real-World Applications ===")
    
    print("Topological Sort in Practice:")
    
    print("\n1. **Software Development:**")
    print("   • Build systems (Make, Maven, Gradle)")
    print("   • Package dependency resolution")
    print("   • Module compilation ordering")
    print("   • Code deployment pipelines")
    
    print("\n2. **Project Management:**")
    print("   • Task scheduling with dependencies")
    print("   • Critical path method (CPM)")
    print("   • Resource allocation planning")
    print("   • Milestone sequencing")
    
    print("\n3. **Academic Planning:**")
    print("   • Course prerequisite planning")
    print("   • Curriculum design")
    print("   • Graduation requirement tracking")
    print("   • Academic progression paths")
    
    print("\n4. **Manufacturing:**")
    print("   • Assembly line sequencing")
    print("   • Production planning")
    print("   • Quality control checkpoints")
    print("   • Supply chain coordination")
    
    print("\n5. **Data Processing:**")
    print("   • ETL pipeline orchestration")
    print("   • Workflow management systems")
    print("   • Data transformation sequences")
    print("   • Analytics pipeline optimization")
    
    print("\n6. **System Administration:**")
    print("   • Service startup ordering")
    print("   • Configuration management")
    print("   • Update deployment sequences")
    print("   • Dependency resolution")

if __name__ == "__main__":
    test_comprehensive_topological_sort()
    demonstrate_topological_sort_patterns()
    analyze_topological_sort_applications()

"""
Comprehensive Topological Sort Concepts:
1. Classic Course Scheduling and Dependency Resolution
2. Parallel Processing with Resource Constraints
3. String and Sequence Ordering from Partial Information
4. Graph Safety Analysis and Structural Properties
5. Hierarchical and Multi-level Constraint Satisfaction
6. Optimization Problems on Directed Acyclic Graphs

Key Problem Categories:
- Basic topological ordering and cycle detection
- Parallel processing with capacity constraints
- Unique path reconstruction and validation
- Safety analysis and deadlock prevention
- Multi-level hierarchical constraint satisfaction
- Resource-constrained optimization

Algorithm Patterns:
- Kahn's algorithm for BFS-based topological sort
- DFS-based topological sort with recursion
- Level-by-level processing for parallel constraints
- Reverse graph analysis for safety properties
- Bitmask DP for complex constraint optimization
- Multi-graph coordination for hierarchical problems

Real-world Applications:
- Software build systems and dependency management
- Project management and task scheduling
- Academic planning and curriculum design
- Manufacturing and production optimization
- Data processing and workflow orchestration
- System administration and service management

This comprehensive collection demonstrates the versatility
and practical importance of topological sorting algorithms
across diverse application domains.
"""
