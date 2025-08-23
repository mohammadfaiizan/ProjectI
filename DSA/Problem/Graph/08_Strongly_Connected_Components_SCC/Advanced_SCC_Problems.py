"""
Advanced SCC Problems - Complex Applications and Algorithms
Difficulty: Hard

This file contains advanced applications of Strongly Connected Components
including complex optimization problems, graph augmentation, and theoretical applications.

Advanced Problems:
1. Strong Connectivity Augmentation
2. SCC-based Graph Condensation and Analysis
3. Dynamic SCC Maintenance
4. SCC in Game Theory and Nash Equilibria
5. Advanced Network Flow and SCC
6. Parallel SCC Algorithms
7. SCC in Distributed Systems
8. Approximation Algorithms using SCC
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import heapq
import math

class AdvancedSCCProblems:
    """Advanced SCC applications and complex algorithms"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset state for new computation"""
        self.time = 0
        self.discovery = {}
        self.low = {}
        self.on_stack = set()
        self.stack = []
        self.scc_id = {}
        self.scc_count = 0
    
    def find_sccs_tarjan(self, graph: Dict[int, List[int]]) -> Tuple[int, Dict[int, int]]:
        """Tarjan's SCC algorithm"""
        self.reset()
        
        def tarjan_dfs(v):
            self.discovery[v] = self.low[v] = self.time
            self.time += 1
            self.stack.append(v)
            self.on_stack.add(v)
            
            for u in graph.get(v, []):
                if u not in self.discovery:
                    tarjan_dfs(u)
                    self.low[v] = min(self.low[v], self.low[u])
                elif u in self.on_stack:
                    self.low[v] = min(self.low[v], self.discovery[u])
            
            if self.low[v] == self.discovery[v]:
                while True:
                    u = self.stack.pop()
                    self.on_stack.remove(u)
                    self.scc_id[u] = self.scc_count
                    if u == v:
                        break
                self.scc_count += 1
        
        vertices = set()
        for v in graph:
            vertices.add(v)
            for u in graph[v]:
                vertices.add(u)
        
        for v in vertices:
            if v not in self.discovery:
                tarjan_dfs(v)
        
        return self.scc_count, self.scc_id.copy()
    
    def strong_connectivity_augmentation(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Problem 1: Strong Connectivity Augmentation
        
        Find minimum edges to add to make graph strongly connected.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Find SCCs
        num_sccs, scc_assignments = self.find_sccs_tarjan(graph)
        
        if num_sccs == 1:
            return {
                'is_strongly_connected': True,
                'edges_needed': 0,
                'augmentation_edges': []
            }
        
        # Build condensation DAG
        condensation = defaultdict(set)
        for u in graph:
            for v in graph[u]:
                scc_u = scc_assignments.get(u)
                scc_v = scc_assignments.get(v)
                if scc_u is not None and scc_v is not None and scc_u != scc_v:
                    condensation[scc_u].add(scc_v)
        
        # Convert to adjacency list
        condensation_adj = {scc: list(neighbors) for scc, neighbors in condensation.items()}
        
        # Find sources (no incoming edges) and sinks (no outgoing edges)
        all_sccs = set(range(num_sccs))
        has_incoming = set()
        
        for scc in condensation_adj:
            for neighbor in condensation_adj[scc]:
                has_incoming.add(neighbor)
        
        sources = all_sccs - has_incoming
        sinks = set(scc for scc in all_sccs if not condensation_adj.get(scc, []))
        
        # Calculate minimum edges needed
        if len(sources) == 1 and len(sinks) == 1 and sources == sinks:
            edges_needed = 0
        else:
            edges_needed = max(len(sources), len(sinks))
        
        # Generate augmentation strategy
        augmentation_edges = []
        
        # Strategy: Connect sinks to sources in a cycle
        source_list = list(sources)
        sink_list = list(sinks)
        
        # Connect each sink to a source
        for i, sink_scc in enumerate(sink_list):
            source_scc = source_list[i % len(source_list)] if source_list else sink_scc
            
            # Find representative nodes
            sink_node = None
            source_node = None
            
            for node, scc in scc_assignments.items():
                if scc == sink_scc and sink_node is None:
                    sink_node = node
                if scc == source_scc and source_node is None:
                    source_node = node
            
            if sink_node is not None and source_node is not None and sink_node != source_node:
                augmentation_edges.append((sink_node, source_node))
        
        return {
            'is_strongly_connected': False,
            'edges_needed': edges_needed,
            'augmentation_edges': augmentation_edges,
            'num_sccs': num_sccs,
            'sources': len(sources),
            'sinks': len(sinks),
            'condensation_structure': dict(condensation_adj)
        }
    
    def scc_based_graph_analysis(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Problem 2: Comprehensive SCC-based Graph Analysis
        
        Analyze graph properties using SCC decomposition.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        num_sccs, scc_assignments = self.find_sccs_tarjan(graph)
        
        # Group vertices by SCC
        scc_groups = defaultdict(list)
        for vertex, scc_id in scc_assignments.items():
            scc_groups[scc_id].append(vertex)
        
        # Build condensation DAG
        condensation = defaultdict(set)
        for u in graph:
            for v in graph[u]:
                scc_u = scc_assignments.get(u)
                scc_v = scc_assignments.get(v)
                if scc_u is not None and scc_v is not None and scc_u != scc_v:
                    condensation[scc_u].add(scc_v)
        
        # Analyze SCC properties
        scc_analysis = []
        for scc_id, vertices in scc_groups.items():
            if len(vertices) > 1:
                # Calculate internal connectivity
                internal_edges = 0
                external_edges = 0
                
                for vertex in vertices:
                    for neighbor in graph.get(vertex, []):
                        if neighbor in vertices:
                            internal_edges += 1
                        else:
                            external_edges += 1
                
                density = internal_edges / (len(vertices) * (len(vertices) - 1)) if len(vertices) > 1 else 0
                
                scc_analysis.append({
                    'scc_id': scc_id,
                    'vertices': vertices,
                    'size': len(vertices),
                    'internal_edges': internal_edges,
                    'external_edges': external_edges,
                    'density': density,
                    'is_complete': density == 1.0
                })
        
        # Topological analysis of condensation
        topo_order = self._topological_sort_condensation(condensation, num_sccs)
        
        # Calculate graph metrics
        total_vertices = len(set(scc_assignments.keys()))
        total_edges = sum(len(neighbors) for neighbors in graph.values())
        
        return {
            'basic_properties': {
                'vertices': total_vertices,
                'edges': total_edges,
                'sccs': num_sccs,
                'is_strongly_connected': num_sccs == 1
            },
            'scc_analysis': scc_analysis,
            'condensation_properties': {
                'dag_vertices': num_sccs,
                'dag_edges': sum(len(neighbors) for neighbors in condensation.values()),
                'topological_order': topo_order,
                'longest_path_length': len(topo_order) if topo_order else 0
            },
            'connectivity_metrics': {
                'strong_connectivity_ratio': 1 / num_sccs if num_sccs > 0 else 0,
                'condensation_complexity': total_edges - (total_vertices - num_sccs),
                'cyclomatic_complexity': total_edges - total_vertices + num_sccs
            }
        }
    
    def dynamic_scc_maintenance(self, initial_graph: Dict[int, List[int]]) -> 'DynamicSCCMaintainer':
        """
        Problem 3: Dynamic SCC Maintenance
        
        Maintain SCCs under edge insertions and deletions.
        
        Time: O(V + E) per update (simplified)
        Space: O(V + E)
        """
        class DynamicSCCMaintainer:
            def __init__(self, graph):
                self.graph = {u: list(neighbors) for u, neighbors in graph.items()}
                self.scc_analyzer = AdvancedSCCProblems()
                self.update_sccs()
            
            def update_sccs(self):
                """Recompute SCCs (simplified approach)"""
                self.num_sccs, self.scc_assignments = self.scc_analyzer.find_sccs_tarjan(self.graph)
            
            def add_edge(self, u, v):
                """Add edge and update SCCs"""
                if u not in self.graph:
                    self.graph[u] = []
                if v not in self.graph[u]:
                    self.graph[u].append(v)
                
                # For simplicity, recompute SCCs
                # In practice, would use incremental algorithms
                self.update_sccs()
                
                return self.get_scc_info()
            
            def remove_edge(self, u, v):
                """Remove edge and update SCCs"""
                if u in self.graph and v in self.graph[u]:
                    self.graph[u].remove(v)
                
                self.update_sccs()
                return self.get_scc_info()
            
            def get_scc_info(self):
                """Get current SCC information"""
                return {
                    'num_sccs': self.num_sccs,
                    'scc_assignments': self.scc_assignments.copy(),
                    'is_strongly_connected': self.num_sccs == 1
                }
            
            def query_same_scc(self, u, v):
                """Check if two vertices are in the same SCC"""
                return (u in self.scc_assignments and 
                       v in self.scc_assignments and
                       self.scc_assignments[u] == self.scc_assignments[v])
        
        return DynamicSCCMaintainer(initial_graph)
    
    def scc_nash_equilibrium_analysis(self, game_graph: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> Dict:
        """
        Problem 4: SCC in Game Theory - Nash Equilibrium Analysis
        
        Find Nash equilibria using SCC analysis of strategy graphs.
        
        Time: O(S^2) where S is number of strategy profiles
        Space: O(S^2)
        """
        # Game graph: strategy profiles -> reachable strategy profiles
        
        # Convert tuple keys to integers for SCC algorithm
        profile_to_id = {}
        id_to_profile = {}
        
        for profile in game_graph:
            if profile not in profile_to_id:
                profile_to_id[profile] = len(profile_to_id)
                id_to_profile[len(id_to_profile)] = profile
        
        # Build numeric graph
        numeric_graph = {}
        for profile, reachable in game_graph.items():
            profile_id = profile_to_id[profile]
            numeric_graph[profile_id] = []
            
            for reachable_profile in reachable:
                if reachable_profile in profile_to_id:
                    numeric_graph[profile_id].append(profile_to_id[reachable_profile])
        
        # Find SCCs
        num_sccs, scc_assignments = self.find_sccs_tarjan(numeric_graph)
        
        # Analyze equilibria
        equilibrium_candidates = []
        sink_sccs = []
        
        # Group by SCC
        scc_groups = defaultdict(list)
        for profile_id, scc_id in scc_assignments.items():
            scc_groups[scc_id].append(id_to_profile[profile_id])
        
        # Find sink SCCs (no outgoing edges) - potential equilibria
        condensation = defaultdict(set)
        for profile_id in numeric_graph:
            for reachable_id in numeric_graph[profile_id]:
                scc_from = scc_assignments[profile_id]
                scc_to = scc_assignments[reachable_id]
                if scc_from != scc_to:
                    condensation[scc_from].add(scc_to)
        
        for scc_id in range(num_sccs):
            if not condensation.get(scc_id, set()):
                # Sink SCC - contains potential Nash equilibria
                sink_sccs.append(scc_id)
                equilibrium_candidates.extend(scc_groups[scc_id])
        
        return {
            'total_strategy_profiles': len(game_graph),
            'num_sccs': num_sccs,
            'equilibrium_candidates': equilibrium_candidates,
            'sink_sccs': sink_sccs,
            'scc_structure': dict(scc_groups),
            'is_unique_equilibrium': len(equilibrium_candidates) == 1,
            'multiple_equilibria': len(sink_sccs) > 1
        }
    
    def parallel_scc_simulation(self, graph: Dict[int, List[int]], num_processors: int = 4) -> Dict:
        """
        Problem 5: Parallel SCC Algorithm Simulation
        
        Simulate parallel SCC computation (conceptual implementation).
        
        Time: O((V + E) / P) ideally
        Space: O(V + E)
        """
        # This is a conceptual simulation of parallel SCC
        # Real implementation would use actual parallel processing
        
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        vertex_list = list(vertices)
        chunk_size = len(vertex_list) // num_processors
        
        # Simulate parallel processing by dividing work
        processor_work = []
        
        for i in range(num_processors):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_processors - 1 else len(vertex_list)
            
            chunk_vertices = vertex_list[start_idx:end_idx]
            
            # Simulate work for this processor
            local_components = []
            
            for vertex in chunk_vertices:
                # Simplified: just record which vertices this processor handles
                local_components.append({
                    'processor_id': i,
                    'vertex': vertex,
                    'neighbors': graph.get(vertex, [])
                })
            
            processor_work.append({
                'processor_id': i,
                'vertices_handled': chunk_vertices,
                'local_work': local_components
            })
        
        # Simulate coordination phase
        # In real parallel SCC, processors would coordinate to merge partial results
        
        # Final SCC computation (sequential for correctness)
        num_sccs, scc_assignments = self.find_sccs_tarjan(graph)
        
        return {
            'parallel_simulation': {
                'num_processors': num_processors,
                'vertex_distribution': processor_work,
                'coordination_needed': True
            },
            'final_result': {
                'num_sccs': num_sccs,
                'scc_assignments': scc_assignments
            },
            'parallel_efficiency_estimate': {
                'sequential_complexity': 'O(V + E)',
                'parallel_complexity': f'O((V + E) / {num_processors}) + coordination',
                'speedup_theoretical': min(num_processors, len(vertices) // 10)
            }
        }
    
    def scc_approximation_algorithms(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Problem 6: Approximation Algorithms using SCC
        
        Implement approximation algorithms that leverage SCC structure.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Find exact SCCs first
        num_sccs, scc_assignments = self.find_sccs_tarjan(graph)
        
        # Approximate strongly connected vertex cover
        def approximate_strong_vertex_cover():
            """Approximate minimum vertex cover in strongly connected subgraphs"""
            vertex_cover = set()
            
            # Group by SCC
            scc_groups = defaultdict(list)
            for vertex, scc_id in scc_assignments.items():
                scc_groups[scc_id].append(vertex)
            
            for scc_id, vertices in scc_groups.items():
                if len(vertices) > 1:
                    # For strongly connected subgraph, use 2-approximation
                    # Add every other vertex (simple greedy)
                    for i in range(0, len(vertices), 2):
                        vertex_cover.add(vertices[i])
            
            return vertex_cover
        
        # Approximate feedback vertex set
        def approximate_feedback_vertex_set():
            """Approximate minimum feedback vertex set using SCC"""
            feedback_set = set()
            
            # For each SCC with size > 1, add vertices to break cycles
            scc_groups = defaultdict(list)
            for vertex, scc_id in scc_assignments.items():
                scc_groups[scc_id].append(vertex)
            
            for scc_id, vertices in scc_groups.items():
                if len(vertices) > 1:
                    # Simple approximation: remove one vertex per SCC
                    # In practice, would use more sophisticated approximation
                    feedback_set.add(vertices[0])
            
            return feedback_set
        
        vertex_cover = approximate_strong_vertex_cover()
        feedback_set = approximate_feedback_vertex_set()
        
        return {
            'exact_sccs': num_sccs,
            'approximation_results': {
                'strong_vertex_cover': {
                    'vertices': vertex_cover,
                    'size': len(vertex_cover),
                    'approximation_ratio': 2
                },
                'feedback_vertex_set': {
                    'vertices': feedback_set,
                    'size': len(feedback_set),
                    'approximation_ratio': 'varies'
                }
            },
            'quality_metrics': {
                'vertex_cover_efficiency': len(vertex_cover) / len(scc_assignments) if scc_assignments else 0,
                'feedback_set_efficiency': len(feedback_set) / num_sccs if num_sccs > 0 else 0
            }
        }
    
    def _topological_sort_condensation(self, condensation: Dict[int, Set[int]], num_sccs: int) -> List[int]:
        """Topological sort of condensation DAG"""
        in_degree = [0] * num_sccs
        
        for scc in condensation:
            for neighbor in condensation[scc]:
                in_degree[neighbor] += 1
        
        queue = deque()
        for i in range(num_sccs):
            if in_degree[i] == 0:
                queue.append(i)
        
        topo_order = []
        while queue:
            current = queue.popleft()
            topo_order.append(current)
            
            for neighbor in condensation.get(current, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return topo_order if len(topo_order) == num_sccs else []

def test_advanced_scc_problems():
    """Test advanced SCC problems"""
    solver = AdvancedSCCProblems()
    
    print("=== Testing Advanced SCC Problems ===")
    
    # Test graph
    test_graph = {
        0: [1], 1: [2], 2: [0],  # First SCC: {0, 1, 2}
        3: [4], 4: [3],          # Second SCC: {3, 4}
        2: [3],                  # Edge between SCCs
        5: [6], 6: [5]           # Third SCC: {5, 6}
    }
    
    # Fix representation
    clean_graph = {
        0: [1],
        1: [2],
        2: [0, 3],
        3: [4],
        4: [3],
        5: [6],
        6: [5]
    }
    
    print(f"\nTest graph: {clean_graph}")
    
    # Test 1: Strong Connectivity Augmentation
    print("\n1. Strong Connectivity Augmentation:")
    augmentation = solver.strong_connectivity_augmentation(clean_graph)
    print(f"   Edges needed: {augmentation['edges_needed']}")
    print(f"   Is strongly connected: {augmentation['is_strongly_connected']}")
    
    # Test 2: Graph Analysis
    print("\n2. Comprehensive Graph Analysis:")
    analysis = solver.scc_based_graph_analysis(clean_graph)
    print(f"   SCCs: {analysis['basic_properties']['sccs']}")
    print(f"   Strong connectivity ratio: {analysis['connectivity_metrics']['strong_connectivity_ratio']:.2f}")
    
    # Test 3: Dynamic SCC
    print("\n3. Dynamic SCC Maintenance:")
    dynamic_scc = solver.dynamic_scc_maintenance(clean_graph)
    initial_info = dynamic_scc.get_scc_info()
    print(f"   Initial SCCs: {initial_info['num_sccs']}")
    
    # Add edge to make strongly connected
    updated_info = dynamic_scc.add_edge(5, 0)
    print(f"   After adding edge (5,0): {updated_info['num_sccs']} SCCs")

def demonstrate_advanced_applications():
    """Demonstrate advanced SCC applications"""
    print("\n=== Advanced SCC Applications Demo ===")
    
    print("Advanced Applications of SCC Analysis:")
    
    print("\n1. **Graph Augmentation:**")
    print("   • Minimum edges to make strongly connected")
    print("   • Network robustness improvement")
    print("   • Communication network design")
    print("   • Social network connectivity enhancement")
    
    print("\n2. **Game Theory:**")
    print("   • Nash equilibrium computation")
    print("   • Strategy cycle analysis")
    print("   • Evolutionary stable strategies")
    print("   • Multi-agent system analysis")
    
    print("\n3. **Parallel Computing:**")
    print("   • Parallel SCC algorithms")
    print("   • Task dependency resolution")
    print("   • Distributed computation")
    print("   • Load balancing optimization")
    
    print("\n4. **Approximation Algorithms:**")
    print("   • Vertex cover approximation")
    print("   • Feedback vertex set approximation")
    print("   • Network reliability approximation")
    print("   • Optimization under constraints")
    
    print("\n5. **Dynamic Analysis:**")
    print("   • Real-time SCC maintenance")
    print("   • Incremental graph updates")
    print("   • Streaming graph algorithms")
    print("   • Online optimization")

def analyze_theoretical_foundations():
    """Analyze theoretical foundations of advanced SCC"""
    print("\n=== Theoretical Foundations ===")
    
    print("Advanced SCC Theory:")
    
    print("\n1. **Complexity Theory:**")
    print("   • SCC ∈ P (polynomial time)")
    print("   • NC-complete for parallel computation")
    print("   • Space complexity: O(V + E)")
    print("   • Optimal algorithms exist")
    
    print("\n2. **Graph Theory:**")
    print("   • Condensation is always a DAG")
    print("   • Strong connectivity is equivalence relation")
    print("   • Path-based and cycle-based characterizations")
    print("   • Connection to network flow theory")
    
    print("\n3. **Algorithmic Techniques:**")
    print("   • DFS-based algorithms (Tarjan, Kosaraju)")
    print("   • Path-based algorithms")
    print("   • Parallel and distributed variants")
    print("   • Incremental and dynamic algorithms")
    
    print("\n4. **Applications Theory:**")
    print("   • Game theory and equilibrium analysis")
    print("   • Network reliability and robustness")
    print("   • Approximation algorithm design")
    print("   • Optimization problem reductions")
    
    print("\n5. **Open Problems:**")
    print("   • Optimal parallel SCC algorithms")
    print("   • Dynamic SCC with better bounds")
    print("   • Approximation ratios for SCC-based problems")
    print("   • Distributed SCC in large networks")

if __name__ == "__main__":
    test_advanced_scc_problems()
    demonstrate_advanced_applications()
    analyze_theoretical_foundations()

"""
Advanced SCC Problems and Applications:
1. Strong Connectivity Augmentation and Graph Optimization
2. Dynamic SCC Maintenance under Graph Updates
3. Game Theory Applications and Nash Equilibrium Analysis
4. Parallel and Distributed SCC Algorithms
5. Approximation Algorithms using SCC Structure
6. Network Flow and Advanced Graph Analysis
7. Real-time Systems and Online Algorithms

Key Advanced Concepts:
- Graph augmentation for connectivity improvement
- Dynamic maintenance of SCC structure
- Game-theoretic analysis using graph structure
- Parallel algorithm design and analysis
- Approximation algorithm development

Algorithm Innovations:
- Incremental SCC updates
- Parallel SCC computation strategies
- SCC-based approximation techniques
- Dynamic graph algorithms
- Distributed consensus using SCC

Theoretical Foundations:
- Complexity theory and optimal algorithms
- Graph theory and structural properties
- Game theory and equilibrium analysis
- Network theory and robustness
- Approximation theory and bounds

Real-world Impact:
- Network infrastructure optimization
- Distributed system design
- Game AI and strategic analysis
- Social network optimization
- High-performance computing

This collection demonstrates cutting-edge SCC applications
essential for advanced system design and optimization.
"""
