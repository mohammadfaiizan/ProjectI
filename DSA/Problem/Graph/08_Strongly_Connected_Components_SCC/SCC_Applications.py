"""
SCC Applications - Comprehensive Analysis and Practical Use Cases
Difficulty: Medium

This file demonstrates comprehensive applications of Strongly Connected Components
in various domains including software engineering, network analysis, and system design.

Key Applications:
1. Dependency Analysis and Circular Reference Detection
2. Software Architecture and Module Organization
3. Social Network Analysis and Community Detection
4. Compiler Optimization and Dead Code Elimination
5. Web Graph Analysis and PageRank Computation
6. Game Theory and Nash Equilibrium Analysis
7. Distributed System Design and Consensus
8. Circuit Analysis and Loop Detection
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque

class SCCApplications:
    """Comprehensive SCC applications and analysis tools"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset state for new analysis"""
        self.time = 0
        self.discovery = {}
        self.low = {}
        self.on_stack = set()
        self.stack = []
        self.scc_id = {}
        self.scc_count = 0
    
    def find_sccs_tarjan(self, graph: Dict[int, List[int]]) -> Tuple[int, Dict[int, int]]:
        """
        Find SCCs using Tarjan's algorithm
        
        Time: O(V + E)
        Space: O(V)
        """
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
            
            # Check if v is root of SCC
            if self.low[v] == self.discovery[v]:
                while True:
                    u = self.stack.pop()
                    self.on_stack.remove(u)
                    self.scc_id[u] = self.scc_count
                    if u == v:
                        break
                self.scc_count += 1
        
        # Find all vertices
        vertices = set()
        for v in graph:
            vertices.add(v)
            for u in graph[v]:
                vertices.add(u)
        
        for v in vertices:
            if v not in self.discovery:
                tarjan_dfs(v)
        
        return self.scc_count, self.scc_id.copy()
    
    def analyze_dependency_cycles(self, dependencies: Dict[str, List[str]]) -> Dict:
        """
        Application 1: Software Dependency Analysis
        
        Detect circular dependencies in software modules.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Convert string names to numeric IDs
        name_to_id = {}
        id_to_name = {}
        
        for module in dependencies:
            if module not in name_to_id:
                name_to_id[module] = len(name_to_id)
                id_to_name[len(id_to_name)] = module
            
            for dep in dependencies[module]:
                if dep not in name_to_id:
                    name_to_id[dep] = len(name_to_id)
                    id_to_name[len(id_to_name)] = dep
        
        # Build numeric graph
        numeric_graph = {}
        for module, deps in dependencies.items():
            numeric_graph[name_to_id[module]] = [name_to_id[dep] for dep in deps]
        
        # Find SCCs
        num_sccs, scc_assignments = self.find_sccs_tarjan(numeric_graph)
        
        # Analyze results
        scc_groups = defaultdict(list)
        for node_id, scc_id in scc_assignments.items():
            scc_groups[scc_id].append(id_to_name[node_id])
        
        # Identify circular dependencies
        circular_dependencies = []
        independent_modules = []
        
        for scc_id, modules in scc_groups.items():
            if len(modules) > 1:
                circular_dependencies.append(modules)
            else:
                independent_modules.extend(modules)
        
        # Build condensation DAG
        condensation = self._build_condensation_dag(numeric_graph, scc_assignments)
        
        return {
            'total_modules': len(name_to_id),
            'strongly_connected_components': num_sccs,
            'circular_dependencies': circular_dependencies,
            'independent_modules': independent_modules,
            'is_acyclic': num_sccs == len(name_to_id),
            'condensation_dag': condensation,
            'dependency_layers': self._compute_dependency_layers(condensation),
            'critical_modules': self._find_critical_modules(numeric_graph, scc_assignments)
        }
    
    def analyze_social_network_communities(self, social_graph: Dict[int, List[int]]) -> Dict:
        """
        Application 2: Social Network Community Detection
        
        Find strongly connected communities in social networks.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Find SCCs in social network
        num_sccs, scc_assignments = self.find_sccs_tarjan(social_graph)
        
        # Group users by community
        communities = defaultdict(list)
        for user, community_id in scc_assignments.items():
            communities[community_id].append(user)
        
        # Analyze community properties
        community_analysis = []
        
        for community_id, members in communities.items():
            if len(members) > 1:
                # Calculate internal connections
                internal_edges = 0
                external_edges = 0
                
                for member in members:
                    for connection in social_graph.get(member, []):
                        if connection in members:
                            internal_edges += 1
                        else:
                            external_edges += 1
                
                # Calculate community metrics
                size = len(members)
                density = internal_edges / (size * (size - 1)) if size > 1 else 0
                isolation = external_edges / max(1, len(members))
                
                community_analysis.append({
                    'community_id': community_id,
                    'members': members,
                    'size': size,
                    'internal_edges': internal_edges,
                    'external_edges': external_edges,
                    'density': density,
                    'isolation_score': isolation
                })
        
        return {
            'total_users': len(set(scc_assignments.keys())),
            'num_communities': num_sccs,
            'mutual_connection_groups': [c for c in community_analysis if c['size'] > 1],
            'isolated_users': [c['members'][0] for c in community_analysis if c['size'] == 1],
            'average_community_size': sum(c['size'] for c in community_analysis) / len(community_analysis) if community_analysis else 0,
            'network_modularity': self._calculate_modularity(communities, social_graph)
        }
    
    def analyze_compiler_optimization_opportunities(self, call_graph: Dict[str, List[str]]) -> Dict:
        """
        Application 3: Compiler Optimization Analysis
        
        Find optimization opportunities using SCC analysis.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Convert to numeric representation
        func_to_id = {func: i for i, func in enumerate(call_graph.keys())}
        id_to_func = {i: func for func, i in func_to_id.items()}
        
        numeric_graph = {}
        for func, calls in call_graph.items():
            numeric_graph[func_to_id[func]] = [func_to_id[called] for called in calls if called in func_to_id]
        
        # Find SCCs
        num_sccs, scc_assignments = self.find_sccs_tarjan(numeric_graph)
        
        # Analyze optimization opportunities
        optimization_opportunities = {
            'recursive_function_groups': [],
            'dead_code_candidates': [],
            'inlining_opportunities': [],
            'loop_optimization_targets': []
        }
        
        # Group functions by SCC
        scc_groups = defaultdict(list)
        for func_id, scc_id in scc_assignments.items():
            scc_groups[scc_id].append(id_to_func[func_id])
        
        for scc_id, functions in scc_groups.items():
            if len(functions) > 1:
                # Mutual recursion group - optimization target
                optimization_opportunities['recursive_function_groups'].append({
                    'functions': functions,
                    'optimization_type': 'tail_call_optimization',
                    'loop_conversion_candidate': True
                })
                optimization_opportunities['loop_optimization_targets'].extend(functions)
            else:
                # Single function - check for inlining
                func = functions[0]
                func_id = func_to_id[func]
                
                # Simple heuristic: functions with few calls might be inlining candidates
                call_count = len(numeric_graph.get(func_id, []))
                if call_count <= 2:
                    optimization_opportunities['inlining_opportunities'].append(func)
        
        # Find potentially dead code (unreachable SCCs)
        reachable_sccs = set()
        
        def mark_reachable(scc_id, condensation):
            if scc_id in reachable_sccs:
                return
            reachable_sccs.add(scc_id)
            for next_scc in condensation.get(scc_id, []):
                mark_reachable(next_scc, condensation)
        
        condensation = self._build_condensation_dag(numeric_graph, scc_assignments)
        
        # Mark reachable from entry points (functions with no incoming calls)
        entry_points = []
        all_called = set()
        for calls in call_graph.values():
            all_called.update(calls)
        
        for func in call_graph:
            if func not in all_called:
                entry_points.append(func)
                mark_reachable(scc_assignments[func_to_id[func]], condensation)
        
        # Find unreachable functions
        for scc_id, functions in scc_groups.items():
            if scc_id not in reachable_sccs:
                optimization_opportunities['dead_code_candidates'].extend(functions)
        
        return {
            'total_functions': len(call_graph),
            'strongly_connected_groups': num_sccs,
            'optimization_opportunities': optimization_opportunities,
            'entry_points': entry_points,
            'call_graph_depth': self._calculate_dag_depth(condensation),
            'cyclomatic_complexity_estimate': len(call_graph) - num_sccs + 1
        }
    
    def analyze_web_graph_structure(self, web_graph: Dict[str, List[str]]) -> Dict:
        """
        Application 4: Web Graph Analysis for PageRank
        
        Analyze web graph structure for PageRank computation.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Convert URLs to numeric IDs
        url_to_id = {url: i for i, url in enumerate(web_graph.keys())}
        id_to_url = {i: url for url, i in url_to_id.items()}
        
        numeric_graph = {}
        for url, links in web_graph.items():
            numeric_graph[url_to_id[url]] = [url_to_id[link] for link in links if link in url_to_id]
        
        # Find SCCs
        num_sccs, scc_assignments = self.find_sccs_tarjan(numeric_graph)
        
        # Analyze web graph structure
        scc_groups = defaultdict(list)
        for page_id, scc_id in scc_assignments.items():
            scc_groups[scc_id].append(id_to_url[page_id])
        
        # Classify page groups
        page_classification = {
            'core_web_communities': [],  # Large SCCs with mutual links
            'isolated_pages': [],        # Single-page SCCs with no incoming links
            'sink_pages': [],            # Pages with incoming but no outgoing links
            'authority_clusters': []      # SCCs with high incoming link density
        }
        
        for scc_id, pages in scc_groups.items():
            if len(pages) > 1:
                # Calculate link metrics for this SCC
                internal_links = 0
                incoming_links = 0
                
                for page in pages:
                    page_id = url_to_id[page]
                    for linked_page_id in numeric_graph.get(page_id, []):
                        if id_to_url[linked_page_id] in pages:
                            internal_links += 1
                
                # Count incoming links from outside SCC
                for other_page_id, other_scc in scc_assignments.items():
                    if other_scc != scc_id:
                        for linked_page_id in numeric_graph.get(other_page_id, []):
                            if id_to_url[linked_page_id] in pages:
                                incoming_links += 1
                
                group_info = {
                    'pages': pages,
                    'size': len(pages),
                    'internal_links': internal_links,
                    'incoming_links': incoming_links,
                    'authority_score': incoming_links / len(pages)
                }
                
                if group_info['authority_score'] > 2.0:  # High incoming link density
                    page_classification['authority_clusters'].append(group_info)
                else:
                    page_classification['core_web_communities'].append(group_info)
            else:
                page = pages[0]
                page_id = url_to_id[page]
                
                # Check if isolated or sink
                has_outgoing = len(numeric_graph.get(page_id, [])) > 0
                has_incoming = any(page_id in links for links in numeric_graph.values())
                
                if not has_incoming and not has_outgoing:
                    page_classification['isolated_pages'].append(page)
                elif has_incoming and not has_outgoing:
                    page_classification['sink_pages'].append(page)
        
        return {
            'total_pages': len(web_graph),
            'strongly_connected_groups': num_sccs,
            'page_classification': page_classification,
            'largest_community_size': max(len(pages) for pages in scc_groups.values()) if scc_groups else 0,
            'web_connectivity_ratio': (len(web_graph) - num_sccs) / len(web_graph) if web_graph else 0,
            'pagerank_convergence_estimate': self._estimate_pagerank_convergence(scc_groups)
        }
    
    def analyze_distributed_system_consensus(self, communication_graph: Dict[int, List[int]]) -> Dict:
        """
        Application 5: Distributed System Consensus Analysis
        
        Analyze communication patterns for consensus algorithms.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Find SCCs in communication network
        num_sccs, scc_assignments = self.find_sccs_tarjan(communication_graph)
        
        # Analyze consensus implications
        consensus_analysis = {
            'consensus_groups': [],
            'isolated_nodes': [],
            'communication_bottlenecks': [],
            'fault_tolerance_assessment': {}
        }
        
        # Group nodes by SCC
        scc_groups = defaultdict(list)
        for node, scc_id in scc_assignments.items():
            scc_groups[scc_id].append(node)
        
        for scc_id, nodes in scc_groups.items():
            if len(nodes) > 1:
                # This group can reach consensus among themselves
                group_info = {
                    'nodes': nodes,
                    'size': len(nodes),
                    'can_reach_consensus': True,
                    'consensus_complexity': len(nodes) * (len(nodes) - 1) // 2  # Complete graph assumption
                }
                consensus_analysis['consensus_groups'].append(group_info)
            else:
                # Isolated node - cannot participate in consensus
                consensus_analysis['isolated_nodes'].append(nodes[0])
        
        # Build condensation graph for bottleneck analysis
        condensation = self._build_condensation_dag(communication_graph, scc_assignments)
        
        # Find bottlenecks (SCCs that must be traversed for global consensus)
        bottlenecks = []
        for scc_id in condensation:
            # Count paths through this SCC
            paths_through = len(condensation.get(scc_id, []))
            if paths_through > 1:
                bottlenecks.append({
                    'scc_id': scc_id,
                    'nodes': scc_groups[scc_id],
                    'criticality': paths_through
                })
        
        consensus_analysis['communication_bottlenecks'] = bottlenecks
        
        # Fault tolerance assessment
        consensus_analysis['fault_tolerance_assessment'] = {
            'min_nodes_for_global_consensus': max(len(nodes) for nodes in scc_groups.values()) if scc_groups else 0,
            'partition_tolerance': num_sccs > 1,
            'byzantine_fault_tolerance': all(len(nodes) >= 4 for nodes in scc_groups.values() if len(nodes) > 1),
            'total_consensus_complexity': sum(len(nodes) * (len(nodes) - 1) for nodes in scc_groups.values())
        }
        
        return consensus_analysis
    
    def _build_condensation_dag(self, graph: Dict[int, List[int]], scc_assignments: Dict[int, int]) -> Dict[int, List[int]]:
        """Build condensation DAG from SCC assignments"""
        condensation = defaultdict(set)
        
        for u in graph:
            for v in graph[u]:
                scc_u = scc_assignments.get(u)
                scc_v = scc_assignments.get(v)
                
                if scc_u is not None and scc_v is not None and scc_u != scc_v:
                    condensation[scc_u].add(scc_v)
        
        return {k: list(v) for k, v in condensation.items()}
    
    def _compute_dependency_layers(self, condensation: Dict[int, List[int]]) -> List[List[int]]:
        """Compute topological layers of condensation DAG"""
        in_degree = defaultdict(int)
        all_nodes = set()
        
        for u in condensation:
            all_nodes.add(u)
            for v in condensation[u]:
                all_nodes.add(v)
                in_degree[v] += 1
        
        layers = []
        current_layer = [node for node in all_nodes if in_degree[node] == 0]
        
        while current_layer:
            layers.append(current_layer[:])
            next_layer = []
            
            for node in current_layer:
                for neighbor in condensation.get(node, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_layer.append(neighbor)
            
            current_layer = next_layer
        
        return layers
    
    def _find_critical_modules(self, graph: Dict[int, List[int]], scc_assignments: Dict[int, int]) -> List[int]:
        """Find critical modules in dependency graph"""
        # Simplified: modules that are in cycles or have high fan-out
        critical = []
        
        scc_sizes = defaultdict(int)
        for node, scc_id in scc_assignments.items():
            scc_sizes[scc_id] += 1
        
        for node, scc_id in scc_assignments.items():
            if scc_sizes[scc_id] > 1 or len(graph.get(node, [])) > 3:
                critical.append(node)
        
        return critical
    
    def _calculate_modularity(self, communities: Dict, graph: Dict[int, List[int]]) -> float:
        """Calculate network modularity score"""
        # Simplified modularity calculation
        total_edges = sum(len(neighbors) for neighbors in graph.values())
        if total_edges == 0:
            return 0
        
        modularity = 0
        for community_members in communities.values():
            if len(community_members) <= 1:
                continue
            
            internal_edges = 0
            for member in community_members:
                for neighbor in graph.get(member, []):
                    if neighbor in community_members:
                        internal_edges += 1
            
            expected = (sum(len(graph.get(member, [])) for member in community_members) ** 2) / (2 * total_edges)
            modularity += (internal_edges / total_edges) - (expected / (2 * total_edges))
        
        return modularity
    
    def _calculate_dag_depth(self, dag: Dict[int, List[int]]) -> int:
        """Calculate depth of DAG using topological sorting"""
        in_degree = defaultdict(int)
        all_nodes = set()
        
        for u in dag:
            all_nodes.add(u)
            for v in dag[u]:
                all_nodes.add(v)
                in_degree[v] += 1
        
        queue = deque([(node, 0) for node in all_nodes if in_degree[node] == 0])
        max_depth = 0
        
        while queue:
            node, depth = queue.popleft()
            max_depth = max(max_depth, depth)
            
            for neighbor in dag.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append((neighbor, depth + 1))
        
        return max_depth
    
    def _estimate_pagerank_convergence(self, scc_groups: Dict) -> str:
        """Estimate PageRank convergence based on SCC structure"""
        if not scc_groups:
            return "immediate"
        
        max_scc_size = max(len(members) for members in scc_groups.values())
        
        if max_scc_size == 1:
            return "fast"
        elif max_scc_size <= 10:
            return "moderate"
        else:
            return "slow"

def test_scc_applications():
    """Test SCC applications with example scenarios"""
    analyzer = SCCApplications()
    
    print("=== Testing SCC Applications ===")
    
    # Test 1: Dependency Analysis
    print("\n1. Software Dependency Analysis:")
    dependencies = {
        'moduleA': ['moduleB', 'moduleC'],
        'moduleB': ['moduleD'],
        'moduleC': ['moduleD'],
        'moduleD': ['moduleA'],  # Creates cycle!
        'moduleE': ['moduleF'],
        'moduleF': ['moduleE']   # Another cycle
    }
    
    dep_analysis = analyzer.analyze_dependency_cycles(dependencies)
    print(f"   Dependencies: {dependencies}")
    print(f"   Circular dependencies found: {dep_analysis['circular_dependencies']}")
    print(f"   Independent modules: {dep_analysis['independent_modules']}")
    print(f"   Is acyclic: {dep_analysis['is_acyclic']}")
    
    # Test 2: Social Network Analysis
    print("\n2. Social Network Community Detection:")
    social_network = {
        1: [2, 3],     # User 1 follows 2, 3
        2: [3],        # User 2 follows 3
        3: [1],        # User 3 follows 1 (creates mutual following)
        4: [5],        # User 4 follows 5
        5: [4],        # User 5 follows 4 (mutual)
        6: []          # User 6 follows nobody
    }
    
    social_analysis = analyzer.analyze_social_network_communities(social_network)
    print(f"   Social network: {social_network}")
    print(f"   Mutual connection groups: {[g['members'] for g in social_analysis['mutual_connection_groups']]}")
    print(f"   Isolated users: {social_analysis['isolated_users']}")
    
    # Test 3: Compiler Optimization
    print("\n3. Compiler Optimization Analysis:")
    call_graph = {
        'main': ['funcA', 'funcB'],
        'funcA': ['funcC'],
        'funcB': ['funcC'],
        'funcC': ['funcA'],  # Mutual recursion
        'funcD': ['funcE'],  # Dead code
        'funcE': []
    }
    
    compiler_analysis = analyzer.analyze_compiler_optimization_opportunities(call_graph)
    print(f"   Call graph: {call_graph}")
    print(f"   Recursive groups: {compiler_analysis['optimization_opportunities']['recursive_function_groups']}")
    print(f"   Dead code candidates: {compiler_analysis['optimization_opportunities']['dead_code_candidates']}")

def demonstrate_practical_impact():
    """Demonstrate practical impact of SCC analysis"""
    print("\n=== Practical Impact Demo ===")
    
    print("SCC Analysis Impact in Software Engineering:")
    
    print("\n1. **Build System Optimization:**")
    print("   • Detect circular dependencies early")
    print("   • Optimize compilation order")
    print("   • Reduce build times through parallelization")
    print("   • Identify refactoring opportunities")
    
    print("\n2. **Code Quality Assessment:**")
    print("   • Measure architectural complexity")
    print("   • Find tightly coupled modules")
    print("   • Guide modularization efforts")
    print("   • Improve maintainability metrics")
    
    print("\n3. **Performance Optimization:**")
    print("   • Identify hot path cycles")
    print("   • Optimize recursive call patterns")
    print("   • Enable tail call optimization")
    print("   • Reduce memory usage in loops")
    
    print("\n4. **Security Analysis:**")
    print("   • Find information flow cycles")
    print("   • Detect privilege escalation paths")
    print("   • Analyze access control dependencies")
    print("   • Validate security boundaries")

if __name__ == "__main__":
    test_scc_applications()
    demonstrate_practical_impact()

"""
SCC Applications and Practical Use Cases:
1. Software Dependency Analysis and Circular Reference Detection
2. Social Network Community Detection and Influence Analysis
3. Compiler Optimization and Dead Code Elimination
4. Web Graph Analysis and PageRank Computation
5. Distributed System Consensus and Fault Tolerance

Key Application Areas:
- Software engineering and architecture analysis
- Social network analysis and community detection
- Compiler optimization and code analysis
- Web search and information retrieval
- Distributed systems and consensus algorithms

Practical Benefits:
- Early detection of design problems
- Optimization opportunity identification
- System reliability and fault tolerance
- Performance improvement guidance
- Security vulnerability assessment

Algorithm Impact:
- O(V + E) time complexity enables large-scale analysis
- Single algorithm serves multiple application domains
- Condensation DAG provides structural insights
- Quantitative metrics for qualitative assessments

This implementation demonstrates the broad applicability
of SCC analysis across diverse engineering domains.
"""
