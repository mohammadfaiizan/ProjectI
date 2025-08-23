"""
PageRank Algorithm - Advanced System Design Implementation
Difficulty: Hard

PageRank is Google's original algorithm for ranking web pages based on their
importance and authority. This file implements various versions of PageRank
and related algorithms used in web search and network analysis.

Key Concepts:
1. Random Walk Model
2. Link Analysis and Authority
3. Damping Factor and Teleportation
4. Power Iteration Method
5. Personalized PageRank
6. Topic-Sensitive PageRank
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
import random
import math

class PageRankAlgorithm:
    """Implementation of PageRank and related algorithms"""
    
    def __init__(self, damping_factor: float = 0.85, max_iterations: int = 100, 
                 tolerance: float = 1e-6):
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.graph = defaultdict(set)
        self.nodes = set()
    
    def add_link(self, from_page: str, to_page: str):
        """Add a link from one page to another"""
        self.graph[from_page].add(to_page)
        self.nodes.add(from_page)
        self.nodes.add(to_page)
    
    def pagerank_power_iteration(self) -> Dict[str, float]:
        """
        Approach 1: Classic PageRank using Power Iteration
        
        Iteratively compute PageRank values until convergence.
        
        Time: O(k * (V + E)), Space: O(V)
        """
        if not self.nodes:
            return {}
        
        n = len(self.nodes)
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        
        # Initialize PageRank values
        pagerank = np.ones(n) / n
        
        # Build transition matrix
        transition_matrix = np.zeros((n, n))
        
        for from_node in self.nodes:
            from_idx = node_to_idx[from_node]
            outlinks = self.graph[from_node]
            
            if outlinks:
                # Distribute PageRank equally among outlinks
                for to_node in outlinks:
                    to_idx = node_to_idx[to_node]
                    transition_matrix[to_idx][from_idx] = 1.0 / len(outlinks)
            else:
                # Dangling node: distribute to all nodes
                for i in range(n):
                    transition_matrix[i][from_idx] = 1.0 / n
        
        # Power iteration
        for iteration in range(self.max_iterations):
            new_pagerank = (self.damping_factor * transition_matrix @ pagerank + 
                           (1 - self.damping_factor) / n * np.ones(n))
            
            # Check convergence
            if np.linalg.norm(new_pagerank - pagerank, 1) < self.tolerance:
                break
            
            pagerank = new_pagerank
        
        # Convert back to dictionary
        return {idx_to_node[i]: pagerank[i] for i in range(n)}
    
    def pagerank_random_walk(self, num_walks: int = 100000, walk_length: int = 100) -> Dict[str, float]:
        """
        Approach 2: PageRank via Random Walk Simulation
        
        Simulate random walks to estimate PageRank values.
        
        Time: O(num_walks * walk_length), Space: O(V)
        """
        if not self.nodes:
            return {}
        
        visit_counts = defaultdict(int)
        nodes_list = list(self.nodes)
        
        for _ in range(num_walks):
            # Start random walk from random node
            current_node = random.choice(nodes_list)
            
            for _ in range(walk_length):
                visit_counts[current_node] += 1
                
                # With probability (1 - damping_factor), teleport to random node
                if random.random() < (1 - self.damping_factor):
                    current_node = random.choice(nodes_list)
                else:
                    # Follow outlink or teleport if no outlinks
                    outlinks = list(self.graph[current_node])
                    if outlinks:
                        current_node = random.choice(outlinks)
                    else:
                        current_node = random.choice(nodes_list)
        
        # Normalize to get probabilities
        total_visits = sum(visit_counts.values())
        return {node: count / total_visits for node, count in visit_counts.items()}
    
    def personalized_pagerank(self, personalization_vector: Dict[str, float]) -> Dict[str, float]:
        """
        Approach 3: Personalized PageRank
        
        Compute PageRank with personalized teleportation preferences.
        
        Time: O(k * (V + E)), Space: O(V)
        """
        if not self.nodes:
            return {}
        
        n = len(self.nodes)
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        
        # Initialize PageRank values
        pagerank = np.ones(n) / n
        
        # Create personalization vector
        personalization = np.zeros(n)
        total_personalization = sum(personalization_vector.values())
        
        for node, weight in personalization_vector.items():
            if node in node_to_idx:
                personalization[node_to_idx[node]] = weight / total_personalization
        
        # If no personalization provided, use uniform
        if total_personalization == 0:
            personalization = np.ones(n) / n
        
        # Build transition matrix
        transition_matrix = np.zeros((n, n))
        
        for from_node in self.nodes:
            from_idx = node_to_idx[from_node]
            outlinks = self.graph[from_node]
            
            if outlinks:
                for to_node in outlinks:
                    to_idx = node_to_idx[to_node]
                    transition_matrix[to_idx][from_idx] = 1.0 / len(outlinks)
            else:
                # Dangling node: use personalization vector
                for i in range(n):
                    transition_matrix[i][from_idx] = personalization[i]
        
        # Power iteration with personalization
        for iteration in range(self.max_iterations):
            new_pagerank = (self.damping_factor * transition_matrix @ pagerank + 
                           (1 - self.damping_factor) * personalization)
            
            if np.linalg.norm(new_pagerank - pagerank, 1) < self.tolerance:
                break
            
            pagerank = new_pagerank
        
        return {idx_to_node[i]: pagerank[i] for i in range(n)}
    
    def topic_sensitive_pagerank(self, topics: Dict[str, Set[str]]) -> Dict[str, Dict[str, float]]:
        """
        Approach 4: Topic-Sensitive PageRank
        
        Compute PageRank for different topic categories.
        
        Time: O(T * k * (V + E)), Space: O(T * V)
        """
        topic_pageranks = {}
        
        for topic, topic_pages in topics.items():
            # Create personalization vector for this topic
            personalization = {}
            for page in topic_pages:
                if page in self.nodes:
                    personalization[page] = 1.0
            
            if personalization:
                topic_pageranks[topic] = self.personalized_pagerank(personalization)
            else:
                topic_pageranks[topic] = {}
        
        return topic_pageranks
    
    def pagerank_with_priors(self, prior_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Approach 5: PageRank with Prior Knowledge
        
        Incorporate prior knowledge about page importance.
        
        Time: O(k * (V + E)), Space: O(V)
        """
        if not self.nodes:
            return {}
        
        n = len(self.nodes)
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        
        # Initialize with prior scores
        pagerank = np.ones(n) / n
        for node, score in prior_scores.items():
            if node in node_to_idx:
                pagerank[node_to_idx[node]] = score
        
        # Normalize
        pagerank = pagerank / np.sum(pagerank)
        
        # Build transition matrix
        transition_matrix = np.zeros((n, n))
        
        for from_node in self.nodes:
            from_idx = node_to_idx[from_node]
            outlinks = self.graph[from_node]
            
            if outlinks:
                for to_node in outlinks:
                    to_idx = node_to_idx[to_node]
                    transition_matrix[to_idx][from_idx] = 1.0 / len(outlinks)
            else:
                # Use prior distribution for dangling nodes
                for i in range(n):
                    prior_weight = prior_scores.get(idx_to_node[i], 1.0)
                    transition_matrix[i][from_idx] = prior_weight / sum(prior_scores.values() or [1.0])
        
        # Power iteration
        for iteration in range(self.max_iterations):
            new_pagerank = (self.damping_factor * transition_matrix @ pagerank + 
                           (1 - self.damping_factor) / n * np.ones(n))
            
            if np.linalg.norm(new_pagerank - pagerank, 1) < self.tolerance:
                break
            
            pagerank = new_pagerank
        
        return {idx_to_node[i]: pagerank[i] for i in range(n)}
    
    def authority_hub_scores(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Approach 6: HITS Algorithm (Authorities and Hubs)
        
        Compute authority and hub scores using HITS algorithm.
        
        Time: O(k * (V + E)), Space: O(V)
        """
        if not self.nodes:
            return {}, {}
        
        n = len(self.nodes)
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        
        # Initialize scores
        authority = np.ones(n)
        hub = np.ones(n)
        
        # Build adjacency matrix
        adjacency = np.zeros((n, n))
        for from_node in self.nodes:
            from_idx = node_to_idx[from_node]
            for to_node in self.graph[from_node]:
                to_idx = node_to_idx[to_node]
                adjacency[to_idx][from_idx] = 1
        
        # HITS iteration
        for iteration in range(self.max_iterations):
            new_authority = adjacency @ hub
            new_hub = adjacency.T @ authority
            
            # Normalize
            new_authority = new_authority / np.linalg.norm(new_authority)
            new_hub = new_hub / np.linalg.norm(new_hub)
            
            # Check convergence
            if (np.linalg.norm(new_authority - authority) < self.tolerance and
                np.linalg.norm(new_hub - hub) < self.tolerance):
                break
            
            authority = new_authority
            hub = new_hub
        
        authority_dict = {idx_to_node[i]: authority[i] for i in range(n)}
        hub_dict = {idx_to_node[i]: hub[i] for i in range(n)}
        
        return authority_dict, hub_dict
    
    def get_top_pages(self, pagerank_scores: Dict[str, float], k: int = 10) -> List[Tuple[str, float]]:
        """Get top k pages by PageRank score"""
        return sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    def analyze_link_structure(self) -> Dict:
        """Analyze the link structure of the graph"""
        if not self.nodes:
            return {}
        
        total_nodes = len(self.nodes)
        total_links = sum(len(outlinks) for outlinks in self.graph.values())
        
        # Calculate in-degree and out-degree distributions
        in_degrees = defaultdict(int)
        out_degrees = defaultdict(int)
        
        for from_node in self.nodes:
            out_degrees[from_node] = len(self.graph[from_node])
            for to_node in self.graph[from_node]:
                in_degrees[to_node] += 1
        
        # Find dangling nodes (no outlinks)
        dangling_nodes = [node for node in self.nodes if len(self.graph[node]) == 0]
        
        return {
            'total_nodes': total_nodes,
            'total_links': total_links,
            'average_out_degree': total_links / total_nodes if total_nodes > 0 else 0,
            'dangling_nodes': len(dangling_nodes),
            'max_in_degree': max(in_degrees.values()) if in_degrees else 0,
            'max_out_degree': max(out_degrees.values()) if out_degrees else 0,
            'nodes_with_no_inlinks': len([node for node in self.nodes if in_degrees[node] == 0])
        }

def test_pagerank_algorithm():
    """Test PageRank algorithm implementations"""
    print("=== Testing PageRank Algorithm ===")
    
    # Create a simple web graph
    pagerank = PageRankAlgorithm(damping_factor=0.85, max_iterations=50)
    
    # Add links (representing web pages and their links)
    links = [
        ("A", "B"), ("A", "C"),
        ("B", "C"), ("B", "D"),
        ("C", "A"), ("C", "D"),
        ("D", "A"), ("D", "B"), ("D", "C")
    ]
    
    for from_page, to_page in links:
        pagerank.add_link(from_page, to_page)
    
    print("Web graph structure:")
    for from_page, to_page in links:
        print(f"  {from_page} -> {to_page}")
    
    # Test different PageRank algorithms
    print(f"\n--- Classic PageRank (Power Iteration) ---")
    classic_pr = pagerank.pagerank_power_iteration()
    for page, score in sorted(classic_pr.items(), key=lambda x: x[1], reverse=True):
        print(f"{page}: {score:.4f}")
    
    print(f"\n--- PageRank via Random Walk ---")
    random_walk_pr = pagerank.pagerank_random_walk(num_walks=10000, walk_length=50)
    for page, score in sorted(random_walk_pr.items(), key=lambda x: x[1], reverse=True):
        print(f"{page}: {score:.4f}")
    
    print(f"\n--- Personalized PageRank (biased toward A) ---")
    personalized_pr = pagerank.personalized_pagerank({"A": 1.0})
    for page, score in sorted(personalized_pr.items(), key=lambda x: x[1], reverse=True):
        print(f"{page}: {score:.4f}")
    
    print(f"\n--- HITS Algorithm (Authority and Hub Scores) ---")
    authorities, hubs = pagerank.authority_hub_scores()
    print("Authority scores:")
    for page, score in sorted(authorities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {page}: {score:.4f}")
    print("Hub scores:")
    for page, score in sorted(hubs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {page}: {score:.4f}")
    
    # Analyze link structure
    print(f"\n--- Link Structure Analysis ---")
    analysis = pagerank.analyze_link_structure()
    for key, value in analysis.items():
        print(f"{key}: {value}")

def demonstrate_pagerank_concepts():
    """Demonstrate PageRank concepts and applications"""
    print("\n=== PageRank Concepts and Applications ===")
    
    print("Core Concepts:")
    print("• Random Walk Model: PageRank as stationary distribution")
    print("• Link Analysis: Pages gain authority from incoming links")
    print("• Damping Factor: Probability of following links vs. teleporting")
    print("• Power Iteration: Iterative method to find dominant eigenvector")
    
    print("\nKey Properties:")
    print("• Convergence: Algorithm converges to unique solution")
    print("• Scale Invariant: Relative rankings preserved under scaling")
    print("• Personalization: Can bias toward specific topics or pages")
    print("• Robustness: Resistant to manipulation attempts")
    
    print("\nReal-World Applications:")
    print("• Web Search: Google's original ranking algorithm")
    print("• Social Networks: Influence and authority measurement")
    print("• Citation Analysis: Academic paper importance")
    print("• Recommendation Systems: Item and user ranking")
    print("• Network Analysis: Node importance in any graph")
    
    print("\nVariations and Extensions:")
    print("• Personalized PageRank: Topic-specific rankings")
    print("• Topic-Sensitive PageRank: Multiple topic categories")
    print("• HITS Algorithm: Separate authority and hub scores")
    print("• TrustRank: Spam-resistant ranking")
    print("• Temporal PageRank: Time-aware rankings")

def analyze_pagerank_complexity():
    """Analyze PageRank algorithm complexity"""
    print("\n=== PageRank Complexity Analysis ===")
    
    print("Algorithm Complexities:")
    
    print("\n1. **Power Iteration Method:**")
    print("   • Time: O(k * (V + E)) per iteration")
    print("   • Space: O(V) for PageRank vector")
    print("   • Convergence: Typically 50-100 iterations")
    print("   • Most commonly used in practice")
    
    print("\n2. **Random Walk Simulation:**")
    print("   • Time: O(num_walks * walk_length)")
    print("   • Space: O(V) for visit counts")
    print("   • Accuracy: Depends on number of walks")
    print("   • Good for understanding and approximation")
    
    print("\n3. **Matrix Methods:**")
    print("   • Time: O(V^3) for direct matrix inversion")
    print("   • Space: O(V^2) for transition matrix")
    print("   • Exact: No iteration needed")
    print("   • Impractical for large graphs")
    
    print("\n4. **Scalability Considerations:**")
    print("   • Web scale: Billions of pages and links")
    print("   • Distributed computation: MapReduce, Spark")
    print("   • Sparse matrices: Most pages link to few others")
    print("   • Incremental updates: Handle new pages and links")
    
    print("\nOptimization Strategies:")
    print("• Sparse matrix operations for efficiency")
    print("• Block-based computation for memory management")
    print("• Parallel processing across multiple machines")
    print("• Incremental updates for dynamic graphs")

if __name__ == "__main__":
    test_pagerank_algorithm()
    demonstrate_pagerank_concepts()
    analyze_pagerank_complexity()

"""
PageRank Algorithm - Key Insights:

1. **Mathematical Foundation:**
   - Based on random walk and Markov chain theory
   - Computes stationary distribution of web surfer
   - Eigenvector of transition matrix
   - Damping factor prevents rank sinks

2. **Algorithm Variants:**
   - Power iteration: Most practical method
   - Random walk: Intuitive simulation approach
   - Personalized: Topic-specific rankings
   - HITS: Separate authority and hub analysis

3. **System Design Considerations:**
   - Scalability: Handle billions of web pages
   - Efficiency: Sparse matrix operations
   - Updates: Incremental computation for changes
   - Distribution: Parallel processing across clusters

4. **Real-World Impact:**
   - Revolutionized web search quality
   - Foundation for modern search engines
   - Applied beyond web: social networks, citations
   - Continues to influence ranking algorithms

5. **Implementation Challenges:**
   - Convergence criteria and iteration limits
   - Handling dangling nodes and disconnected components
   - Memory efficiency for large-scale graphs
   - Robustness against spam and manipulation

PageRank demonstrates how mathematical elegance
can solve practical problems at unprecedented scale.
"""
