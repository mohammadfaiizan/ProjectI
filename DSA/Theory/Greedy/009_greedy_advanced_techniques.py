"""
Advanced Greedy Techniques and Proof Methods
============================================

Topics: Proof techniques, approximation algorithms, advanced applications
Companies: Google, Amazon, Microsoft, Facebook, Research positions
Difficulty: Hard to Expert
Time Complexity: Varies by technique and application
Space Complexity: Problem-dependent, often O(n) or O(log n)
"""

from typing import List, Tuple, Optional, Dict, Any, Set, Callable
import math
import heapq
from collections import defaultdict, deque
from abc import ABC, abstractmethod

class AdvancedGreedyTechniques:
    
    def __init__(self):
        """Initialize with advanced technique tracking"""
        self.proof_steps = []
        self.algorithm_analysis = {}
    
    # ==========================================
    # 1. FORMAL PROOF TECHNIQUES
    # ==========================================
    
    def exchange_argument_demonstration(self) -> None:
        """
        Comprehensive Exchange Argument Proof Technique
        
        The most important proof method for greedy algorithms
        Shows how to prove optimality using exchange arguments
        """
        print("=== EXCHANGE ARGUMENT PROOF TECHNIQUE ===")
        print("The fundamental method for proving greedy algorithm correctness")
        print()
        
        print("EXCHANGE ARGUMENT TEMPLATE:")
        print("1. Assume optimal solution O exists that differs from greedy solution G")
        print("2. Find first position where O and G differ")
        print("3. Show we can 'exchange' O's choice with G's choice")
        print("4. Prove the exchange doesn't worsen the solution")
        print("5. Apply recursively to show G is optimal")
        print()
        
        self._prove_activity_selection_exchange()
        print()
        self._prove_fractional_knapsack_exchange()
    
    def _prove_activity_selection_exchange(self) -> None:
        """Detailed exchange argument for activity selection"""
        print("EXAMPLE: Activity Selection Exchange Argument")
        print()
        
        print("Problem: Select maximum number of non-overlapping activities")
        print("Greedy Choice: Always select activity that ends earliest")
        print()
        
        print("PROOF BY EXCHANGE ARGUMENT:")
        print()
        print("1. SETUP:")
        print("   Let O = {o₁, o₂, ..., oₖ} be an optimal solution")
        print("   Let G = {g₁, g₂, ..., gₘ} be greedy solution")
        print("   Assume both are sorted by end time")
        print()
        
        print("2. BASE CASE:")
        print("   Greedy chooses g₁ = activity with earliest end time")
        print("   If o₁ = g₁, we're done with first choice")
        print("   If o₁ ≠ g₁, then end(g₁) ≤ end(o₁) by greedy choice")
        print()
        
        print("3. EXCHANGE STEP:")
        print("   Replace o₁ with g₁ in O to get O' = {g₁, o₂, ..., oₖ}")
        print("   Claim: O' is still feasible and optimal")
        print()
        
        print("4. FEASIBILITY:")
        print("   Since end(g₁) ≤ end(o₁) and o₁ didn't overlap with o₂")
        print("   Therefore g₁ won't overlap with o₂ either")
        print("   So O' remains feasible")
        print()
        
        print("5. OPTIMALITY:")
        print("   |O'| = |O| (same number of activities)")
        print("   Since O was optimal, O' is also optimal")
        print()
        
        print("6. INDUCTION:")
        print("   Apply same argument to remaining activities")
        print("   Eventually show greedy solution is optimal")
        print()
        
        print("✓ CONCLUSION: Greedy choice is always safe")
    
    def _prove_fractional_knapsack_exchange(self) -> None:
        """Exchange argument for fractional knapsack"""
        print("EXAMPLE: Fractional Knapsack Exchange Argument")
        print()
        
        print("Problem: Maximize value in knapsack (fractions allowed)")
        print("Greedy Choice: Take items in order of value/weight ratio")
        print()
        
        print("PROOF BY EXCHANGE ARGUMENT:")
        print()
        print("1. SETUP:")
        print("   Let items be sorted by value/weight ratio: r₁ ≥ r₂ ≥ ... ≥ rₙ")
        print("   Greedy takes items in this order")
        print("   Let O be any optimal solution")
        print()
        
        print("2. EXCHANGE PRINCIPLE:")
        print("   If O takes less of item i and more of item j where i < j:")
        print("   Since rᵢ ≥ rⱼ, we can exchange some amount from j to i")
        print("   This increases or maintains total value")
        print()
        
        print("3. MATHEMATICAL PROOF:")
        print("   Exchange δ weight from item j to item i:")
        print("   Value change = δ × rᵢ - δ × rⱼ = δ(rᵢ - rⱼ) ≥ 0")
        print("   Therefore exchange doesn't decrease value")
        print()
        
        print("4. CONCLUSION:")
        print("   Any optimal solution can be transformed to greedy solution")
        print("   Without decreasing value")
        print("   Therefore greedy is optimal")
    
    def greedy_stays_ahead_proof(self) -> None:
        """
        Greedy Stays Ahead Proof Technique
        
        Alternative proof method showing greedy maintains advantage
        """
        print("=== GREEDY STAYS AHEAD PROOF TECHNIQUE ===")
        print("Shows greedy algorithm maintains advantage at each step")
        print()
        
        print("STAYS AHEAD TEMPLATE:")
        print("1. Define measure of 'ahead' (progress metric)")
        print("2. Show greedy makes progress at least as good as any algorithm")
        print("3. Use induction to prove greedy stays ahead")
        print("4. Conclude greedy achieves optimal result")
        print()
        
        self._demonstrate_interval_scheduling_stays_ahead()
    
    def _demonstrate_interval_scheduling_stays_ahead(self) -> None:
        """Stays ahead proof for interval scheduling"""
        print("EXAMPLE: Interval Scheduling Stays Ahead")
        print()
        
        print("Problem: Schedule maximum number of non-overlapping intervals")
        print("Greedy: Select interval with earliest end time")
        print()
        
        print("STAYS AHEAD PROOF:")
        print()
        print("1. PROGRESS MEASURE:")
        print("   After k steps, measure = end time of last selected interval")
        print("   Lower end time = more room for future intervals = 'ahead'")
        print()
        
        print("2. INDUCTIVE HYPOTHESIS:")
        print("   After k intervals, greedy's last end time ≤ any algorithm's")
        print()
        
        print("3. INDUCTIVE STEP:")
        print("   Greedy chooses interval with earliest end time among remaining")
        print("   Any other algorithm must choose interval ending no earlier")
        print("   Therefore greedy maintains advantage")
        print()
        
        print("4. CONCLUSION:")
        print("   Greedy always has ≥ room for future intervals as any algorithm")
        print("   When no more intervals can be added, greedy has found maximum")
    
    # ==========================================
    # 2. APPROXIMATION ALGORITHMS
    # ==========================================
    
    def vertex_cover_approximation(self, edges: List[Tuple[str, str]]) -> Tuple[Set[str], float]:
        """
        2-Approximation Algorithm for Vertex Cover
        
        Company: Research positions, advanced algorithm roles
        Difficulty: Hard
        Time: O(E), Space: O(V)
        
        Problem: Find minimum vertex cover (NP-hard)
        Greedy Strategy: Pick endpoints of arbitrary edge, remove all incident edges
        Approximation Ratio: 2 (within factor of 2 of optimal)
        """
        print("=== VERTEX COVER 2-APPROXIMATION ===")
        print("Problem: Find minimum set of vertices covering all edges")
        print("Algorithm: Greedy edge-based selection")
        print("Approximation Ratio: 2 (guaranteed within 2× of optimal)")
        print()
        
        print(f"Input edges: {edges}")
        
        vertex_cover = set()
        uncovered_edges = set(edges)
        
        print("\nGreedy approximation process:")
        step = 1
        
        while uncovered_edges:
            # Pick any uncovered edge
            edge = next(iter(uncovered_edges))
            u, v = edge
            
            print(f"Step {step}: Select edge ({u}, {v})")
            
            # Add both endpoints to vertex cover
            vertex_cover.add(u)
            vertex_cover.add(v)
            
            print(f"   Add vertices {u} and {v} to cover")
            
            # Remove all edges incident to u or v
            edges_to_remove = set()
            for e in uncovered_edges:
                if u in e or v in e:
                    edges_to_remove.add(e)
            
            uncovered_edges -= edges_to_remove
            
            print(f"   Removed {len(edges_to_remove)} incident edges")
            print(f"   Remaining edges: {len(uncovered_edges)}")
            print(f"   Current cover: {sorted(vertex_cover)}")
            print()
            
            step += 1
        
        approximation_ratio = 2.0  # Theoretical guarantee
        
        print(f"Final vertex cover: {sorted(vertex_cover)}")
        print(f"Cover size: {len(vertex_cover)}")
        print(f"Approximation ratio: {approximation_ratio}")
        
        # Explanation of why it's 2-approximation
        print("\nWhy this is a 2-approximation:")
        print("1. Let OPT be optimal vertex cover size")
        print("2. Our algorithm selects disjoint edges")
        print("3. Each edge requires at least 1 vertex in any cover")
        print("4. We select 2 vertices per edge")
        print("5. Therefore: |our_cover| ≤ 2 × |edges_selected| ≤ 2 × OPT")
        
        return vertex_cover, approximation_ratio
    
    def set_cover_greedy_approximation(self, universe: Set[str], sets: Dict[str, Set[str]]) -> Tuple[List[str], float]:
        """
        Greedy Approximation for Set Cover
        
        Company: Research, advanced algorithms
        Difficulty: Hard
        Time: O(|U| × |S|), Space: O(|U|)
        
        Problem: Find minimum collection of sets covering universe
        Greedy Strategy: Always pick set covering most uncovered elements
        Approximation Ratio: ln(n) + 1 where n = |universe|
        """
        print("=== SET COVER GREEDY APPROXIMATION ===")
        print("Problem: Find minimum sets to cover all elements")
        print("Greedy: Always choose set covering most uncovered elements")
        print(f"Approximation Ratio: ln({len(universe)}) + 1 ≈ {math.log(len(universe)) + 1:.2f}")
        print()
        
        print(f"Universe: {sorted(universe)}")
        print("Available sets:")
        for set_name, elements in sets.items():
            print(f"   {set_name}: {sorted(elements)}")
        print()
        
        selected_sets = []
        covered = set()
        uncovered = universe.copy()
        
        print("Greedy selection process:")
        step = 1
        
        while uncovered:
            # Find set that covers most uncovered elements
            best_set = None
            max_new_coverage = 0
            
            print(f"Step {step}: {len(uncovered)} elements remain uncovered")
            print(f"   Uncovered: {sorted(uncovered)}")
            
            for set_name, elements in sets.items():
                if set_name not in selected_sets:
                    new_coverage = len(elements & uncovered)
                    print(f"   {set_name}: covers {new_coverage} new elements")
                    
                    if new_coverage > max_new_coverage:
                        max_new_coverage = new_coverage
                        best_set = set_name
            
            if best_set and max_new_coverage > 0:
                selected_sets.append(best_set)
                newly_covered = sets[best_set] & uncovered
                covered |= newly_covered
                uncovered -= newly_covered
                
                print(f"   ✓ Selected {best_set}")
                print(f"   Newly covered: {sorted(newly_covered)}")
                print(f"   Progress: {len(covered)}/{len(universe)} elements covered")
            else:
                print("   ✗ No set covers new elements - impossible to cover universe")
                break
            
            print()
            step += 1
        
        approximation_ratio = math.log(len(universe)) + 1
        
        print(f"Final solution:")
        print(f"   Selected sets: {selected_sets}")
        print(f"   Total sets used: {len(selected_sets)}")
        print(f"   Elements covered: {len(covered)}/{len(universe)}")
        print(f"   Theoretical approximation ratio: {approximation_ratio:.2f}")
        
        return selected_sets, approximation_ratio
    
    # ==========================================
    # 3. MATROID THEORY APPLICATIONS
    # ==========================================
    
    def matroid_greedy_framework(self) -> None:
        """
        Matroid Theory and Greedy Algorithms
        
        Advanced mathematical framework explaining when greedy works
        Foundation for understanding greedy algorithm optimality
        """
        print("=== MATROID THEORY AND GREEDY ALGORITHMS ===")
        print("Mathematical framework explaining when greedy algorithms work")
        print()
        
        print("MATROID DEFINITION:")
        print("A matroid M = (S, I) consists of:")
        print("• S: Finite ground set")
        print("• I: Family of independent sets with properties:")
        print("  1. ∅ ∈ I (empty set is independent)")
        print("  2. If A ∈ I and B ⊆ A, then B ∈ I (hereditary property)")
        print("  3. If A, B ∈ I and |A| < |B|, then ∃x ∈ B\\A: A ∪ {x} ∈ I")
        print("     (augmentation property)")
        print()
        
        print("GREEDY ALGORITHM FOR MATROIDS:")
        print("1. Sort elements by weight (descending)")
        print("2. For each element in sorted order:")
        print("   If adding element maintains independence, add it")
        print("3. Return the independent set")
        print()
        
        print("FUNDAMENTAL THEOREM:")
        print("Greedy algorithm finds maximum weight independent set")
        print("if and only if the structure is a matroid")
        print()
        
        self._demonstrate_graphic_matroid()
        print()
        self._demonstrate_uniform_matroid()
    
    def _demonstrate_graphic_matroid(self) -> None:
        """Example: Graphic matroid (forests in graphs)"""
        print("EXAMPLE 1: Graphic Matroid (Minimum Spanning Tree)")
        print()
        print("• Ground set S: All edges in graph")
        print("• Independent sets I: All forests (acyclic edge sets)")
        print("• Weight function: Edge weights")
        print("• Greedy algorithm: Kruskal's MST algorithm")
        print()
        print("Why it's a matroid:")
        print("1. Empty set (no edges) is acyclic ✓")
        print("2. Subset of forest is forest ✓") 
        print("3. If forest A has fewer edges than forest B,")
        print("   can add edge from B to A without creating cycle ✓")
        print()
        print("Result: Kruskal's algorithm finds MST optimally")
    
    def _demonstrate_uniform_matroid(self) -> None:
        """Example: Uniform matroid (bounded size sets)"""
        print("EXAMPLE 2: Uniform Matroid (Fractional Knapsack)")
        print()
        print("• Ground set S: All items")
        print("• Independent sets I: All sets of size ≤ k")
        print("• Weight function: Value/weight ratios")
        print("• Greedy algorithm: Take highest ratio items")
        print()
        print("Why it's a matroid:")
        print("1. Empty set has size 0 ≤ k ✓")
        print("2. Subset of size-k set has size ≤ k ✓")
        print("3. If |A| < |B| ≤ k, can add element from B to A ✓")
        print()
        print("Result: Greedy gives optimal fractional knapsack solution")
    
    # ==========================================
    # 4. COMPETITIVE ANALYSIS
    # ==========================================
    
    def online_algorithm_analysis(self) -> None:
        """
        Competitive Analysis for Online Algorithms
        
        Framework for analyzing algorithms that make decisions without
        complete information about future inputs
        """
        print("=== COMPETITIVE ANALYSIS FOR ONLINE ALGORITHMS ===")
        print("Analyzing algorithms that make irrevocable decisions")
        print("without knowing future inputs")
        print()
        
        print("COMPETITIVE RATIO:")
        print("An online algorithm ALG is c-competitive if:")
        print("ALG(σ) ≤ c × OPT(σ) + α")
        print("for all input sequences σ, where:")
        print("• ALG(σ) = cost of online algorithm")
        print("• OPT(σ) = cost of optimal offline algorithm")
        print("• c = competitive ratio")
        print("• α = additive constant")
        print()
        
        self._analyze_ski_rental_problem()
        print()
        self._analyze_paging_problem()
    
    def _analyze_ski_rental_problem(self) -> None:
        """Classic example of competitive analysis"""
        print("EXAMPLE: Ski Rental Problem")
        print()
        print("Problem: Rent skis ($1/day) or buy ($B)?")
        print("Don't know how many days you'll ski")
        print()
        
        print("Online Strategies:")
        print("1. Always rent: competitive ratio = ∞")
        print("2. Buy immediately: competitive ratio = B")
        print("3. Rent for k days, then buy:")
        print()
        
        B = 10  # Cost to buy
        print(f"Analysis with B = ${B}:")
        print()
        
        for k in [1, B-1, B, B+1]:
            print(f"Strategy: Rent {k} days, then buy")
            
            # Case 1: Ski for ≤ k days
            print(f"   If ski ≤ {k} days:")
            print(f"     Online cost: min({k}, days)")
            print(f"     Offline cost: min({B}, days)")
            print(f"     Ratio: ≤ {k}")
            
            # Case 2: Ski for > k days  
            print(f"   If ski > {k} days:")
            print(f"     Online cost: {k} + {B} = {k + B}")
            print(f"     Offline cost: {B}")
            print(f"     Ratio: {(k + B) / B:.2f}")
            
            competitive_ratio = max(k, (k + B) / B)
            print(f"   Competitive ratio: {competitive_ratio:.2f}")
            print()
        
        print(f"Optimal strategy: Rent {B-1} days, then buy")
        print(f"Achieves 2-competitive ratio")
    
    def _analyze_paging_problem(self) -> None:
        """Paging problem competitive analysis"""
        print("EXAMPLE: Paging Problem")
        print()
        print("Problem: Cache of size k, page requests")
        print("On cache miss, which page to evict?")
        print()
        
        print("Common Online Algorithms:")
        print("• FIFO (First In, First Out)")
        print("• LRU (Least Recently Used)")  
        print("• LFU (Least Frequently Used)")
        print()
        
        print("Competitive Analysis Results:")
        print("• Any deterministic online algorithm: ≥ k-competitive")
        print("• LRU is k-competitive (optimal for deterministic)")
        print("• Randomized algorithms can achieve O(log k)-competitive")
        print()
        
        print("Lower Bound Proof Sketch:")
        print("1. Adversary maintains k+1 pages")
        print("2. Always requests page not in online algorithm's cache")
        print("3. Online algorithm has k misses on k+1 requests")
        print("4. Offline algorithm has 1 miss (loads all k+1 pages initially)")
        print("5. Ratio = k/1 = k")
    
    # ==========================================
    # 5. ADVANCED APPLICATIONS
    # ==========================================
    
    def submodular_optimization(self, ground_set: Set[str], utility_function: Callable) -> Tuple[Set[str], str]:
        """
        Greedy Algorithm for Submodular Function Maximization
        
        Company: Machine learning, optimization research
        Difficulty: Expert
        Time: O(n²), Space: O(n)
        
        Problem: Maximize submodular function subject to cardinality constraint
        Greedy Strategy: Always add element with maximum marginal gain
        Approximation: (1 - 1/e) ≈ 0.632 of optimal
        """
        print("=== SUBMODULAR FUNCTION MAXIMIZATION ===")
        print("Problem: Maximize f(S) subject to |S| ≤ k")
        print("Requirement: f must be submodular (diminishing returns)")
        print("Greedy gives (1 - 1/e)-approximation ≈ 63.2% of optimal")
        print()
        
        # Example: Influence maximization in social networks
        # Submodular because adding a node has diminishing influence
        
        def influence_function(selected_nodes: Set[str]) -> float:
            """
            Mock influence function (submodular)
            In practice, this would model information spread
            """
            if not selected_nodes:
                return 0
            
            # Simple submodular function: square root of coverage
            # with diminishing returns for overlapping influence
            base_influence = len(selected_nodes) * 10
            overlap_penalty = len(selected_nodes) * (len(selected_nodes) - 1) * 0.5
            return math.sqrt(max(1, base_influence - overlap_penalty))
        
        k = 3  # Budget constraint
        selected = set()
        
        print(f"Ground set: {sorted(ground_set)}")
        print(f"Budget constraint k: {k}")
        print("Utility function: influence_function (submodular)")
        print()
        
        print("Greedy selection process:")
        
        for step in range(k):
            if len(selected) >= len(ground_set):
                break
                
            best_element = None
            max_marginal_gain = -1
            
            print(f"Step {step + 1}:")
            print(f"   Current selection: {sorted(selected) if selected else 'empty'}")
            print(f"   Current utility: {utility_function(selected):.2f}")
            
            # Find element with maximum marginal gain
            for element in ground_set:
                if element not in selected:
                    current_utility = utility_function(selected)
                    new_utility = utility_function(selected | {element})
                    marginal_gain = new_utility - current_utility
                    
                    print(f"     {element}: marginal gain = {marginal_gain:.2f}")
                    
                    if marginal_gain > max_marginal_gain:
                        max_marginal_gain = marginal_gain
                        best_element = element
            
            if best_element:
                selected.add(best_element)
                print(f"   ✓ Selected {best_element} (gain: {max_marginal_gain:.2f})")
            else:
                print(f"   No beneficial elements remaining")
                break
            
            print()
        
        final_utility = utility_function(selected)
        theoretical_guarantee = "(1 - 1/e) × OPT ≈ 0.632 × OPT"
        
        print(f"Final solution: {sorted(selected)}")
        print(f"Final utility: {final_utility:.2f}")
        print(f"Approximation guarantee: {theoretical_guarantee}")
        
        return selected, theoretical_guarantee


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_advanced_greedy_techniques():
    """Demonstrate all advanced greedy techniques"""
    print("=== ADVANCED GREEDY TECHNIQUES DEMONSTRATION ===\n")
    
    advanced = AdvancedGreedyTechniques()
    
    # 1. Proof Techniques
    print("1. FORMAL PROOF TECHNIQUES")
    advanced.exchange_argument_demonstration()
    print("\n" + "="*60 + "\n")
    
    advanced.greedy_stays_ahead_proof()
    print("\n" + "="*60 + "\n")
    
    # 2. Approximation Algorithms
    print("2. APPROXIMATION ALGORITHMS")
    
    print("a) Vertex Cover 2-Approximation:")
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'D'), ('B', 'D')]
    advanced.vertex_cover_approximation(edges)
    print("\n" + "-"*40 + "\n")
    
    print("b) Set Cover Greedy Approximation:")
    universe = {'1', '2', '3', '4', '5'}
    sets = {
        'S1': {'1', '2', '3'},
        'S2': {'2', '4'},
        'S3': {'3', '4', '5'},
        'S4': {'1', '5'}
    }
    advanced.set_cover_greedy_approximation(universe, sets)
    print("\n" + "="*60 + "\n")
    
    # 3. Matroid Theory
    print("3. MATROID THEORY")
    advanced.matroid_greedy_framework()
    print("\n" + "="*60 + "\n")
    
    # 4. Competitive Analysis
    print("4. COMPETITIVE ANALYSIS")
    advanced.online_algorithm_analysis()
    print("\n" + "="*60 + "\n")
    
    # 5. Advanced Applications
    print("5. ADVANCED APPLICATIONS")
    
    print("Submodular Function Maximization:")
    ground_set = {'A', 'B', 'C', 'D', 'E'}
    def utility_func(s): return math.sqrt(len(s) * 10) if s else 0
    advanced.submodular_optimization(ground_set, utility_func)


if __name__ == "__main__":
    demonstrate_advanced_greedy_techniques()
    
    print("\n=== ADVANCED GREEDY MASTERY GUIDE ===")
    
    print("\n🎯 ADVANCED TECHNIQUE CATEGORIES:")
    print("• Formal Proofs: Exchange argument, stays ahead, structural induction")
    print("• Approximation: Algorithms for NP-hard problems with guarantees")
    print("• Matroid Theory: Mathematical framework for greedy optimality")
    print("• Competitive Analysis: Online algorithms and worst-case ratios")
    print("• Submodular Optimization: Diminishing returns function maximization")
    
    print("\n📊 THEORETICAL FOUNDATIONS:")
    print("• Exchange Argument: Most important proof technique for greedy")
    print("• Matroid Theory: Explains exactly when greedy works optimally")
    print("• Competitive Ratios: Measure online algorithm performance")
    print("• Approximation Ratios: Guarantee quality for NP-hard problems")
    print("• Submodularity: Captures diminishing returns property")
    
    print("\n⚡ WHEN TO USE EACH TECHNIQUE:")
    print("• Exchange Argument: Proving optimality of greedy algorithms")
    print("• Approximation: NP-hard problems needing practical solutions")
    print("• Matroid Framework: Understanding greedy algorithm design")
    print("• Competitive Analysis: Online/streaming algorithm evaluation")
    print("• Submodular Methods: Machine learning, social networks")
    
    print("\n🔬 RESEARCH APPLICATIONS:")
    print("• Machine Learning: Feature selection, active learning")
    print("• Social Networks: Influence maximization, viral marketing")
    print("• Operations Research: Facility location, resource allocation")
    print("• Computer Systems: Caching, load balancing, scheduling")
    print("• Combinatorial Optimization: Network design, covering problems")
    
    print("\n🎓 ADVANCED STUDY TOPICS:")
    print("• Randomized approximation algorithms")
    print("• Semi-definite programming relaxations")
    print("• Primal-dual algorithm design")
    print("• Hardness of approximation theory")
    print("• Online convex optimization")
    
    print("\n💡 RESEARCH DIRECTIONS:")
    print("• Improved approximation ratios for classical problems")
    print("• Online algorithms with predictions/advice")
    print("• Robust optimization under uncertainty")
    print("• Streaming algorithms for massive datasets")
    print("• Quantum algorithms for combinatorial optimization")
    
    print("\n🏆 CAREER APPLICATIONS:")
    print("• Research Scientist: Algorithm design and analysis")
    print("• ML Engineer: Feature selection and model optimization")
    print("• Systems Engineer: Resource management and scheduling")
    print("• Quantitative Analyst: Financial optimization problems")
    print("• Product Manager: Understanding algorithm trade-offs")
    
    print("\n📈 MASTERY INDICATORS:")
    print("• Can prove greedy algorithm correctness using exchange arguments")
    print("• Understands when approximation algorithms are necessary")
    print("• Recognizes matroidal structure in optimization problems")
    print("• Can analyze competitive ratios for online algorithms")
    print("• Applies submodular optimization to real-world problems")
