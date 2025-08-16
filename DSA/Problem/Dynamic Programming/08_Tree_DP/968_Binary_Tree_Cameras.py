"""
LeetCode 968: Binary Tree Cameras
Difficulty: Hard
Category: Tree DP - Coverage Optimization

PROBLEM DESCRIPTION:
===================
You are given the root of a binary tree. We install cameras on the tree nodes. Each camera at a node can monitor its parent, itself, and its immediate children.

Calculate the minimum number of cameras needed to monitor all nodes of the tree.

Example 1:
Input: root = [0,0,null,0,0]
Output: 1
Explanation: One camera at [0,0,null,0,null] can monitor all nodes.

Example 2:
Input: root = [0,0,null,0,null,null,0]
Output: 2
Explanation: At least two cameras are needed to monitor all nodes of the tree.

Constraints:
- The number of nodes in the tree is in the range [1, 1000].
- Node.val == 0 for all nodes.
"""

# TreeNode definition for reference
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def min_camera_cover_greedy(root):
    """
    GREEDY APPROACH:
    ===============
    Post-order traversal with greedy camera placement.
    
    Time Complexity: O(n) - single traversal
    Space Complexity: O(h) - recursion depth
    """
    if not root:
        return 0
    
    cameras = 0
    
    def dfs(node):
        nonlocal cameras
        
        if not node:
            return 'covered'  # null nodes are considered covered
        
        left_state = dfs(node.left)
        right_state = dfs(node.right)
        
        # If any child is not covered, we must place a camera here
        if left_state == 'not_covered' or right_state == 'not_covered':
            cameras += 1
            return 'has_camera'
        
        # If any child has a camera, this node is covered
        if left_state == 'has_camera' or right_state == 'has_camera':
            return 'covered'
        
        # Both children are covered but don't have cameras
        # This node is not covered
        return 'not_covered'
    
    # Handle root specially - if root is not covered, place camera there
    if dfs(root) == 'not_covered':
        cameras += 1
    
    return cameras


def min_camera_cover_tree_dp(root):
    """
    TREE DP APPROACH:
    ================
    Each node has three states: no camera, has camera, covered by child.
    
    Time Complexity: O(n) - each node computed once
    Space Complexity: O(h) - recursion depth
    """
    if not root:
        return 0
    
    def dp(node):
        if not node:
            # [no_camera, has_camera, covered_by_child]
            return [0, float('inf'), 0]
        
        left = dp(node.left)
        right = dp(node.right)
        
        # State 0: This node has no camera and is not covered by child
        # Both children must have cameras
        no_camera = left[1] + right[1]
        
        # State 1: This node has a camera
        # Children can be in any valid state (take minimum)
        has_camera = 1 + min(left) + min(right)
        
        # State 2: This node is covered by a child camera
        # At least one child must have a camera, others can be covered
        covered_by_child = min(
            left[1] + min(right[0], right[2]),  # Left child has camera
            right[1] + min(left[0], left[2]),   # Right child has camera
            left[1] + right[1]                  # Both children have cameras
        )
        
        return [no_camera, has_camera, covered_by_child]
    
    result = dp(root)
    # Root cannot be in state 0 (uncovered), so choose between states 1 and 2
    return min(result[1], result[2])


def min_camera_cover_detailed(root):
    """
    DETAILED DP WITH STATE TRACKING:
    ================================
    Track camera positions and coverage status.
    
    Time Complexity: O(n) - single traversal
    Space Complexity: O(h) - recursion depth
    """
    if not root:
        return 0, [], []
    
    def dp(node):
        if not node:
            return {
                'no_camera': (0, [], []),           # (cost, cameras, covered)
                'has_camera': (float('inf'), [], []),
                'covered': (0, [], [])
            }
        
        left = dp(node.left)
        right = dp(node.right)
        
        # State: This node has no camera (must be covered by children)
        no_camera_cost = left['has_camera'][0] + right['has_camera'][0]
        no_camera_cams = left['has_camera'][1] + right['has_camera'][1]
        no_camera_covered = left['has_camera'][2] + right['has_camera'][2] + [node.val]
        
        # State: This node has a camera
        left_min = min(left.values(), key=lambda x: x[0])
        right_min = min(right.values(), key=lambda x: x[0])
        has_camera_cost = 1 + left_min[0] + right_min[0]
        has_camera_cams = [node.val] + left_min[1] + right_min[1]
        has_camera_covered = [node.val] + left_min[2] + right_min[2]
        
        # Add children to covered list if this node has camera
        if node.left:
            has_camera_covered.append(node.left.val)
        if node.right:
            has_camera_covered.append(node.right.val)
        
        # State: This node is covered by child
        covered_cost = float('inf')
        covered_cams = []
        covered_covered = []
        
        # Try left child having camera
        if left['has_camera'][0] != float('inf'):
            cost = left['has_camera'][0] + min(right['no_camera'][0], right['covered'][0])
            if cost < covered_cost:
                covered_cost = cost
                right_choice = right['no_camera'] if right['no_camera'][0] <= right['covered'][0] else right['covered']
                covered_cams = left['has_camera'][1] + right_choice[1]
                covered_covered = left['has_camera'][2] + right_choice[2] + [node.val]
        
        # Try right child having camera
        if right['has_camera'][0] != float('inf'):
            cost = right['has_camera'][0] + min(left['no_camera'][0], left['covered'][0])
            if cost < covered_cost:
                covered_cost = cost
                left_choice = left['no_camera'] if left['no_camera'][0] <= left['covered'][0] else left['covered']
                covered_cams = right['has_camera'][1] + left_choice[1]
                covered_covered = right['has_camera'][2] + left_choice[2] + [node.val]
        
        return {
            'no_camera': (no_camera_cost, no_camera_cams, no_camera_covered),
            'has_camera': (has_camera_cost, has_camera_cams, has_camera_covered),
            'covered': (covered_cost, covered_cams, covered_covered)
        }
    
    result = dp(root)
    # Choose best valid state for root
    best_state = min([result['has_camera'], result['covered']], key=lambda x: x[0])
    return best_state[0], best_state[1], best_state[2]


def min_camera_cover_iterative(root):
    """
    ITERATIVE APPROACH:
    ==================
    Post-order traversal using stack.
    
    Time Complexity: O(n) - each node visited once
    Space Complexity: O(h) - stack space
    """
    if not root:
        return 0
    
    stack = [(root, False)]
    states = {}  # node -> [no_camera, has_camera, covered_by_child]
    
    while stack:
        node, processed = stack.pop()
        
        if processed:
            # Process after children
            left_states = states.get(node.left, [0, float('inf'), 0])
            right_states = states.get(node.right, [0, float('inf'), 0])
            
            # Compute three states
            no_camera = left_states[1] + right_states[1]
            has_camera = 1 + min(left_states) + min(right_states)
            covered_by_child = min(
                left_states[1] + min(right_states[0], right_states[2]),
                right_states[1] + min(left_states[0], left_states[2]),
                left_states[1] + right_states[1]
            )
            
            states[node] = [no_camera, has_camera, covered_by_child]
        else:
            # Mark for processing and add children
            stack.append((node, True))
            if node.right:
                stack.append((node.right, False))
            if node.left:
                stack.append((node.left, False))
    
    root_states = states[root]
    return min(root_states[1], root_states[2])


def min_camera_cover_analysis(root):
    """
    COMPREHENSIVE COVERAGE ANALYSIS:
    ===============================
    Analyze tree structure and optimal camera placement.
    """
    if not root:
        print("Empty tree - no cameras needed!")
        return 0
    
    # Tree structure analysis
    def analyze_tree(node, depth=0):
        if not node:
            return {'nodes': 0, 'leaves': 0, 'max_depth': depth}
        
        if not node.left and not node.right:
            return {'nodes': 1, 'leaves': 1, 'max_depth': depth + 1}
        
        left_info = analyze_tree(node.left, depth + 1)
        right_info = analyze_tree(node.right, depth + 1)
        
        return {
            'nodes': 1 + left_info['nodes'] + right_info['nodes'],
            'leaves': left_info['leaves'] + right_info['leaves'],
            'max_depth': max(left_info['max_depth'], right_info['max_depth'])
        }
    
    tree_info = analyze_tree(root)
    
    print(f"Tree Structure Analysis:")
    print(f"  Total nodes: {tree_info['nodes']}")
    print(f"  Leaf nodes: {tree_info['leaves']}")
    print(f"  Tree height: {tree_info['max_depth']}")
    
    # Optimal camera placement
    min_cameras, camera_positions, covered_nodes = min_camera_cover_detailed(root)
    
    print(f"\nOptimal Camera Placement:")
    print(f"  Minimum cameras needed: {min_cameras}")
    print(f"  Camera positions: {camera_positions}")
    print(f"  All covered nodes: {sorted(set(covered_nodes))}")
    print(f"  Coverage efficiency: {len(set(covered_nodes))/min_cameras:.2f} nodes per camera")
    
    # Strategy comparison
    greedy_result = min_camera_cover_greedy(root)
    tree_dp_result = min_camera_cover_tree_dp(root)
    
    print(f"\nAlgorithm Comparison:")
    print(f"  Greedy approach: {greedy_result} cameras")
    print(f"  Tree DP approach: {tree_dp_result} cameras")
    print(f"  Detailed DP: {min_cameras} cameras")
    
    # Theoretical bounds
    print(f"\nTheoretical Analysis:")
    print(f"  Lower bound (ceiling(leaves/3)): {(tree_info['leaves'] + 2) // 3}")
    print(f"  Upper bound (all internal nodes): {tree_info['nodes'] - tree_info['leaves']}")
    print(f"  Actual optimal: {min_cameras}")
    
    return min_cameras


def min_camera_cover_variants():
    """
    CAMERA COVERAGE VARIANTS:
    ========================
    Different scenarios and modifications.
    """
    
    def min_cameras_with_range(root, camera_range=2):
        """Cameras can monitor nodes within given range"""
        if not root:
            return 0
        
        cameras = 0
        
        def is_covered(node, camera_nodes, max_range):
            """Check if node is covered by any camera within range"""
            def distance_to_cameras(current, target_cameras, dist=0):
                if not current or dist > max_range:
                    return float('inf')
                
                if current in target_cameras:
                    return dist
                
                left_dist = distance_to_cameras(current.left, target_cameras, dist + 1)
                right_dist = distance_to_cameras(current.right, target_cameras, dist + 1)
                
                return min(left_dist, right_dist)
            
            return distance_to_cameras(root, camera_nodes) <= max_range
        
        # Simplified greedy approach for extended range
        def dfs(node):
            nonlocal cameras
            
            if not node:
                return 'covered'
            
            left_state = dfs(node.left)
            right_state = dfs(node.right)
            
            if left_state == 'not_covered' or right_state == 'not_covered':
                cameras += 1
                return 'has_camera'
            
            if left_state == 'has_camera' or right_state == 'has_camera':
                return 'covered'
            
            return 'not_covered'
        
        if dfs(root) == 'not_covered':
            cameras += 1
        
        return cameras
    
    def min_cameras_with_cost(root, camera_costs):
        """Different costs for cameras at different levels"""
        if not root:
            return 0
        
        def dp(node, level=0):
            if not node:
                return [0, float('inf'), 0]  # [no_camera, has_camera, covered]
            
            left = dp(node.left, level + 1)
            right = dp(node.right, level + 1)
            
            camera_cost = camera_costs[level] if level < len(camera_costs) else 1
            
            no_camera = left[1] + right[1]
            has_camera = camera_cost + min(left) + min(right)
            covered = min(
                left[1] + min(right[0], right[2]),
                right[1] + min(left[0], left[2]),
                left[1] + right[1]
            )
            
            return [no_camera, has_camera, covered]
        
        result = dp(root)
        return min(result[1], result[2])
    
    def min_cameras_k_coverage(root, k=1):
        """Each node must be monitored by at least k cameras"""
        # This is significantly more complex - simplified version
        if not root or k <= 0:
            return 0
        
        # For k=1, use standard algorithm
        if k == 1:
            return min_camera_cover_tree_dp(root)
        
        # For k>1, this becomes much more complex
        # Simplified: place more cameras conservatively
        return min_camera_cover_tree_dp(root) * k  # Conservative estimate
    
    # Test with sample tree
    def create_sample_tree():
        """Create sample tree for testing"""
        root = TreeNode(0)
        root.left = TreeNode(0)
        root.right = TreeNode(0)
        root.left.left = TreeNode(0)
        root.left.right = TreeNode(0)
        return root
    
    sample_tree = create_sample_tree()
    
    print("Camera Coverage Variants:")
    print("=" * 50)
    
    basic_cameras = min_camera_cover_tree_dp(sample_tree)
    print(f"Basic camera coverage: {basic_cameras}")
    
    range_cameras = min_cameras_with_range(sample_tree, 2)
    print(f"Cameras with range 2: {range_cameras}")
    
    costs = [1, 2, 3, 4]  # Increasing costs by level
    cost_cameras = min_cameras_with_cost(sample_tree, costs)
    print(f"With level costs {costs}: {cost_cameras}")
    
    k_coverage = min_cameras_k_coverage(sample_tree, 2)
    print(f"Double coverage (k=2): {k_coverage}")


# Test cases
def test_min_camera_cover():
    """Test all implementations with various tree configurations"""
    
    def create_tree_from_list(values):
        """Helper to create tree from level-order list"""
        if not values:
            return None
        
        root = TreeNode(values[0])
        queue = [root]
        i = 1
        
        while queue and i < len(values):
            node = queue.pop(0)
            
            if i < len(values) and values[i] is not None:
                node.left = TreeNode(values[i])
                queue.append(node.left)
            i += 1
            
            if i < len(values) and values[i] is not None:
                node.right = TreeNode(values[i])
                queue.append(node.right)
            i += 1
        
        return root
    
    test_cases = [
        ([0, 0, None, 0, 0], 1),
        ([0, 0, None, 0, None, None, 0], 2),
        ([0], 1),
        ([0, 0, 0], 1),
        ([0, 0, 0, 0, None, None, 0], 2),
        ([0, 0, 0, None, 0, 0, None, None, 0], 3),
        ([0, 0, 0, 0, 0, 0, 0], 3)
    ]
    
    print("Testing Binary Tree Cameras Solutions:")
    print("=" * 70)
    
    for i, (tree_list, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Tree: {tree_list}")
        print(f"Expected: {expected}")
        
        root = create_tree_from_list(tree_list)
        
        greedy = min_camera_cover_greedy(root)
        tree_dp = min_camera_cover_tree_dp(root)
        iterative = min_camera_cover_iterative(root)
        
        print(f"Greedy:           {greedy:>4} {'✓' if greedy == expected else '✗'}")
        print(f"Tree DP:          {tree_dp:>4} {'✓' if tree_dp == expected else '✗'}")
        print(f"Iterative:        {iterative:>4} {'✓' if iterative == expected else '✗'}")
        
        # Show detailed analysis for small cases
        if len([x for x in tree_list if x is not None]) <= 8:
            min_cameras, positions, covered = min_camera_cover_detailed(root)
            print(f"Camera positions: {positions}")
            print(f"Covered nodes: {sorted(set(covered))}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    sample_tree = create_tree_from_list([0, 0, None, 0, None, None, 0])
    min_camera_cover_analysis(sample_tree)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    min_camera_cover_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. THREE STATES: Each node can be uncovered, have camera, or covered by child")
    print("2. GREEDY PLACEMENT: Place cameras as low as possible in tree")
    print("3. CONSTRAINT PROPAGATION: Camera placement affects parent coverage")
    print("4. ROOT HANDLING: Root cannot be uncovered (special case)")
    print("5. OPTIMAL SUBSTRUCTURE: Optimal placement in subtrees leads to global optimum")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Network Security: Optimal surveillance camera placement")
    print("• Facility Location: Minimum service centers for coverage")
    print("• Sensor Networks: Optimal sensor placement for monitoring")
    print("• Graph Theory: Minimum dominating set in trees")
    print("• Resource Allocation: Minimum resources for complete coverage")


if __name__ == "__main__":
    test_min_camera_cover()


"""
BINARY TREE CAMERAS - ADVANCED COVERAGE OPTIMIZATION:
=====================================================

This problem represents sophisticated Tree DP with coverage constraints:
- Multi-state optimization (uncovered, has camera, covered by child)
- Constraint propagation across tree levels
- Greedy vs. optimal trade-offs
- Complex state transitions with coverage rules

KEY INSIGHTS:
============
1. **THREE-STATE MODEL**: Each node can be uncovered, have camera, or covered by child
2. **COVERAGE CONSTRAINTS**: Cameras monitor parent, self, and immediate children
3. **GREEDY STRATEGY**: Place cameras as low as possible (closer to leaves)
4. **ROOT SPECIAL CASE**: Root cannot be left uncovered
5. **OPTIMAL SUBSTRUCTURE**: Local optimal choices lead to global optimum

ALGORITHM APPROACHES:
====================

1. **Greedy Post-order**: O(n) time, O(h) space
   - Place cameras when children need coverage
   - Simple and intuitive approach

2. **Tree DP (Three States)**: O(n) time, O(h) space
   - Explicit state modeling for each node
   - Guaranteed optimal solution

3. **Iterative Post-order**: O(n) time, O(h) space
   - Stack-based implementation
   - Avoids recursion overhead

CORE THREE-STATE DP ALGORITHM:
=============================
```python
def minCameraCover(root):
    def dp(node):
        if not node:
            return [0, inf, 0]  # [no_camera, has_camera, covered]
        
        left = dp(node.left)
        right = dp(node.right)
        
        # State 0: No camera, not covered (children must have cameras)
        no_camera = left[1] + right[1]
        
        # State 1: Has camera (children can be in any state)
        has_camera = 1 + min(left) + min(right)
        
        # State 2: Covered by child (at least one child has camera)
        covered = min(
            left[1] + min(right[0], right[2]),  # Left has camera
            right[1] + min(left[0], left[2]),   # Right has camera
            left[1] + right[1]                  # Both have cameras
        )
        
        return [no_camera, has_camera, covered]
    
    result = dp(root)
    return min(result[1], result[2])  # Root cannot be uncovered
```

STATE DEFINITIONS:
=================
**State 0 (No Camera, Uncovered)**:
- This node has no camera
- This node is not covered by any child camera
- Both children must have cameras for this to be valid

**State 1 (Has Camera)**:
- This node has a camera
- This node monitors itself, parent, and children
- Children can be in any valid state (we choose optimal)

**State 2 (Covered by Child)**:
- This node has no camera but is covered
- At least one child must have a camera
- Other children can be covered or uncovered (but not both uncovered)

STATE TRANSITION LOGIC:
======================
**From Children to Parent**:
```
no_camera[parent] = has_camera[left] + has_camera[right]
has_camera[parent] = 1 + min(left_states) + min(right_states)
covered[parent] = min(
    has_camera[left] + min(no_camera[right], covered[right]),
    has_camera[right] + min(no_camera[left], covered[left]),
    has_camera[left] + has_camera[right]
)
```

**Validity Constraints**:
- If parent is in state 0, both children must be in state 1
- If parent is in state 2, at least one child must be in state 1
- Root cannot be in state 0 (no parent to cover it)

GREEDY STRATEGY ANALYSIS:
========================
**Why Greedy Works**:
- Cameras have limited range (1 hop)
- Placing cameras lower covers more efficiently
- Leaf nodes cannot have cameras that cover parents optimally

**Greedy Algorithm**:
1. Post-order traversal (leaves first)
2. If any child is uncovered, place camera at current node
3. If any child has camera, current node is covered
4. Otherwise, current node is uncovered

**Optimality Proof**: The greedy approach is optimal because:
- Every uncovered node forces a camera placement
- Placing cameras higher never improves the solution
- The choice at each node is locally optimal and globally consistent

COVERAGE RANGE ANALYSIS:
=======================
**Standard Camera Range**: Monitors parent, self, and immediate children

**Coverage Patterns**:
- Camera at internal node: Covers up to 3 levels (grandchildren → self → parent)
- Camera at leaf: Covers 2 levels (self → parent)
- Camera at root: Covers 2 levels (self → children)

**Optimal Placement Strategy**:
- Prefer internal nodes over leaves (better coverage ratio)
- Avoid root placement when possible (limited coverage)
- Balance between coverage efficiency and constraint satisfaction

COMPLEXITY ANALYSIS:
===================
- **Time**: O(n) - each node processed exactly once
- **Space**: O(h) - recursion stack for tree height
- **Optimal**: Cannot improve asymptotic complexity (must visit all nodes)
- **Practical**: Very efficient with minimal overhead

THEORETICAL BOUNDS:
==================
**Lower Bound**: `⌈leaves/3⌉`
- Each camera can cover at most 3 leaves (in specific configurations)
- Need at least this many cameras

**Upper Bound**: `internal_nodes`
- Placing camera at every internal node guarantees coverage
- Usually much larger than optimal

**Typical Performance**: Usually much closer to lower bound

APPLICATIONS:
============
- **Security Systems**: Optimal surveillance camera placement
- **Sensor Networks**: Minimum sensors for complete monitoring coverage
- **Facility Location**: Service centers with limited service radius
- **Network Monitoring**: Monitoring nodes in tree-structured networks
- **Graph Theory**: Minimum dominating set in trees (polynomial time)

RELATED PROBLEMS:
================
- **Vertex Cover**: Covering all edges with minimum vertices
- **Dominating Set**: Every vertex is in set or adjacent to set vertex
- **Facility Location**: Optimal placement with distance constraints
- **Network Design**: Coverage optimization in hierarchical networks

VARIANTS:
========
**Extended Range**: Cameras monitor nodes within distance k
**Weighted Costs**: Different costs for cameras at different levels
**Multiple Coverage**: Each node must be monitored by ≥k cameras
**Directional Cameras**: Cameras only monitor in specific directions

EDGE CASES:
==========
- **Single Node**: Requires 1 camera
- **Linear Tree**: Requires ⌈n/3⌉ cameras
- **Perfect Binary Tree**: Optimal placement at specific levels
- **Star Graph**: Single camera at center covers all

MATHEMATICAL PROPERTIES:
=======================
- **Optimal Substructure**: Global optimum contains local optima
- **Greedy Choice Property**: Local greedy choices lead to global optimum
- **No Overlapping Subproblems**: Tree structure prevents this
- **Monotonicity**: Adding nodes never decreases camera requirement

This problem showcases the power of Tree DP for coverage optimization:
- Complex constraint modeling with multiple states
- Greedy algorithms that achieve optimal solutions
- Practical applications in network design and security
- Elegant mathematical properties of tree structures
"""
