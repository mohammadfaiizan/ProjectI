"""
979. Distribute Coins in Binary Tree - Multiple Approaches
Difficulty: Medium

You are given the root of a binary tree with n nodes where each node in the tree has node.val coins. There are n coins in total throughout the whole tree.

In one move, we may choose two adjacent nodes and move one coin from one node to another. A move consists of choosing two adjacent nodes and moving one coin from one node to another.

Return the minimum number of moves required to make every node have exactly one coin.
"""

from typing import Optional, Tuple

# Definition for a binary tree node
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class DistributeCoins:
    """Multiple approaches to find minimum moves to distribute coins"""
    
    def distributeCoins_postorder_dfs(self, root: Optional[TreeNode]) -> int:
        """
        Approach 1: Post-order DFS with Balance Calculation
        
        Calculate excess/deficit for each subtree and accumulate moves.
        
        Time: O(N), Space: O(H)
        """
        self.moves = 0
        
        def dfs(node: TreeNode) -> int:
            """
            Returns the excess coins in subtree rooted at node.
            Positive means excess, negative means deficit.
            """
            if not node:
                return 0
            
            left_excess = dfs(node.left)
            right_excess = dfs(node.right)
            
            # Add absolute values to total moves (coins flowing through this node)
            self.moves += abs(left_excess) + abs(right_excess)
            
            # Return net excess for this subtree
            # (coins in subtree) - (nodes in subtree)
            return node.val + left_excess + right_excess - 1
        
        dfs(root)
        return self.moves
    
    def distributeCoins_bottom_up_balance(self, root: Optional[TreeNode]) -> int:
        """
        Approach 2: Bottom-up Balance Tracking
        
        Track balance (coins - nodes) for each subtree.
        
        Time: O(N), Space: O(H)
        """
        def solve(node: TreeNode) -> Tuple[int, int]:
            """
            Returns (moves_in_subtree, balance_of_subtree)
            balance = coins - nodes
            """
            if not node:
                return 0, 0
            
            left_moves, left_balance = solve(node.left)
            right_moves, right_balance = solve(node.right)
            
            # Moves needed to balance this subtree
            current_moves = left_moves + right_moves + abs(left_balance) + abs(right_balance)
            
            # Balance of current subtree
            current_balance = node.val + left_balance + right_balance - 1
            
            return current_moves, current_balance
        
        moves, _ = solve(root)
        return moves
    
    def distributeCoins_flow_calculation(self, root: Optional[TreeNode]) -> int:
        """
        Approach 3: Flow Calculation Approach
        
        Calculate coin flow through each edge.
        
        Time: O(N), Space: O(H)
        """
        total_moves = 0
        
        def calculate_flow(node: TreeNode) -> int:
            """
            Returns net flow of coins from subtree to parent.
            Positive = sending coins up, Negative = needs coins from parent
            """
            nonlocal total_moves
            
            if not node:
                return 0
            
            left_flow = calculate_flow(node.left)
            right_flow = calculate_flow(node.right)
            
            # Add absolute flows to total moves
            total_moves += abs(left_flow) + abs(right_flow)
            
            # Net flow from this subtree
            return node.val + left_flow + right_flow - 1
        
        calculate_flow(root)
        return total_moves
    
    def distributeCoins_recursive_with_count(self, root: Optional[TreeNode]) -> int:
        """
        Approach 4: Recursive with Node Count
        
        Explicitly track node count and coin count for each subtree.
        
        Time: O(N), Space: O(H)
        """
        def dfs(node: TreeNode) -> Tuple[int, int, int]:
            """
            Returns (moves, coins, nodes) for subtree rooted at node
            """
            if not node:
                return 0, 0, 0
            
            left_moves, left_coins, left_nodes = dfs(node.left)
            right_moves, right_coins, right_nodes = dfs(node.right)
            
            # Total coins and nodes in current subtree
            total_coins = node.val + left_coins + right_coins
            total_nodes = 1 + left_nodes + right_nodes
            
            # Moves needed to balance left and right subtrees
            current_moves = left_moves + right_moves
            
            # Additional moves needed to balance current subtree
            left_imbalance = left_coins - left_nodes
            right_imbalance = right_coins - right_nodes
            
            current_moves += abs(left_imbalance) + abs(right_imbalance)
            
            return current_moves, total_coins, total_nodes
        
        moves, _, _ = dfs(root)
        return moves
    
    def distributeCoins_iterative_postorder(self, root: Optional[TreeNode]) -> int:
        """
        Approach 5: Iterative Post-order Traversal
        
        Use iterative approach to avoid recursion.
        
        Time: O(N), Space: O(N)
        """
        if not root:
            return 0
        
        stack = []
        last_visited = None
        current = root
        node_balance = {}  # node -> balance (coins - 1)
        total_moves = 0
        
        while stack or current:
            if current:
                stack.append(current)
                current = current.left
            else:
                peek_node = stack[-1]
                
                if peek_node.right and last_visited != peek_node.right:
                    current = peek_node.right
                else:
                    # Process current node (post-order)
                    node = stack.pop()
                    last_visited = node
                    
                    # Calculate balance for current node
                    left_balance = node_balance.get(node.left, 0) if node.left else 0
                    right_balance = node_balance.get(node.right, 0) if node.right else 0
                    
                    # Add moves for balancing children
                    total_moves += abs(left_balance) + abs(right_balance)
                    
                    # Calculate balance for current subtree
                    node_balance[node] = node.val + left_balance + right_balance - 1
        
        return total_moves
    
    def distributeCoins_greedy_approach(self, root: Optional[TreeNode]) -> int:
        """
        Approach 6: Greedy Approach with Local Optimization
        
        Greedily balance each subtree locally.
        
        Time: O(N), Space: O(H)
        """
        def greedy_balance(node: TreeNode) -> Tuple[int, int]:
            """
            Returns (moves_needed, net_excess)
            """
            if not node:
                return 0, 0
            
            left_moves, left_excess = greedy_balance(node.left)
            right_moves, right_excess = greedy_balance(node.right)
            
            # Current node needs 1 coin, has node.val coins
            current_excess = node.val - 1
            
            # Total excess in subtree
            total_excess = current_excess + left_excess + right_excess
            
            # Moves needed: balance children + move excess through current node
            total_moves = left_moves + right_moves + abs(left_excess) + abs(right_excess)
            
            return total_moves, total_excess
        
        moves, _ = greedy_balance(root)
        return moves
    
    def distributeCoins_mathematical_approach(self, root: Optional[TreeNode]) -> int:
        """
        Approach 7: Mathematical Approach
        
        Use mathematical properties of tree coin distribution.
        
        Time: O(N), Space: O(H)
        """
        def calculate_moves(node: TreeNode) -> int:
            """
            Returns the net coin flow from subtree (positive = excess, negative = deficit)
            """
            if not node:
                return 0
            
            left_flow = calculate_moves(node.left)
            right_flow = calculate_moves(node.right)
            
            # Each unit of flow costs 1 move
            self.total_moves += abs(left_flow) + abs(right_flow)
            
            # Net flow from this subtree = coins - nodes_needed
            return node.val + left_flow + right_flow - 1
        
        self.total_moves = 0
        calculate_moves(root)
        return self.total_moves

def test_distribute_coins():
    """Test distribute coins algorithms"""
    solver = DistributeCoins()
    
    test_cases = [
        # Test case 1: [3,0,0] -> 2 moves
        (TreeNode(3, TreeNode(0), TreeNode(0)), 2, "Root has 3, children have 0"),
        
        # Test case 2: [0,3,0] -> 3 moves  
        (TreeNode(0, TreeNode(3), TreeNode(0)), 3, "Left child has 3"),
        
        # Test case 3: [1,0,2] -> 2 moves
        (TreeNode(1, TreeNode(0), TreeNode(2)), 2, "Balanced distribution"),
        
        # Test case 4: [1,0,0,None,3] -> 4 moves
        (TreeNode(1, TreeNode(0, None, TreeNode(3)), TreeNode(0)), 4, "Deep imbalance"),
    ]
    
    algorithms = [
        ("Post-order DFS", solver.distributeCoins_postorder_dfs),
        ("Bottom-up Balance", solver.distributeCoins_bottom_up_balance),
        ("Flow Calculation", solver.distributeCoins_flow_calculation),
        ("Recursive with Count", solver.distributeCoins_recursive_with_count),
        ("Iterative Post-order", solver.distributeCoins_iterative_postorder),
        ("Greedy Approach", solver.distributeCoins_greedy_approach),
        ("Mathematical Approach", solver.distributeCoins_mathematical_approach),
    ]
    
    print("=== Testing Distribute Coins in Binary Tree ===")
    
    for i, (root, expected, description) in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {description} (Expected: {expected}) ---")
        
        for alg_name, alg_func in algorithms:
            try:
                # Reset any instance variables
                if hasattr(solver, 'moves'):
                    solver.moves = 0
                if hasattr(solver, 'total_moves'):
                    solver.total_moves = 0
                
                result = alg_func(root)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:22} | {status} | Moves: {result}")
            except Exception as e:
                print(f"{alg_name:22} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_distribute_coins()

"""
Distribute Coins in Binary Tree demonstrates post-order
traversal, balance calculation, and flow optimization
in tree structures with resource distribution problems.
"""
