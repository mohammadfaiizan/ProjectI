"""
1372. Longest ZigZag Path in a Binary Tree - Multiple Approaches
Difficulty: Medium

You are given the root of a binary tree.

A ZigZag path for a binary tree is defined as follow:
- Choose any node in the binary tree and a direction (right or left).
- If the current direction is right, move to the right child of the current node; otherwise, move to the left child.
- Change the direction from right to left or from left to right.
- Repeat the second and third steps until you can't move in the tree.

Zigzag length is defined as the number of nodes visited - 1. (A single node has a length of 0).

Return the longest ZigZag path contained in that tree.
"""

from typing import Optional, Tuple

# Definition for a binary tree node
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class LongestZigZagPath:
    """Multiple approaches to find longest zigzag path in binary tree"""
    
    def longestZigZag_dfs_with_direction(self, root: Optional[TreeNode]) -> int:
        """
        Approach 1: DFS with Direction Tracking
        
        Track direction and length for each path during DFS.
        
        Time: O(N), Space: O(H)
        """
        self.max_length = 0
        
        def dfs(node: TreeNode, direction: str, length: int):
            """
            DFS with current direction and length
            direction: 'L' for left, 'R' for right
            """
            if not node:
                return
            
            self.max_length = max(self.max_length, length)
            
            if direction == 'L':
                # Coming from left, next should go right
                dfs(node.right, 'R', length + 1)
                dfs(node.left, 'L', 1)  # Start new path
            else:  # direction == 'R'
                # Coming from right, next should go left
                dfs(node.left, 'L', length + 1)
                dfs(node.right, 'R', 1)  # Start new path
        
        if root:
            dfs(root, 'L', 0)
            dfs(root, 'R', 0)
        
        return self.max_length
    
    def longestZigZag_bottom_up_dp(self, root: Optional[TreeNode]) -> int:
        """
        Approach 2: Bottom-up Dynamic Programming
        
        Return max zigzag lengths ending at current node in both directions.
        
        Time: O(N), Space: O(H)
        """
        def dfs(node: TreeNode) -> Tuple[int, int, int]:
            """
            Returns (max_zigzag_in_subtree, left_ending_length, right_ending_length)
            left_ending_length: max zigzag ending at this node coming from left
            right_ending_length: max zigzag ending at this node coming from right
            """
            if not node:
                return 0, -1, -1
            
            left_max, left_left, left_right = dfs(node.left)
            right_max, right_left, right_right = dfs(node.right)
            
            # Zigzag ending at current node from left child (going right)
            current_left = left_right + 1 if node.left else 0
            
            # Zigzag ending at current node from right child (going left)
            current_right = right_left + 1 if node.right else 0
            
            # Maximum zigzag in current subtree
            current_max = max(left_max, right_max, current_left, current_right)
            
            return current_max, current_left, current_right
        
        if not root:
            return 0
        
        max_zigzag, _, _ = dfs(root)
        return max_zigzag
    
    def longestZigZag_state_machine(self, root: Optional[TreeNode]) -> int:
        """
        Approach 3: State Machine Approach
        
        Model zigzag as a state machine with left/right states.
        
        Time: O(N), Space: O(H)
        """
        def solve(node: TreeNode) -> Tuple[int, int, int]:
            """
            Returns (max_length_in_subtree, max_ending_left, max_ending_right)
            max_ending_left: max zigzag ending at this node with last move being left
            max_ending_right: max zigzag ending at this node with last move being right
            """
            if not node:
                return 0, 0, 0
            
            left_max, left_ending_left, left_ending_right = solve(node.left)
            right_max, right_ending_left, right_ending_right = solve(node.right)
            
            # If we go left from current node, we extend right-ending paths from left child
            go_left_length = left_ending_right + 1 if node.left else 0
            
            # If we go right from current node, we extend left-ending paths from right child
            go_right_length = right_ending_left + 1 if node.right else 0
            
            # Maximum in current subtree
            subtree_max = max(left_max, right_max, go_left_length, go_right_length)
            
            return subtree_max, go_left_length, go_right_length
        
        if not root:
            return 0
        
        result, _, _ = solve(root)
        return result
    
    def longestZigZag_preorder_traversal(self, root: Optional[TreeNode]) -> int:
        """
        Approach 4: Preorder Traversal with Path Tracking
        
        Use preorder traversal to track all possible zigzag paths.
        
        Time: O(N), Space: O(H)
        """
        self.answer = 0
        
        def preorder(node: TreeNode, left_length: int, right_length: int):
            """
            left_length: length of zigzag ending here with last move left
            right_length: length of zigzag ending here with last move right
            """
            if not node:
                return
            
            self.answer = max(self.answer, left_length, right_length)
            
            # Go left: extend right_length, reset left_length
            preorder(node.left, right_length + 1, 0)
            
            # Go right: extend left_length, reset right_length
            preorder(node.right, 0, left_length + 1)
        
        if root:
            preorder(root, 0, 0)
        
        return self.answer
    
    def longestZigZag_iterative_dfs(self, root: Optional[TreeNode]) -> int:
        """
        Approach 5: Iterative DFS with Stack
        
        Use iterative approach to avoid recursion.
        
        Time: O(N), Space: O(N)
        """
        if not root:
            return 0
        
        max_length = 0
        # Stack stores (node, direction, length)
        # direction: 0 = left, 1 = right
        stack = [(root, 0, 0), (root, 1, 0)]
        
        while stack:
            node, direction, length = stack.pop()
            
            if not node:
                continue
            
            max_length = max(max_length, length)
            
            if direction == 0:  # Last move was left
                # Next move should be right (zigzag)
                if node.right:
                    stack.append((node.right, 1, length + 1))
                # Can also start new path going left
                if node.left:
                    stack.append((node.left, 0, 1))
            else:  # Last move was right
                # Next move should be left (zigzag)
                if node.left:
                    stack.append((node.left, 0, length + 1))
                # Can also start new path going right
                if node.right:
                    stack.append((node.right, 1, 1))
        
        return max_length
    
    def longestZigZag_memoized_dfs(self, root: Optional[TreeNode]) -> int:
        """
        Approach 6: DFS with Memoization
        
        Use memoization to cache results for subtrees.
        
        Time: O(N), Space: O(N)
        """
        memo = {}  # (node, direction) -> max_zigzag_length
        
        def dfs(node: TreeNode, direction: int) -> int:
            """
            direction: 0 = can go left next, 1 = can go right next
            Returns maximum zigzag length starting from this node in given direction
            """
            if not node:
                return 0
            
            if (node, direction) in memo:
                return memo[(node, direction)]
            
            result = 0
            
            if direction == 0:  # Can go left
                if node.left:
                    result = max(result, 1 + dfs(node.left, 1))  # Go left, next must go right
                if node.right:
                    result = max(result, dfs(node.right, 0))     # Start new path going right
            else:  # Can go right
                if node.right:
                    result = max(result, 1 + dfs(node.right, 0))  # Go right, next must go left
                if node.left:
                    result = max(result, dfs(node.left, 1))       # Start new path going left
            
            memo[(node, direction)] = result
            return result
        
        if not root:
            return 0
        
        return max(dfs(root, 0), dfs(root, 1))
    
    def longestZigZag_path_enumeration(self, root: Optional[TreeNode]) -> int:
        """
        Approach 7: Explicit Path Enumeration
        
        Enumerate all possible zigzag paths and find maximum.
        
        Time: O(N²) worst case, Space: O(H)
        """
        max_length = 0
        
        def explore_zigzag(node: TreeNode, last_direction: str, length: int):
            """
            Explore zigzag path from current node
            last_direction: 'L' or 'R' or 'START'
            """
            nonlocal max_length
            
            if not node:
                return
            
            max_length = max(max_length, length)
            
            if last_direction == 'START':
                # Can start in either direction
                explore_zigzag(node.left, 'L', 1)
                explore_zigzag(node.right, 'R', 1)
            elif last_direction == 'L':
                # Last move was left, next must be right for zigzag
                if node.right:
                    explore_zigzag(node.right, 'R', length + 1)
                # Can also start new zigzag going left
                if node.left:
                    explore_zigzag(node.left, 'L', 1)
            else:  # last_direction == 'R'
                # Last move was right, next must be left for zigzag
                if node.left:
                    explore_zigzag(node.left, 'L', length + 1)
                # Can also start new zigzag going right
                if node.right:
                    explore_zigzag(node.right, 'R', 1)
        
        if root:
            explore_zigzag(root, 'START', 0)
        
        return max_length

def test_longest_zigzag_path():
    """Test longest zigzag path algorithms"""
    solver = LongestZigZagPath()
    
    # Create test tree:      1
    #                       / \
    #                      2   3
    #                       \   \
    #                        4   5
    #                       / \
    #                      6   7
    
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.right = TreeNode(4)
    root.right.right = TreeNode(5)
    root.left.right.left = TreeNode(6)
    root.left.right.right = TreeNode(7)
    
    expected = 3  # Path: 1 -> 2 -> 4 -> 6 or 1 -> 2 -> 4 -> 7
    
    algorithms = [
        ("DFS with Direction", solver.longestZigZag_dfs_with_direction),
        ("Bottom-up DP", solver.longestZigZag_bottom_up_dp),
        ("State Machine", solver.longestZigZag_state_machine),
        ("Preorder Traversal", solver.longestZigZag_preorder_traversal),
        ("Iterative DFS", solver.longestZigZag_iterative_dfs),
        ("Memoized DFS", solver.longestZigZag_memoized_dfs),
        ("Path Enumeration", solver.longestZigZag_path_enumeration),
    ]
    
    print("=== Testing Longest ZigZag Path ===")
    print(f"Expected length: {expected}")
    
    for alg_name, alg_func in algorithms:
        try:
            # Reset instance variables
            if hasattr(solver, 'max_length'):
                solver.max_length = 0
            if hasattr(solver, 'answer'):
                solver.answer = 0
            
            result = alg_func(root)
            status = "✓" if result == expected else "✗"
            print(f"{alg_name:20} | {status} | Length: {result}")
        except Exception as e:
            print(f"{alg_name:20} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_longest_zigzag_path()

"""
Longest ZigZag Path demonstrates dynamic programming
on trees, state machine modeling, and path optimization
in binary tree traversal problems.
"""
