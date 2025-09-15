"""
Problem: Binary Tree Maximum Path Sum
URL: https://leetcode.com/problems/binary-tree-maximum-path-sum/

Problem Statement:
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. 
A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.
The path sum of a path is the sum of the node's values in the path.
Given the root of a binary tree, return the maximum path sum of any non-empty path.

Sample Input/Output:
Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.

Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.
"""

from typing import Optional, List, Tuple

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def Max_Path_Sum_Brute_Force(self, root: Optional[TreeNode]) -> int:
        """
        Brute Force - Check all possible paths
        Time Complexity: O(n²)
        Space Complexity: O(h)
        """
        def Get_All_Paths(node: Optional[TreeNode], current_path: List[int], all_paths: List[List[int]]) -> None:
            if not node:
                return
            
            current_path.append(node.val)
            
            if not node.left and not node.right:
                all_paths.append(current_path[:])
            else:
                if node.left:
                    Get_All_Paths(node.left, current_path, all_paths)
                if node.right:
                    Get_All_Paths(node.right, current_path, all_paths)
            
            current_path.pop()
        
        def Get_All_Nodes(node: Optional[TreeNode]) -> List[TreeNode]:
            if not node:
                return []
            return [node] + Get_All_Nodes(node.left) + Get_All_Nodes(node.right)
        
        def Max_Path_From_Node(node: TreeNode) -> int:
            max_sum = node.val
            
            def DFS_Path(curr: Optional[TreeNode], visited: set, current_sum: int) -> None:
                nonlocal max_sum
                if not curr or curr in visited:
                    return
                
                visited.add(curr)
                current_sum += curr.val
                max_sum = max(max_sum, current_sum)
                
                if curr.left:
                    DFS_Path(curr.left, visited, current_sum)
                if curr.right:
                    DFS_Path(curr.right, visited, current_sum)
                if curr != node:
                    parent = Find_Parent(root, curr)
                    if parent:
                        DFS_Path(parent, visited, current_sum)
                
                visited.remove(curr)
            
            DFS_Path(node, set(), 0)
            return max_sum
        
        def Find_Parent(root: Optional[TreeNode], target: TreeNode) -> Optional[TreeNode]:
            if not root or root == target:
                return None
            
            if (root.left == target) or (root.right == target):
                return root
            
            left_parent = Find_Parent(root.left, target)
            if left_parent:
                return left_parent
            
            return Find_Parent(root.right, target)
        
        all_nodes = Get_All_Nodes(root)
        max_path_sum = float('-inf')
        
        for node in all_nodes:
            path_sum = Max_Path_From_Node(node)
            max_path_sum = max(max_path_sum, path_sum)
        
        return max_path_sum
    
    def Max_Path_Sum_DFS_Optimal(self, root: Optional[TreeNode]) -> int:
        """
        DFS Optimal - Each node returns max path sum ending at that node
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        max_sum = float('-inf')
        
        def Max_Path_Ending_At_Node(node: Optional[TreeNode]) -> int:
            nonlocal max_sum
            
            if not node:
                return 0
            
            left_max = max(0, Max_Path_Ending_At_Node(node.left))
            right_max = max(0, Max_Path_Ending_At_Node(node.right))
            
            current_max_path = node.val + left_max + right_max
            max_sum = max(max_sum, current_max_path)
            
            return node.val + max(left_max, right_max)
        
        Max_Path_Ending_At_Node(root)
        return max_sum
    
    def Max_Path_Sum_Bottom_Up_DP(self, root: Optional[TreeNode]) -> int:
        """
        Bottom Up DP - Post-order traversal with DP
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        def Post_Order_DP(node: Optional[TreeNode]) -> Tuple[int, int]:
            if not node:
                return (0, float('-inf'))
            
            left_path, left_global = Post_Order_DP(node.left)
            right_path, right_global = Post_Order_DP(node.right)
            
            left_path = max(0, left_path)
            right_path = max(0, right_path)
            
            max_path_through_node = node.val + left_path + right_path
            max_path_ending_here = node.val + max(left_path, right_path)
            
            global_max = max(left_global, right_global, max_path_through_node)
            
            return (max_path_ending_here, global_max)
        
        _, global_max = Post_Order_DP(root)
        return global_max
    
    def Max_Path_Sum_With_Path_Tracking(self, root: Optional[TreeNode]) -> Tuple[int, List[int]]:
        """
        With Path Tracking - Return max sum and actual path
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        max_sum = float('-inf')
        best_path = []
        
        def DFS_With_Path(node: Optional[TreeNode]) -> Tuple[int, List[int]]:
            nonlocal max_sum, best_path
            
            if not node:
                return (0, [])
            
            left_sum, left_path = DFS_With_Path(node.left)
            right_sum, right_path = DFS_With_Path(node.right)
            
            left_sum = max(0, left_sum)
            right_sum = max(0, right_sum)
            
            current_path_sum = node.val + left_sum + right_sum
            
            if current_path_sum > max_sum:
                max_sum = current_path_sum
                
                path = []
                if left_sum > 0:
                    path.extend(left_path[::-1])
                path.append(node.val)
                if right_sum > 0:
                    path.extend(right_path)
                
                best_path = path
            
            if left_sum > right_sum:
                return (node.val + left_sum, [node.val] + left_path)
            else:
                return (node.val + right_sum, [node.val] + right_path)
        
        DFS_With_Path(root)
        return max_sum, best_path
    
    def Max_Path_Sum_Iterative_DFS(self, root: Optional[TreeNode]) -> int:
        """
        Iterative DFS - Use stack to avoid recursion
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not root:
            return 0
        
        stack = [root]
        node_results = {}
        max_sum = float('-inf')
        
        def Process_Node(node: TreeNode) -> int:
            nonlocal max_sum
            
            left_max = max(0, node_results.get(node.left, 0))
            right_max = max(0, node_results.get(node.right, 0))
            
            current_max_path = node.val + left_max + right_max
            max_sum = max(max_sum, current_max_path)
            
            return node.val + max(left_max, right_max)
        
        while stack:
            node = stack[-1]
            
            if node in node_results:
                stack.pop()
                continue
            
            children_processed = True
            
            if node.left and node.left not in node_results:
                stack.append(node.left)
                children_processed = False
            
            if node.right and node.right not in node_results:
                stack.append(node.right)
                children_processed = False
            
            if children_processed:
                node_results[node] = Process_Node(node)
                stack.pop()
        
        return max_sum
    
    def Max_Path_Sum_Level_Order_Bottom_Up(self, root: Optional[TreeNode]) -> int:
        """
        Level Order Bottom Up - Process levels from bottom to top
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not root:
            return 0
        
        from collections import deque
        
        queue = deque([root])
        levels = []
        
        while queue:
            level_size = len(queue)
            current_level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            levels.append(current_level)
        
        node_max_path = {}
        max_sum = float('-inf')
        
        for level in reversed(levels):
            for node in level:
                left_max = max(0, node_max_path.get(node.left, 0))
                right_max = max(0, node_max_path.get(node.right, 0))
                
                current_max_through = node.val + left_max + right_max
                max_sum = max(max_sum, current_max_through)
                
                node_max_path[node] = node.val + max(left_max, right_max)
        
        return max_sum
    
    def Max_Path_Sum_Memoized(self, root: Optional[TreeNode]) -> int:
        """
        Memoized - Cache results for repeated subproblems
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        memo = {}
        max_sum = float('-inf')
        
        def Max_Path_Memo(node: Optional[TreeNode]) -> int:
            nonlocal max_sum
            
            if not node:
                return 0
            
            if node in memo:
                return memo[node]
            
            left_max = max(0, Max_Path_Memo(node.left))
            right_max = max(0, Max_Path_Memo(node.right))
            
            current_max_path = node.val + left_max + right_max
            max_sum = max(max_sum, current_max_path)
            
            result = node.val + max(left_max, right_max)
            memo[node] = result
            return result
        
        Max_Path_Memo(root)
        return max_sum

def Build_Tree_From_List(values: List) -> Optional[TreeNode]:
    """Helper function to build tree from list representation"""
    if not values:
        return None
    
    from collections import deque
    
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    
    while queue and i < len(values):
        node = queue.popleft()
        
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    
    return root

def Tree_To_String(root: Optional[TreeNode]) -> str:
    """Helper function to convert tree to string representation"""
    if not root:
        return "[]"
    
    from collections import deque
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        if node:
            result.append(str(node.val))
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append("null")
    
    while result and result[-1] == "null":
        result.pop()
    
    return "[" + ",".join(result) + "]"

def Test_Max_Path_Sum():
    solution = Solution()
    
    test_cases = [
        ([1,2,3], 6),
        ([-10,9,20,None,None,15,7], 42),
        ([1,2], 3),
        ([-3], -3),
        ([2,-1], 2),
        ([5,4,8,11,None,13,4,7,2,None,None,None,1], 48)
    ]
    
    methods = [
        ("DFS Optimal", solution.Max_Path_Sum_DFS_Optimal),
        ("Bottom Up DP", solution.Max_Path_Sum_Bottom_Up_DP),
        ("Iterative DFS", solution.Max_Path_Sum_Iterative_DFS),
        ("Level Order Bottom Up", solution.Max_Path_Sum_Level_Order_Bottom_Up),
        ("Memoized", solution.Max_Path_Sum_Memoized)
    ]
    
    for tree_list, expected in test_cases:
        root = Build_Tree_From_List(tree_list)
        print(f"Tree: {Tree_To_String(root)}")
        print(f"Expected: {expected}")
        
        if len([x for x in tree_list if x is not None]) <= 8:
            try:
                result_bf = solution.Max_Path_Sum_Brute_Force(root)
                print(f"Brute Force: {result_bf}")
            except Exception as e:
                print(f"Brute Force: Error - {e}")
        
        for method_name, method in methods:
            try:
                result = method(root)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        if len([x for x in tree_list if x is not None]) <= 10:
            max_sum, path = solution.Max_Path_Sum_With_Path_Tracking(root)
            print(f"With Path: Sum={max_sum}, Path={path}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Path_Sum()
