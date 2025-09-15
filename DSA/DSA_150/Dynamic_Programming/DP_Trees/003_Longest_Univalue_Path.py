"""
Problem: Longest Univalue Path
URL: https://leetcode.com/problems/longest-univalue-path/

Problem Statement:
Given the root of a binary tree, return the length of the longest path, where each node in the path has the same value.
This path may or may not pass through the root.
The length of the path between two nodes is represented by the number of edges between them.

Sample Input/Output:
Input: root = [5,4,5,1,1,null,5]
Output: 2
Explanation: The longest path with the same value is [5,5,5].

Input: root = [1,4,5,4,4,null,5]
Output: 2
Explanation: The longest path with the same value is [4,4,4].
"""

from typing import Optional, List, Tuple

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def Longest_Univalue_Path_Brute_Force(self, root: Optional[TreeNode]) -> int:
        """
        Brute Force - Check all possible paths
        Time Complexity: O(n²)
        Space Complexity: O(h)
        """
        def Get_All_Nodes(node: Optional[TreeNode]) -> List[TreeNode]:
            if not node:
                return []
            return [node] + Get_All_Nodes(node.left) + Get_All_Nodes(node.right)
        
        def Max_Univalue_Path_From_Node(node: Optional[TreeNode], target_val: int) -> int:
            if not node or node.val != target_val:
                return 0
            
            left_path = Max_Univalue_Path_From_Node(node.left, target_val)
            right_path = Max_Univalue_Path_From_Node(node.right, target_val)
            
            return max(left_path, right_path) + 1
        
        def Max_Univalue_Diameter_From_Node(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            
            left_path = Max_Univalue_Path_From_Node(node.left, node.val)
            right_path = Max_Univalue_Path_From_Node(node.right, node.val)
            
            return left_path + right_path
        
        all_nodes = Get_All_Nodes(root)
        max_path = 0
        
        for node in all_nodes:
            path_length = Max_Univalue_Diameter_From_Node(node)
            max_path = max(max_path, path_length)
        
        return max_path
    
    def Longest_Univalue_Path_DFS_Recursive(self, root: Optional[TreeNode]) -> int:
        """
        DFS Recursive - Each node returns max univalue path length
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        max_path = 0
        
        def DFS(node: Optional[TreeNode]) -> int:
            nonlocal max_path
            
            if not node:
                return 0
            
            left_length = DFS(node.left)
            right_length = DFS(node.right)
            
            left_path = 0
            right_path = 0
            
            if node.left and node.left.val == node.val:
                left_path = left_length + 1
            
            if node.right and node.right.val == node.val:
                right_path = right_length + 1
            
            max_path = max(max_path, left_path + right_path)
            
            return max(left_path, right_path)
        
        DFS(root)
        return max_path
    
    def Longest_Univalue_Path_Post_Order_Optimal(self, root: Optional[TreeNode]) -> int:
        """
        Post Order Optimal - Bottom-up approach
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        def Post_Order(node: Optional[TreeNode]) -> Tuple[int, int]:
            if not node:
                return (0, 0)
            
            left_max_path, left_max_diameter = Post_Order(node.left)
            right_max_path, right_max_diameter = Post_Order(node.right)
            
            current_left_path = 0
            current_right_path = 0
            
            if node.left and node.left.val == node.val:
                current_left_path = left_max_path + 1
            
            if node.right and node.right.val == node.val:
                current_right_path = right_max_path + 1
            
            current_diameter = current_left_path + current_right_path
            max_diameter = max(left_max_diameter, right_max_diameter, current_diameter)
            max_path = max(current_left_path, current_right_path)
            
            return (max_path, max_diameter)
        
        _, max_diameter = Post_Order(root)
        return max_diameter
    
    def Longest_Univalue_Path_With_Details(self, root: Optional[TreeNode]) -> Tuple[int, List[int]]:
        """
        With Details - Return length and actual path values
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        max_path_length = 0
        best_path_value = None
        
        def DFS(node: Optional[TreeNode]) -> int:
            nonlocal max_path_length, best_path_value
            
            if not node:
                return 0
            
            left_length = DFS(node.left)
            right_length = DFS(node.right)
            
            left_path = 0
            right_path = 0
            
            if node.left and node.left.val == node.val:
                left_path = left_length + 1
            
            if node.right and node.right.val == node.val:
                right_path = right_length + 1
            
            total_path = left_path + right_path
            
            if total_path > max_path_length:
                max_path_length = total_path
                best_path_value = node.val
            
            return max(left_path, right_path)
        
        DFS(root)
        
        if best_path_value is not None:
            path = [best_path_value] * (max_path_length + 1)
        else:
            path = []
        
        return max_path_length, path
    
    def Longest_Univalue_Path_Memoized(self, root: Optional[TreeNode]) -> int:
        """
        Memoized - Cache results for repeated subproblems
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        memo = {}
        max_path = 0
        
        def DFS_Memo(node: Optional[TreeNode]) -> int:
            nonlocal max_path
            
            if not node:
                return 0
            
            if node in memo:
                return memo[node]
            
            left_length = DFS_Memo(node.left)
            right_length = DFS_Memo(node.right)
            
            left_path = 0
            right_path = 0
            
            if node.left and node.left.val == node.val:
                left_path = left_length + 1
            
            if node.right and node.right.val == node.val:
                right_path = right_length + 1
            
            max_path = max(max_path, left_path + right_path)
            
            result = max(left_path, right_path)
            memo[node] = result
            return result
        
        DFS_Memo(root)
        return max_path
    
    def Longest_Univalue_Path_Iterative_DFS(self, root: Optional[TreeNode]) -> int:
        """
        Iterative DFS - Use stack to avoid recursion
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not root:
            return 0
        
        stack = [root]
        node_results = {}
        max_path = 0
        
        def Process_Node(node: TreeNode) -> int:
            nonlocal max_path
            
            left_length = node_results.get(node.left, 0)
            right_length = node_results.get(node.right, 0)
            
            left_path = 0
            right_path = 0
            
            if node.left and node.left.val == node.val:
                left_path = left_length + 1
            
            if node.right and node.right.val == node.val:
                right_path = right_length + 1
            
            max_path = max(max_path, left_path + right_path)
            
            return max(left_path, right_path)
        
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
        
        return max_path
    
    def Longest_Univalue_Path_BFS_Level_Order(self, root: Optional[TreeNode]) -> int:
        """
        BFS Level Order - Process level by level from bottom
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
        
        node_results = {}
        max_path = 0
        
        for level in reversed(levels):
            for node in level:
                left_length = node_results.get(node.left, 0)
                right_length = node_results.get(node.right, 0)
                
                left_path = 0
                right_path = 0
                
                if node.left and node.left.val == node.val:
                    left_path = left_length + 1
                
                if node.right and node.right.val == node.val:
                    right_path = right_length + 1
                
                max_path = max(max_path, left_path + right_path)
                node_results[node] = max(left_path, right_path)
        
        return max_path

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

def Test_Longest_Univalue_Path():
    solution = Solution()
    
    test_cases = [
        ([5,4,5,1,1,None,5], 2),
        ([1,4,5,4,4,None,5], 2),
        ([1], 0),
        ([1,1,1,1,1,1,1], 4),
        ([5,5,5,5,5,None,5], 4),
        ([1,None,1,None,1,None,1], 2)
    ]
    
    methods = [
        ("DFS Recursive", solution.Longest_Univalue_Path_DFS_Recursive),
        ("Post Order Optimal", solution.Longest_Univalue_Path_Post_Order_Optimal),
        ("Memoized", solution.Longest_Univalue_Path_Memoized),
        ("Iterative DFS", solution.Longest_Univalue_Path_Iterative_DFS),
        ("BFS Level Order", solution.Longest_Univalue_Path_BFS_Level_Order)
    ]
    
    for tree_list, expected in test_cases:
        root = Build_Tree_From_List(tree_list)
        print(f"Tree: {Tree_To_String(root)}")
        print(f"Expected: {expected}")
        
        if len([x for x in tree_list if x is not None]) <= 10:
            result_bf = solution.Longest_Univalue_Path_Brute_Force(root)
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method(root)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        length, path = solution.Longest_Univalue_Path_With_Details(root)
        print(f"With Details: Length={length}, Path={path}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Longest_Univalue_Path()
