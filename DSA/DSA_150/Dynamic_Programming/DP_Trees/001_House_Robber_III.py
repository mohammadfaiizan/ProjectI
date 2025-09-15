"""
Problem: House Robber III
URL: https://leetcode.com/problems/house-robber-iii/

Problem Statement:
The thief has found himself a new place for his thievery again. There is only one entrance to this area, called "root".
Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that all houses in this place form a binary tree.
It will automatically contact the police if two directly-linked houses were broken into on the same night.
Given the root of the binary tree, return the maximum amount of money the thief can rob without alerting the police.

Sample Input/Output:
Input: root = [3,2,3,null,3,null,1]
Output: 7
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.

Input: root = [3,4,5,1,3,null,1]
Output: 9
Explanation: Maximum amount of money the thief can rob = 4 + 5 = 9.
"""

from typing import Optional, Tuple, Dict

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def Rob_Brute_Force(self, root: Optional[TreeNode]) -> int:
        """
        Brute Force - Try both choices at each node
        Time Complexity: O(2^n)
        Space Complexity: O(h)
        """
        if not root:
            return 0
        
        rob_root = root.val
        if root.left:
            rob_root += self.Rob_Brute_Force(root.left.left) + self.Rob_Brute_Force(root.left.right)
        if root.right:
            rob_root += self.Rob_Brute_Force(root.right.left) + self.Rob_Brute_Force(root.right.right)
        
        not_rob_root = self.Rob_Brute_Force(root.left) + self.Rob_Brute_Force(root.right)
        
        return max(rob_root, not_rob_root)
    
    def Rob_Memoized(self, root: Optional[TreeNode]) -> int:
        """
        Memoized - Cache results to avoid recomputation
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        memo = {}
        
        def Rob_Helper(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            
            if node in memo:
                return memo[node]
            
            rob_node = node.val
            if node.left:
                rob_node += Rob_Helper(node.left.left) + Rob_Helper(node.left.right)
            if node.right:
                rob_node += Rob_Helper(node.right.left) + Rob_Helper(node.right.right)
            
            not_rob_node = Rob_Helper(node.left) + Rob_Helper(node.right)
            
            memo[node] = max(rob_node, not_rob_node)
            return memo[node]
        
        return Rob_Helper(root)
    
    def Rob_DP_Optimal(self, root: Optional[TreeNode]) -> int:
        """
        DP Optimal - Return both rob and not_rob values
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        def Rob_Helper(node: Optional[TreeNode]) -> Tuple[int, int]:
            if not node:
                return (0, 0)
            
            left_rob, left_not_rob = Rob_Helper(node.left)
            right_rob, right_not_rob = Rob_Helper(node.right)
            
            rob_node = node.val + left_not_rob + right_not_rob
            not_rob_node = max(left_rob, left_not_rob) + max(right_rob, right_not_rob)
            
            return (rob_node, not_rob_node)
        
        rob_root, not_rob_root = Rob_Helper(root)
        return max(rob_root, not_rob_root)
    
    def Rob_Bottom_Up(self, root: Optional[TreeNode]) -> int:
        """
        Bottom Up - Post-order traversal with state
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        def Post_Order(node: Optional[TreeNode]) -> Dict[str, int]:
            if not node:
                return {"rob": 0, "not_rob": 0}
            
            left = Post_Order(node.left)
            right = Post_Order(node.right)
            
            rob = node.val + left["not_rob"] + right["not_rob"]
            not_rob = max(left["rob"], left["not_rob"]) + max(right["rob"], right["not_rob"])
            
            return {"rob": rob, "not_rob": not_rob}
        
        result = Post_Order(root)
        return max(result["rob"], result["not_rob"])
    
    def Rob_With_Path(self, root: Optional[TreeNode]) -> Tuple[int, List[int]]:
        """
        With Path - Return max money and robbed houses
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        def Rob_Helper(node: Optional[TreeNode]) -> Tuple[int, int, List[int], List[int]]:
            if not node:
                return (0, 0, [], [])
            
            left_rob, left_not_rob, left_rob_path, left_not_rob_path = Rob_Helper(node.left)
            right_rob, right_not_rob, right_rob_path, right_not_rob_path = Rob_Helper(node.right)
            
            rob_node = node.val + left_not_rob + right_not_rob
            rob_path = [node.val] + left_not_rob_path + right_not_rob_path
            
            not_rob_node = max(left_rob, left_not_rob) + max(right_rob, right_not_rob)
            not_rob_path = []
            
            if left_rob > left_not_rob:
                not_rob_path.extend(left_rob_path)
            else:
                not_rob_path.extend(left_not_rob_path)
            
            if right_rob > right_not_rob:
                not_rob_path.extend(right_rob_path)
            else:
                not_rob_path.extend(right_not_rob_path)
            
            return (rob_node, not_rob_node, rob_path, not_rob_path)
        
        rob_root, not_rob_root, rob_path, not_rob_path = Rob_Helper(root)
        
        if rob_root > not_rob_root:
            return rob_root, rob_path
        else:
            return not_rob_root, not_rob_path
    
    def Rob_Iterative_DFS(self, root: Optional[TreeNode]) -> int:
        """
        Iterative DFS - Use stack for traversal
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not root:
            return 0
        
        stack = [root]
        node_states = {}
        
        while stack:
            node = stack[-1]
            
            if node in node_states:
                stack.pop()
                continue
            
            children_processed = True
            
            if node.left and node.left not in node_states:
                stack.append(node.left)
                children_processed = False
            
            if node.right and node.right not in node_states:
                stack.append(node.right)
                children_processed = False
            
            if children_processed:
                left_rob = node_states.get(node.left, {}).get("rob", 0)
                left_not_rob = node_states.get(node.left, {}).get("not_rob", 0)
                right_rob = node_states.get(node.right, {}).get("rob", 0)
                right_not_rob = node_states.get(node.right, {}).get("not_rob", 0)
                
                rob = node.val + left_not_rob + right_not_rob
                not_rob = max(left_rob, left_not_rob) + max(right_rob, right_not_rob)
                
                node_states[node] = {"rob": rob, "not_rob": not_rob}
                stack.pop()
        
        root_state = node_states[root]
        return max(root_state["rob"], root_state["not_rob"])
    
    def Rob_Level_Order(self, root: Optional[TreeNode]) -> int:
        """
        Level Order - BFS approach with state tracking
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not root:
            return 0
        
        from collections import deque, defaultdict
        
        queue = deque([root])
        node_states = defaultdict(lambda: {"rob": 0, "not_rob": 0})
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
        
        for level in reversed(levels):
            for node in level:
                left_rob = node_states[node.left]["rob"] if node.left else 0
                left_not_rob = node_states[node.left]["not_rob"] if node.left else 0
                right_rob = node_states[node.right]["rob"] if node.right else 0
                right_not_rob = node_states[node.right]["not_rob"] if node.right else 0
                
                rob = node.val + left_not_rob + right_not_rob
                not_rob = max(left_rob, left_not_rob) + max(right_rob, right_not_rob)
                
                node_states[node] = {"rob": rob, "not_rob": not_rob}
        
        return max(node_states[root]["rob"], node_states[root]["not_rob"])

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

def Test_Rob():
    solution = Solution()
    
    test_cases = [
        ([3,2,3,None,3,None,1], 7),
        ([3,4,5,1,3,None,1], 9),
        ([2,1,3,None,4], 7),
        ([4,1,None,2,None,3], 9),
        ([5,5,5,5,5,None,5], 15)
    ]
    
    methods = [
        ("Memoized", solution.Rob_Memoized),
        ("DP Optimal", solution.Rob_DP_Optimal),
        ("Bottom Up", solution.Rob_Bottom_Up),
        ("Iterative DFS", solution.Rob_Iterative_DFS),
        ("Level Order", solution.Rob_Level_Order)
    ]
    
    for tree_list, expected in test_cases:
        root = Build_Tree_From_List(tree_list)
        print(f"Tree: {Tree_To_String(root)}")
        print(f"Expected: {expected}")
        
        if len([x for x in tree_list if x is not None]) <= 10:
            result_bf = solution.Rob_Brute_Force(root)
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method(root)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        max_money, robbed_houses = solution.Rob_With_Path(root)
        print(f"With Path: Money={max_money}, Houses={robbed_houses}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Rob()
