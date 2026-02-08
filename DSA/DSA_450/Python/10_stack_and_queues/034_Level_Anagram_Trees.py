"""
Problem: Check if All Levels of Two Trees are Anagrams
URL: https://www.geeksforgeeks.org/check-if-all-levels-of-two-trees-are-anagrams-or-not/

Problem Statement:
Given two binary trees, check if all levels of one tree are anagrams of the corresponding levels in the other tree.

Sample Input/Output:
Input: Tree1:      Tree2:
        1              1
       / \            / \
      3   2          2   3
     / \            / \
    5   4          4   5
Output: Yes (Level 0: [1] = [1], Level 1: [3,2] = [2,3], Level 2: [5,4] = [4,5])
"""

from collections import deque


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def Level_Anagram_Trees_BFS(self, root1, root2):
        """
        Check level anagrams using BFS level order with sort comparison.
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        if not root1 and not root2:
            return True
        if not root1 or not root2:
            return False
        
        q1 = deque([root1])
        q2 = deque([root2])
        
        while q1 and q2:
            size1 = len(q1)
            size2 = len(q2)
            
            if size1 != size2:
                return False
            
            level1 = []
            level2 = []
            
            for i in range(size1):
                node1 = q1.popleft()
                node2 = q2.popleft()
                
                level1.append(node1.val)
                level2.append(node2.val)
                
                if node1.left:
                    q1.append(node1.left)
                if node1.right:
                    q1.append(node1.right)
                if node2.left:
                    q2.append(node2.left)
                if node2.right:
                    q2.append(node2.right)
            
            level1.sort()
            level2.sort()
            
            if level1 != level2:
                return False
        
        return len(q1) == 0 and len(q2) == 0


def Test_Level_Anagram_Trees():
    solution = Solution()
    
    root1 = TreeNode(1)
    root1.left = TreeNode(3)
    root1.right = TreeNode(2)
    root1.left.left = TreeNode(5)
    root1.left.right = TreeNode(4)
    
    root2 = TreeNode(1)
    root2.left = TreeNode(2)
    root2.right = TreeNode(3)
    root2.left.left = TreeNode(4)
    root2.left.right = TreeNode(5)
    
    result1 = solution.Level_Anagram_Trees_BFS(root1, root2)
    print(f"Test 1 - Level Anagrams: {'Yes' if result1 else 'No'}")
    
    root3 = TreeNode(1)
    root3.left = TreeNode(2)
    root3.right = TreeNode(3)
    
    root4 = TreeNode(1)
    root4.left = TreeNode(3)
    root4.right = TreeNode(2)
    root4.left.left = TreeNode(4)
    
    result2 = solution.Level_Anagram_Trees_BFS(root3, root4)
    print(f"Test 2 - Level Anagrams: {'Yes' if result2 else 'No'}")
    
    root5 = TreeNode(1)
    root5.left = TreeNode(2)
    root5.right = TreeNode(3)
    
    root6 = TreeNode(1)
    root6.left = TreeNode(2)
    root6.right = TreeNode(4)
    
    result3 = solution.Level_Anagram_Trees_BFS(root5, root6)
    print(f"Test 3 - Level Anagrams: {'Yes' if result3 else 'No'}")


if __name__ == "__main__":
    Test_Level_Anagram_Trees()
