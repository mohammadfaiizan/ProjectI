"""
Problem: Print Unique Rows in a Given Boolean Matrix
URL: https://practice.geeksforgeeks.org/problems/unique-rows-in-boolean-matrix/1

Problem Statement:
Given a boolean matrix, print all unique rows.

Sample Input/Output:
Input: matrix with duplicate and unique rows
Output: unique rows only
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_row = False


class Solution:
    def Unique_Rows_Trie(self, matrix):
        """
        Unique_Rows_Trie (insert each row into trie, print only new insertions, O(R*C))
        Time Complexity: O(R*C) where R is rows, C is columns
        Space Complexity: O(R*C)
        """
        root = TrieNode()
        result = []
        
        for row in matrix:
            node = root
            is_new = False
            
            for val in row:
                if val not in node.children:
                    node.children[val] = TrieNode()
                    is_new = True
                node = node.children[val]
            
            if not node.is_end_of_row:
                node.is_end_of_row = True
                result.append(row)
        
        return result
    
    def Unique_Rows_Set(self, matrix):
        """
        Unique_Rows_Set (use set of tuples, O(R*C*log R))
        Time Complexity: O(R*C*log R) where R is rows, C is columns
        Space Complexity: O(R*C)
        """
        seen = set()
        result = []
        
        for row in matrix:
            row_tuple = tuple(row)
            if row_tuple not in seen:
                seen.add(row_tuple)
                result.append(row)
        
        return result
    
    def Unique_Rows_Set_String(self, matrix):
        """
        Unique_Rows_Set_String (convert rows to strings, use set, O(R*C*log R))
        Time Complexity: O(R*C*log R)
        Space Complexity: O(R*C)
        """
        seen = set()
        result = []
        
        for row in matrix:
            row_str = ''.join(str(val) for val in row)
            
            if row_str not in seen:
                seen.add(row_str)
                result.append(row)
        
        return result
    
    def Unique_Rows_Map(self, matrix):
        """
        Unique_Rows_Map (use map to track first occurrence, O(R*C))
        Time Complexity: O(R*C)
        Space Complexity: O(R*C)
        """
        seen = {}
        result = []
        
        for row in matrix:
            row_str = ''.join(str(val) for val in row)
            
            if row_str not in seen:
                seen[row_str] = True
                result.append(row)
        
        return result


def Test_Unique_Rows_Boolean_Matrix():
    solution = Solution()
    
    print("=== Test Case 1 ===")
    matrix1 = [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 1]
    ]
    print("Input Matrix:")
    for row in matrix1:
        print(' '.join(str(val) for val in row))
    
    result1 = solution.Unique_Rows_Trie(matrix1)
    print("Unique Rows (Trie):")
    for row in result1:
        print(' '.join(str(val) for val in row))
    
    print("\n=== Test Case 2 ===")
    matrix2 = [
        [1, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1]
    ]
    print("Input Matrix:")
    for row in matrix2:
        print(' '.join(str(val) for val in row))
    
    result2 = solution.Unique_Rows_Trie(matrix2)
    print("Unique Rows (Trie):")
    for row in result2:
        print(' '.join(str(val) for val in row))
    
    print("\n=== Test Case 3 (All unique) ===")
    matrix3 = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    print("Input Matrix:")
    for row in matrix3:
        print(' '.join(str(val) for val in row))
    
    result3 = solution.Unique_Rows_Trie(matrix3)
    print("Unique Rows (Trie):")
    for row in result3:
        print(' '.join(str(val) for val in row))
    
    print("\n=== Test Case 4 (All same) ===")
    matrix4 = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    print("Input Matrix:")
    for row in matrix4:
        print(' '.join(str(val) for val in row))
    
    result4 = solution.Unique_Rows_Trie(matrix4)
    print("Unique Rows (Trie):")
    for row in result4:
        print(' '.join(str(val) for val in row))
    
    print("\n=== Test Case 5 ===")
    matrix5 = [
        [0],
        [1],
        [0],
        [1],
        [0]
    ]
    print("Input Matrix:")
    for row in matrix5:
        print(' '.join(str(val) for val in row))
    
    result5 = solution.Unique_Rows_Trie(matrix5)
    print("Unique Rows (Trie):")
    for row in result5:
        print(' '.join(str(val) for val in row))
    
    print("\n=== Comparison Test ===")
    matrix6 = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0]
    ]
    
    trie_result = solution.Unique_Rows_Trie(matrix6)
    set_result = solution.Unique_Rows_Set(matrix6)
    map_result = solution.Unique_Rows_Map(matrix6)
    
    print(f"Trie method count: {len(trie_result)}")
    print(f"Set method count: {len(set_result)}")
    print(f"Map method count: {len(map_result)}")


if __name__ == "__main__":
    Test_Unique_Rows_Boolean_Matrix()
