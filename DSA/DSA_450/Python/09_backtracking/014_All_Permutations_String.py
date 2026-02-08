"""
Problem: All Permutations of String
URL: https://practice.geeksforgeeks.org/problems/permutations-of-a-given-string2041/1

Problem Statement:
Print all permutations of a given string.

Sample Input/Output:
Input: str = "ABC"
Output: ABC ACB BAC BCA CAB CBA
Explanation: All permutations of ABC
"""


class Solution:
    def All_Permutations_Swap_Based(self, str_val):
        """
        Swap-based backtracking
        Time Complexity: O(n*n!)
        Space Complexity: O(n)
        """
        result = []
        
        def backtrack(s, idx):
            if idx == len(s):
                result.append(''.join(s))
                return
            
            s_list = list(s)
            for i in range(idx, len(s_list)):
                s_list[idx], s_list[i] = s_list[i], s_list[idx]
                backtrack(''.join(s_list), idx + 1)
                s_list[idx], s_list[i] = s_list[i], s_list[idx]
        
        backtrack(str_val, 0)
        return result
    
    def All_Permutations_Build_Exclude(self, str_val):
        """
        Build by excluding characters
        Time Complexity: O(n*n!)
        Space Complexity: O(n)
        """
        result = []
        current = []
        used = [False] * len(str_val)
        
        def backtrack():
            if len(current) == len(str_val):
                result.append(''.join(current))
                return
            
            for i in range(len(str_val)):
                if not used[i]:
                    used[i] = True
                    current.append(str_val[i])
                    backtrack()
                    current.pop()
                    used[i] = False
        
        backtrack()
        return result


def Test_All_Permutations_String():
    solution = Solution()
    str_val = "ABC"
    result1 = solution.All_Permutations_Swap_Based(str_val)
    result2 = solution.All_Permutations_Build_Exclude(str_val)
    
    print("Swap-Based Approach:")
    print(" ".join(result1))
    
    print("Build-Exclude Approach:")
    print(" ".join(result2))


if __name__ == "__main__":
    Test_All_Permutations_String()
