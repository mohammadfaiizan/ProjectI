"""
Problem: Stack Permutations
URL: https://www.geeksforgeeks.org/stack-permutations-check-if-an-array-is-stack-permutation-of-other/

Problem Statement:
Given two arrays, check if one is a stack permutation of other (can second be obtained from first using a stack).

Sample Input/Output:
Input: input=[1,2,3], output=[2,1,3]
Output: Yes

Input: input=[1,2,3], output=[3,1,2]
Output: No
"""


class Solution:
    def Check_Stack_Permutation_Simulation(self, input_arr, output_arr):
        """
        Check stack permutation using simulation.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        st = []
        j = 0
        for i in range(len(input_arr)):
            st.append(input_arr[i])
            while st and j < len(output_arr) and st[-1] == output_arr[j]:
                st.pop()
                j += 1
        return j == len(output_arr) and len(st) == 0


def Test_Check_Stack_Permutation_Simulation():
    solution = Solution()
    
    input1 = [1, 2, 3]
    output1 = [2, 1, 3]
    result1 = solution.Check_Stack_Permutation_Simulation(input1, output1)
    print(f"Input: [1,2,3], Output: [2,1,3] -> {'Yes' if result1 else 'No'}")

    input2 = [1, 2, 3]
    output2 = [3, 1, 2]
    result2 = solution.Check_Stack_Permutation_Simulation(input2, output2)
    print(f"Input: [1,2,3], Output: [3,1,2] -> {'Yes' if result2 else 'No'}")

    input3 = [1, 2, 3, 4]
    output3 = [2, 4, 3, 1]
    result3 = solution.Check_Stack_Permutation_Simulation(input3, output3)
    print(f"Input: [1,2,3,4], Output: [2,4,3,1] -> {'Yes' if result3 else 'No'}")


if __name__ == "__main__":
    Test_Check_Stack_Permutation_Simulation()
