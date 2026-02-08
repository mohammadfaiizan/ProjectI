"""
Problem: Maximum Sum Possible Equal Sum Three Stacks
URL: https://www.geeksforgeeks.org/find-maximum-sum-possible-equal-sum-three-stacks/

Problem Statement:
Given three stacks, find max possible equal sum by removing top elements.

Sample Input/Output:
Input: stack1=[3,2,1,1,1], stack2=[4,3,2], stack3=[1,1,4,1]
Output: 5
Explanation: Greedy remove from max sum stack approach.
"""


class Solution:
    def Max_Equal_Sum(self, stack1, stack2, stack3):
        """
        Greedy remove from max sum stack approach
        Time Complexity: O(n1+n2+n3)
        Space Complexity: O(1)
        """
        sum1 = sum(stack1)
        sum2 = sum(stack2)
        sum3 = sum(stack3)
        
        idx1 = 0
        idx2 = 0
        idx3 = 0
        
        while idx1 < len(stack1) and idx2 < len(stack2) and idx3 < len(stack3):
            if sum1 == sum2 == sum3:
                return sum1
            
            if sum1 >= sum2 and sum1 >= sum3:
                sum1 -= stack1[idx1]
                idx1 += 1
            elif sum2 >= sum1 and sum2 >= sum3:
                sum2 -= stack2[idx2]
                idx2 += 1
            else:
                sum3 -= stack3[idx3]
                idx3 += 1
        
        return 0


def Test_Max_Equal_Sum_Three_Stacks():
    solution = Solution()
    
    stack1 = [3, 2, 1, 1, 1]
    stack2 = [4, 3, 2]
    stack3 = [1, 1, 4, 1]
    print(f"Test 1: {solution.Max_Equal_Sum(stack1, stack2, stack3)}")
    
    stack4 = [1, 1, 1, 1]
    stack5 = [2, 2]
    stack6 = [4]
    print(f"Test 2: {solution.Max_Equal_Sum(stack4, stack5, stack6)}")
    
    stack7 = [1, 2, 3]
    stack8 = [2, 3, 1]
    stack9 = [3, 1, 2]
    print(f"Test 3: {solution.Max_Equal_Sum(stack7, stack8, stack9)}")


if __name__ == "__main__":
    Test_Max_Equal_Sum_Three_Stacks()
