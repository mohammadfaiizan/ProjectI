"""
Problem: Arranging Amplifiers
URL: https://www.spoj.com/problems/ARRANGE/

Problem Statement:
Arrange N amplifiers to maximize a^(b^(c^(...))). Put 1s first, rest in descending order. Special case: swap 2,3 if both present.

Sample Input/Output:
Input: [2,3,1,4]
Output: [1,3,2,4]
Explanation: Sort + special handling for 2 and 3.
"""


class Solution:
    def Arrange_Amplifiers(self, amplifiers):
        """
        Sort + special handling approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        ones = []
        others = []
        
        for amp in amplifiers:
            if amp == 1:
                ones.append(amp)
            else:
                others.append(amp)
        
        others.sort()
        
        result = ones[:]
        
        if len(others) == 2 and others[0] == 2 and others[1] == 3:
            result.append(3)
            result.append(2)
        else:
            result.extend(others)
        
        return result


def Test_Arranging_Amplifiers():
    solution = Solution()
    
    amps1 = [2, 3, 1, 4]
    result1 = solution.Arrange_Amplifiers(amps1)
    print("Test 1:", end=" ")
    for x in result1:
        print(x, end=" ")
    print()
    
    amps2 = [1, 1, 2, 3]
    result2 = solution.Arrange_Amplifiers(amps2)
    print("Test 2:", end=" ")
    for x in result2:
        print(x, end=" ")
    print()
    
    amps3 = [4, 5, 6]
    result3 = solution.Arrange_Amplifiers(amps3)
    print("Test 3:", end=" ")
    for x in result3:
        print(x, end=" ")
    print()


if __name__ == "__main__":
    Test_Arranging_Amplifiers()
