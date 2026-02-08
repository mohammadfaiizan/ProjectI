"""
Problem: EKO - Eko Wood Cutting
URL: https://www.spoj.com/problems/EKO/

Problem Statement:
Find maximum height to cut trees to get at least M meters of wood.
Binary search on the answer (cutting height).

Sample Input:
4 7
20 15 10 17

Sample Output:
15
"""


class Solution:
    def Solve_Binary_Search(self, trees, required_wood):
        """
        Approach: Binary search on cutting height. For each height,
        calculate total wood obtained and adjust search range.
        Time Complexity: O(n log(max_height)) where n = trees
        Space Complexity: O(1)
        """
        left = 0
        right = max(trees)
        result = 0
        
        while left <= right:
            mid = left + (right - left) // 2
            wood_obtained = self.Get_Wood_At_Height(trees, mid)
            
            if wood_obtained >= required_wood:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    def Solve_Sorting_Math(self, trees, required_wood):
        """
        Approach: Sort trees, then calculate cumulative wood from top.
        Find the height where cumulative wood >= required.
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        trees_sorted = sorted(trees, reverse=True)
        n = len(trees_sorted)
        cumulative_wood = 0
        
        for i in range(n - 1):
            cumulative_wood += (trees_sorted[i] - trees_sorted[i + 1]) * (i + 1)
            if cumulative_wood >= required_wood:
                return trees_sorted[i + 1] + (cumulative_wood - required_wood) // (i + 1)
        
        cumulative_wood += trees_sorted[n - 1] * n
        if cumulative_wood >= required_wood:
            return (cumulative_wood - required_wood) // n
        
        return 0
    
    def Get_Wood_At_Height(self, trees, height):
        total = 0
        for tree in trees:
            if tree > height:
                total += tree - height
        return total


def Test_EKO_Wood_Cutting():
    sol = Solution()
    
    trees1 = [20, 15, 10, 17]
    assert sol.Solve_Binary_Search(trees1, 7) == 15
    
    trees2 = [4, 42, 40, 26, 46]
    result2 = sol.Solve_Binary_Search(trees2, 20)
    assert result2 >= 36
    
    trees3 = [10, 10, 10]
    assert sol.Solve_Binary_Search(trees3, 15) == 5
    
    trees4 = [1, 2, 3, 4, 5]
    assert sol.Solve_Binary_Search(trees4, 5) == 3
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_EKO_Wood_Cutting()
