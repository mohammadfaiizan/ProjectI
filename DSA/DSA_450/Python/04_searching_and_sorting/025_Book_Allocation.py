"""
Problem: Allocate Minimum Number of Pages
URL: https://practice.geeksforgeeks.org/problems/allocate-minimum-number-of-pages0937/1

Problem Statement:
Allocate books to students minimizing the maximum pages any student reads.
Each book must be allocated to exactly one student.
Binary search on the answer (maximum pages).

Sample Input:
books = [12, 34, 67, 90]
students = 2

Sample Output:
113
"""


class Solution:
    def Solve_Binary_Search_On_Answer(self, books, students):
        """
        Approach: Binary search on maximum pages. For each maximum,
        check if we can allocate all books to students without exceeding limit.
        Time Complexity: O(n log(sum)) where n = books, sum = total pages
        Space Complexity: O(1)
        """
        if students > len(books):
            return -1
        
        left = max(books)
        right = sum(books)
        result = right
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if self.Can_Allocate_Books(books, students, mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result
    
    def Can_Allocate_Books(self, books, students, max_pages):
        count = 1
        current_sum = 0
        
        for pages in books:
            if current_sum + pages > max_pages:
                count += 1
                current_sum = pages
                if count > students:
                    return False
            else:
                current_sum += pages
        
        return count <= students


def Test_Book_Allocation():
    sol = Solution()
    
    books1 = [12, 34, 67, 90]
    assert sol.Solve_Binary_Search_On_Answer(books1, 2) == 113
    
    books2 = [10, 20, 30, 40]
    assert sol.Solve_Binary_Search_On_Answer(books2, 2) == 60
    
    books3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert sol.Solve_Binary_Search_On_Answer(books3, 5) == 15
    
    books4 = [100]
    assert sol.Solve_Binary_Search_On_Answer(books4, 1) == 100
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Book_Allocation()
