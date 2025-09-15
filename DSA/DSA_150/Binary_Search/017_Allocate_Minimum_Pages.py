"""
Problem: Allocate Min number of Pages
URL: https://www.geeksforgeeks.org/problems/allocate-minimum-number-of-pages0937/1

Problem Statement:
Given N books with pages and M students. Allocate books to students such that:
1. Each student gets at least one book
2. Books are allocated in contiguous manner
3. Maximum pages allocated to a student is minimized

Sample Input/Output:
Input: books = [12, 34, 67, 90], students = 2
Output: 113
Explanation: Student 1: [12, 34, 67] = 113 pages, Student 2: [90] = 90 pages. Max = 113

Input: books = [15, 17, 20], students = 2
Output: 32
Explanation: Student 1: [15, 17] = 32, Student 2: [20] = 20. Max = 32
"""

from typing import List

class Solution:
    def Allocate_Books_Brute_Force(self, books: List[int], students: int) -> int:
        """
        Brute Force - Try all possible maximum page limits
        Time Complexity: O(sum * n)
        Space Complexity: O(1)
        """
        if students > len(books):
            return -1
        
        def Can_Allocate(max_pages: int) -> bool:
            student_count = 1
            current_pages = 0
            
            for pages in books:
                if pages > max_pages:
                    return False
                
                if current_pages + pages <= max_pages:
                    current_pages += pages
                else:
                    student_count += 1
                    current_pages = pages
                    
                    if student_count > students:
                        return False
            
            return True
        
        total_pages = sum(books)
        max_book_pages = max(books)
        
        for max_pages in range(max_book_pages, total_pages + 1):
            if Can_Allocate(max_pages):
                return max_pages
        
        return total_pages
    
    def Allocate_Books_Binary_Search_Optimal(self, books: List[int], students: int) -> int:
        """
        Binary Search on Answer - Optimal approach
        Time Complexity: O(n * log(sum))
        Space Complexity: O(1)
        """
        if students > len(books):
            return -1
        
        def Can_Allocate(max_pages: int) -> bool:
            student_count = 1
            current_pages = 0
            
            for pages in books:
                if pages > max_pages:
                    return False
                
                if current_pages + pages <= max_pages:
                    current_pages += pages
                else:
                    student_count += 1
                    current_pages = pages
                    
                    if student_count > students:
                        return False
            
            return True
        
        left = max(books)
        right = sum(books)
        result = right
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if Can_Allocate(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result

def Test_Allocate_Books():
    solution = Solution()
    
    test_cases = [
        ([12, 34, 67, 90], 2, 113),
        ([15, 17, 20], 2, 32),
        ([1, 2, 3, 4], 2, 6),
        ([10, 20, 30, 40], 2, 60)
    ]
    
    for books, students, expected in test_cases:
        result1 = solution.Allocate_Books_Brute_Force(books.copy(), students)
        result2 = solution.Allocate_Books_Binary_Search_Optimal(books.copy(), students)
        
        print(f"Books: {books}, Students: {students}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Binary Search Optimal: {result2}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Allocate_Books()
