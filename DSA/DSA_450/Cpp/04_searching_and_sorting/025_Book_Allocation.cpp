/*
 * Problem: Allocate Minimum Number of Pages
 * URL: https://practice.geeksforgeeks.org/problems/allocate-minimum-number-of-pages0937/1
 * Problem Statement:
 * Allocate books to students minimizing the maximum pages any student reads.
 * Each book must be allocated to exactly one student.
 * Binary search on the answer (maximum pages).
 * 
 * Sample Input:
 * books = [12, 34, 67, 90]
 * students = 2
 * 
 * Sample Output:
 * 113
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Solve_Binary_Search_On_Answer(vector<int>& books, int students) {
        /*
         * Approach: Binary search on maximum pages. For each maximum,
         * check if we can allocate all books to students without exceeding limit.
         * Time Complexity: O(n log(sum)) where n = books, sum = total pages
         * Space Complexity: O(1)
         */
        if (students > books.size()) {
            return -1;
        }
        
        int left = *max_element(books.begin(), books.end());
        int right = accumulate(books.begin(), books.end(), 0);
        int result = right;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (Can_Allocate_Books(books, students, mid)) {
                result = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        return result;
    }
    
private:
    bool Can_Allocate_Books(vector<int>& books, int students, int max_pages) {
        int count = 1;
        int current_sum = 0;
        
        for (int pages : books) {
            if (current_sum + pages > max_pages) {
                count++;
                current_sum = pages;
                if (count > students) {
                    return false;
                }
            } else {
                current_sum += pages;
            }
        }
        
        return count <= students;
    }
};

void Test_Book_Allocation() {
    Solution sol;
    
    vector<int> books1 = {12, 34, 67, 90};
    assert(sol.Solve_Binary_Search_On_Answer(books1, 2) == 113);
    
    vector<int> books2 = {10, 20, 30, 40};
    assert(sol.Solve_Binary_Search_On_Answer(books2, 2) == 60);
    
    vector<int> books3 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    assert(sol.Solve_Binary_Search_On_Answer(books3, 5) == 15);
    
    vector<int> books4 = {100};
    assert(sol.Solve_Binary_Search_On_Answer(books4, 1) == 100);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Book_Allocation();
    return 0;
}
