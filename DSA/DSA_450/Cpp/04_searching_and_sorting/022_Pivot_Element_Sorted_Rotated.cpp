/*
 * Problem: Find Pivot Element in Sorted Rotated Array
 * URL: http://theoryofprogramming.com/2017/12/16/find-pivot-element-sorted-rotated-array/
 * Problem Statement:
 * Find the pivot (maximum) element in a sorted and rotated array.
 * The array was originally sorted in ascending order, then rotated.
 * 
 * Sample Input:
 * 5 6 7 8 9 10 1 2 3 4
 * 
 * Sample Output:
 * 10
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Find_Pivot_Binary_Search(vector<int>& arr) {
        /*
         * Approach: Binary search to find pivot where arr[mid] > arr[mid+1]
         * or arr[mid-1] > arr[mid]. Pivot is the maximum element.
         * Time Complexity: O(log n)
         * Space Complexity: O(1)
         */
        int n = arr.size();
        int left = 0, right = n - 1;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            
            if (mid < n - 1 && arr[mid] > arr[mid + 1]) {
                return arr[mid];
            }
            
            if (mid > 0 && arr[mid - 1] > arr[mid]) {
                return arr[mid - 1];
            }
            
            if (arr[left] > arr[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        return arr[left];
    }
    
    int Find_Pivot_Modified_Binary_Search(vector<int>& arr) {
        /*
         * Approach: Modified binary search with boundary checks.
         * Compare mid with left and right boundaries to determine search direction.
         * Time Complexity: O(log n)
         * Space Complexity: O(1)
         */
        int n = arr.size();
        int left = 0, right = n - 1;
        
        while (left <= right) {
            if (left == right) {
                return arr[left];
            }
            
            int mid = left + (right - left) / 2;
            
            if (mid < right && arr[mid] > arr[mid + 1]) {
                return arr[mid];
            }
            
            if (mid > left && arr[mid] < arr[mid - 1]) {
                return arr[mid - 1];
            }
            
            if (arr[left] >= arr[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        return arr[n - 1];
    }
};

void Test_Pivot_Element_Sorted_Rotated() {
    Solution sol;
    
    vector<int> arr1 = {5, 6, 7, 8, 9, 10, 1, 2, 3, 4};
    assert(sol.Find_Pivot_Binary_Search(arr1) == 10);
    assert(sol.Find_Pivot_Modified_Binary_Search(arr1) == 10);
    
    vector<int> arr2 = {3, 4, 5, 1, 2};
    assert(sol.Find_Pivot_Binary_Search(arr2) == 5);
    assert(sol.Find_Pivot_Modified_Binary_Search(arr2) == 5);
    
    vector<int> arr3 = {1, 2, 3, 4, 5};
    assert(sol.Find_Pivot_Binary_Search(arr3) == 5);
    assert(sol.Find_Pivot_Modified_Binary_Search(arr3) == 5);
    
    vector<int> arr4 = {7, 1, 2, 3, 4, 5, 6};
    assert(sol.Find_Pivot_Binary_Search(arr4) == 7);
    assert(sol.Find_Pivot_Modified_Binary_Search(arr4) == 7);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Pivot_Element_Sorted_Rotated();
    return 0;
}
