/*
 * Problem: Inversion of Array
 * URL: https://practice.geeksforgeeks.org/problems/inversion-of-array-1587115620/1
 * 
 * Problem Statement:
 * Count number of inversions (i < j but arr[i] > arr[j]).
 * Modified merge sort approach.
 * 
 * Sample Input:
 * arr[] = {2, 4, 1, 3, 5}
 * 
 * Sample Output:
 * 3
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    /*
     * Approach: Modified merge sort
     * During merge, count inversions when element from right half is smaller
     * Inversions = number of elements remaining in left half
     * 
     * Time Complexity: O(n log n)
     * Space Complexity: O(n)
     */
    long long Merge_And_Count(long long arr[], long long temp[], int left, int mid, int right) {
        int i = left;
        int j = mid + 1;
        int k = left;
        long long inversions = 0;
        
        while (i <= mid && j <= right) {
            if (arr[i] <= arr[j]) {
                temp[k++] = arr[i++];
            } else {
                temp[k++] = arr[j++];
                inversions += (mid - i + 1);
            }
        }
        
        while (i <= mid) {
            temp[k++] = arr[i++];
        }
        
        while (j <= right) {
            temp[k++] = arr[j++];
        }
        
        for (i = left; i <= right; i++) {
            arr[i] = temp[i];
        }
        
        return inversions;
    }

    long long Merge_Sort_And_Count(long long arr[], long long temp[], int left, int right) {
        long long inversions = 0;
        if (left < right) {
            int mid = left + (right - left) / 2;
            inversions += Merge_Sort_And_Count(arr, temp, left, mid);
            inversions += Merge_Sort_And_Count(arr, temp, mid + 1, right);
            inversions += Merge_And_Count(arr, temp, left, mid, right);
        }
        return inversions;
    }

    long long Inversion_Count_Merge_Sort(long long arr[], int n) {
        long long* temp = new long long[n];
        return Merge_Sort_And_Count(arr, temp, 0, n - 1);
    }

    /*
     * Approach: Brute force
     * Check all pairs (i, j) where i < j and arr[i] > arr[j]
     * 
     * Time Complexity: O(n^2)
     * Space Complexity: O(1)
     */
    long long Inversion_Count_Brute_Force(long long arr[], int n) {
        long long count = 0;
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (arr[i] > arr[j]) {
                    count++;
                }
            }
        }
        return count;
    }
};

void Test_Inversion_Count() {
    Solution sol;
    
    long long arr1[] = {2, 4, 1, 3, 5};
    long long arr1_copy[] = {2, 4, 1, 3, 5};
    assert(sol.Inversion_Count_Merge_Sort(arr1, 5) == 3);
    assert(sol.Inversion_Count_Brute_Force(arr1_copy, 5) == 3);
    
    long long arr2[] = {2, 3, 4, 5, 6};
    long long arr2_copy[] = {2, 3, 4, 5, 6};
    assert(sol.Inversion_Count_Merge_Sort(arr2, 5) == 0);
    assert(sol.Inversion_Count_Brute_Force(arr2_copy, 5) == 0);
    
    long long arr3[] = {5, 4, 3, 2, 1};
    long long arr3_copy[] = {5, 4, 3, 2, 1};
    assert(sol.Inversion_Count_Merge_Sort(arr3, 5) == 10);
    assert(sol.Inversion_Count_Brute_Force(arr3_copy, 5) == 10);
    
    long long arr4[] = {1};
    assert(sol.Inversion_Count_Merge_Sort(arr4, 1) == 0);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Inversion_Count();
    return 0;
}
