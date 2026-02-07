/*
 * Problem: In-Place Merge Sort
 * URL: https://www.geeksforgeeks.org/in-place-merge-sort/
 * 
 * Problem Statement:
 * Implement merge sort with O(1) extra space using modular arithmetic trick.
 * 
 * Sample Input:
 * arr[] = {12, 11, 13, 5, 6, 7}
 * 
 * Sample Output:
 * {5, 6, 7, 11, 12, 13}
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    /*
     * Approach: In-place merge using modular arithmetic
     * Store two values at same position: arr[i] = arr[i] + arr[j] * max_val
     * Extract original values using modulo and division
     * 
     * Time Complexity: O(n log n)
     * Space Complexity: O(1)
     */
    void Merge_In_Place(int arr[], int left, int mid, int right) {
        int max_val = *max_element(arr + left, arr + right + 1) + 1;
        int i = left, j = mid + 1, k = left;
        
        while (i <= mid && j <= right && k <= right) {
            int val1 = arr[i] % max_val;
            int val2 = arr[j] % max_val;
            
            if (val1 <= val2) {
                arr[k] += val1 * max_val;
                i++;
            } else {
                arr[k] += val2 * max_val;
                j++;
            }
            k++;
        }
        
        while (i <= mid) {
            int val1 = arr[i] % max_val;
            arr[k] += val1 * max_val;
            i++;
            k++;
        }
        
        while (j <= right) {
            int val2 = arr[j] % max_val;
            arr[k] += val2 * max_val;
            j++;
            k++;
        }
        
        for (i = left; i <= right; i++) {
            arr[i] /= max_val;
        }
    }

    void Merge_Sort_In_Place(int arr[], int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2;
            Merge_Sort_In_Place(arr, left, mid);
            Merge_Sort_In_Place(arr, mid + 1, right);
            Merge_In_Place(arr, left, mid, right);
        }
    }

    /*
     * Approach: Standard merge sort with extra space
     * Use temporary array to merge two sorted halves
     * 
     * Time Complexity: O(n log n)
     * Space Complexity: O(n)
     */
    void Merge_Standard(int arr[], int temp[], int left, int mid, int right) {
        int i = left, j = mid + 1, k = left;
        
        while (i <= mid && j <= right) {
            if (arr[i] <= arr[j]) {
                temp[k++] = arr[i++];
            } else {
                temp[k++] = arr[j++];
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
    }

    void Merge_Sort_Standard(int arr[], int temp[], int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2;
            Merge_Sort_Standard(arr, temp, left, mid);
            Merge_Sort_Standard(arr, temp, mid + 1, right);
            Merge_Standard(arr, temp, left, mid, right);
        }
    }
};

void Test_Merge_Sort_In_Place() {
    Solution sol;
    
    int arr1[] = {12, 11, 13, 5, 6, 7};
    int arr1_copy[] = {12, 11, 13, 5, 6, 7};
    sol.Merge_Sort_In_Place(arr1, 0, 5);
    int expected1[] = {5, 6, 7, 11, 12, 13};
    for (int i = 0; i < 6; i++) {
        assert(arr1[i] == expected1[i]);
    }
    
    int temp[6];
    sol.Merge_Sort_Standard(arr1_copy, temp, 0, 5);
    for (int i = 0; i < 6; i++) {
        assert(arr1_copy[i] == expected1[i]);
    }
    
    int arr2[] = {5, 4, 3, 2, 1};
    int arr2_copy[] = {5, 4, 3, 2, 1};
    sol.Merge_Sort_In_Place(arr2, 0, 4);
    int expected2[] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5; i++) {
        assert(arr2[i] == expected2[i]);
    }
    
    int temp2[5];
    sol.Merge_Sort_Standard(arr2_copy, temp2, 0, 4);
    for (int i = 0; i < 5; i++) {
        assert(arr2_copy[i] == expected2[i]);
    }
    
    int arr3[] = {1};
    sol.Merge_Sort_In_Place(arr3, 0, 0);
    assert(arr3[0] == 1);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Merge_Sort_In_Place();
    return 0;
}
