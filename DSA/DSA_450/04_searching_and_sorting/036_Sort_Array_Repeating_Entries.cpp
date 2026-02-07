/*
 * Problem: Partitioning and Sorting Array with Repeating Entries
 * URL: https://www.baeldung.com/java-sorting-arrays-with-repeated-entries
 * 
 * Problem Statement:
 * Sort array with many repeated entries efficiently using 3-way partitioning (Dutch National Flag).
 * 
 * Sample Input:
 * arr[] = {2, 0, 2, 1, 1, 0}
 * 
 * Sample Output:
 * {0, 0, 1, 1, 2, 2}
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    /*
     * Approach: Three-way partition (Dutch National Flag)
     * Partition array into three parts: < pivot, == pivot, > pivot
     * Efficient for arrays with many duplicates
     * 
     * Time Complexity: O(n)
     * Space Complexity: O(1)
     */
    void Three_Way_Partition(int arr[], int n, int pivot) {
        int low = 0, mid = 0, high = n - 1;
        
        while (mid <= high) {
            if (arr[mid] < pivot) {
                swap(arr[low], arr[mid]);
                low++;
                mid++;
            } else if (arr[mid] == pivot) {
                mid++;
            } else {
                swap(arr[mid], arr[high]);
                high--;
            }
        }
    }

    void Sort_Three_Way(int arr[], int n) {
        if (n <= 1) return;
        
        int min_val = *min_element(arr, arr + n);
        int max_val = *max_element(arr, arr + n);
        
        if (min_val == max_val) return;
        
        int pivot = min_val + (max_val - min_val) / 2;
        Three_Way_Partition(arr, n, pivot);
        
        int pivot_pos = -1;
        for (int i = 0; i < n; i++) {
            if (arr[i] == pivot) {
                pivot_pos = i;
                break;
            }
        }
        
        if (pivot_pos != -1) {
            int left_size = pivot_pos;
            int right_start = pivot_pos;
            while (right_start < n && arr[right_start] == pivot) {
                right_start++;
            }
            
            if (left_size > 0) {
                Sort_Three_Way(arr, left_size);
            }
            if (right_start < n) {
                Sort_Three_Way(arr + right_start, n - right_start);
            }
        }
    }

    /*
     * Approach: Counting sort
     * Count frequency of each element, then reconstruct array
     * Efficient when range of values is small
     * 
     * Time Complexity: O(n + k) where k is range of values
     * Space Complexity: O(k)
     */
    void Sort_Counting(int arr[], int n) {
        if (n == 0) return;
        
        int min_val = *min_element(arr, arr + n);
        int max_val = *max_element(arr, arr + n);
        int range = max_val - min_val + 1;
        
        vector<int> count(range, 0);
        
        for (int i = 0; i < n; i++) {
            count[arr[i] - min_val]++;
        }
        
        int idx = 0;
        for (int i = 0; i < range; i++) {
            while (count[i] > 0) {
                arr[idx++] = i + min_val;
                count[i]--;
            }
        }
    }

    /*
     * Approach: STL sort
     * Use standard library sort function
     * 
     * Time Complexity: O(n log n)
     * Space Complexity: O(1)
     */
    void Sort_STL(int arr[], int n) {
        sort(arr, arr + n);
    }
};

void Test_Sort_Array_Repeating_Entries() {
    Solution sol;
    
    int arr1[] = {2, 0, 2, 1, 1, 0};
    int arr1_copy1[] = {2, 0, 2, 1, 1, 0};
    int arr1_copy2[] = {2, 0, 2, 1, 1, 0};
    int arr1_copy3[] = {2, 0, 2, 1, 1, 0};
    
    sol.Sort_Three_Way(arr1, 6);
    sol.Sort_Counting(arr1_copy1, 6);
    sol.Sort_STL(arr1_copy2, 6);
    
    int expected[] = {0, 0, 1, 1, 2, 2};
    for (int i = 0; i < 6; i++) {
        assert(arr1[i] == expected[i]);
        assert(arr1_copy1[i] == expected[i]);
        assert(arr1_copy2[i] == expected[i]);
    }
    
    int arr2[] = {1, 1, 1, 1};
    int arr2_copy[] = {1, 1, 1, 1};
    sol.Sort_Three_Way(arr2, 4);
    sol.Sort_Counting(arr2_copy, 4);
    for (int i = 0; i < 4; i++) {
        assert(arr2[i] == 1);
        assert(arr2_copy[i] == 1);
    }
    
    int arr3[] = {3, 1, 2};
    int arr3_copy[] = {3, 1, 2};
    sol.Sort_Three_Way(arr3, 3);
    sol.Sort_Counting(arr3_copy, 3);
    int expected3[] = {1, 2, 3};
    for (int i = 0; i < 3; i++) {
        assert(arr3[i] == expected3[i]);
        assert(arr3_copy[i] == expected3[i]);
    }
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Sort_Array_Repeating_Entries();
    return 0;
}
