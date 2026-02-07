/*
 * Problem: K-th Element of Two Sorted Arrays
 * URL: https://practice.geeksforgeeks.org/problems/k-th-element-of-two-sorted-array1317/1
 * Problem Statement:
 * Given two sorted arrays of size m and n, find the kth element
 * in the union of the two arrays.
 * 
 * Sample Input:
 * arr1 = [2, 3, 6, 7, 9]
 * arr2 = [1, 4, 8, 10]
 * k = 5
 * 
 * Sample Output:
 * 6
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Kth_Element_Merge(vector<int>& arr1, vector<int>& arr2, int k) {
        /*
         * Approach: Merge both arrays until kth element is found.
         * Use two pointers to traverse both arrays simultaneously.
         * Time Complexity: O(k)
         * Space Complexity: O(1)
         */
        int i = 0, j = 0;
        int count = 0;
        int m = arr1.size(), n = arr2.size();
        
        while (i < m && j < n) {
            if (arr1[i] <= arr2[j]) {
                count++;
                if (count == k) return arr1[i];
                i++;
            } else {
                count++;
                if (count == k) return arr2[j];
                j++;
            }
        }
        
        while (i < m) {
            count++;
            if (count == k) return arr1[i];
            i++;
        }
        
        while (j < n) {
            count++;
            if (count == k) return arr2[j];
            j++;
        }
        
        return -1;
    }
    
    int Kth_Element_Binary_Search(vector<int>& arr1, vector<int>& arr2, int k) {
        /*
         * Approach: Binary search on smaller array. Partition both arrays
         * such that left partition has k elements total.
         * Time Complexity: O(log(min(m,n)))
         * Space Complexity: O(1)
         */
        int m = arr1.size(), n = arr2.size();
        if (m > n) {
            return Kth_Element_Binary_Search(arr2, arr1, k);
        }
        
        int left = max(0, k - n), right = min(k, m);
        
        while (left <= right) {
            int partition1 = (left + right) / 2;
            int partition2 = k - partition1;
            
            int left1 = (partition1 == 0) ? INT_MIN : arr1[partition1 - 1];
            int right1 = (partition1 == m) ? INT_MAX : arr1[partition1];
            int left2 = (partition2 == 0) ? INT_MIN : arr2[partition2 - 1];
            int right2 = (partition2 == n) ? INT_MAX : arr2[partition2];
            
            if (left1 <= right2 && left2 <= right1) {
                return max(left1, left2);
            } else if (left1 > right2) {
                right = partition1 - 1;
            } else {
                left = partition1 + 1;
            }
        }
        
        return -1;
    }
    
    int Kth_Element_Min_Heap(vector<int>& arr1, vector<int>& arr2, int k) {
        /*
         * Approach: Use min heap to merge and find kth element.
         * Push elements from both arrays and pop k times.
         * Time Complexity: O(k log k)
         * Space Complexity: O(k)
         */
        priority_queue<int, vector<int>, greater<int>> pq;
        
        for (int num : arr1) {
            pq.push(num);
        }
        
        for (int num : arr2) {
            pq.push(num);
        }
        
        for (int i = 0; i < k - 1; i++) {
            pq.pop();
        }
        
        return pq.top();
    }
};

void Test_Kth_Element_Two_Sorted_Arrays() {
    Solution sol;
    
    vector<int> arr1 = {2, 3, 6, 7, 9};
    vector<int> arr2 = {1, 4, 8, 10};
    assert(sol.Kth_Element_Merge(arr1, arr2, 5) == 6);
    assert(sol.Kth_Element_Binary_Search(arr1, arr2, 5) == 6);
    assert(sol.Kth_Element_Min_Heap(arr1, arr2, 5) == 6);
    
    vector<int> arr3 = {1, 3, 5};
    vector<int> arr4 = {2, 4, 6};
    assert(sol.Kth_Element_Merge(arr3, arr4, 4) == 4);
    assert(sol.Kth_Element_Binary_Search(arr3, arr4, 4) == 4);
    
    vector<int> arr5 = {1};
    vector<int> arr6 = {2, 3, 4};
    assert(sol.Kth_Element_Merge(arr5, arr6, 2) == 2);
    assert(sol.Kth_Element_Binary_Search(arr5, arr6, 2) == 2);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Kth_Element_Two_Sorted_Arrays();
    return 0;
}
