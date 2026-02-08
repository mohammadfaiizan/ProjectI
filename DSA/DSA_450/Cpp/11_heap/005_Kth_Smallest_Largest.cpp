/*
Problem: Kth Smallest and Kth Largest Element in Unsorted Array
URL: https://practice.geeksforgeeks.org/problems/kth-smallest-element5635/1

Problem Statement:
Find the Kth smallest and Kth largest element in an unsorted array.

Sample Input/Output:
Input: [7,10,4,3,20,15], k=3
Output: kth smallest=7, kth largest=10
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Kth_Element_Heap(vector<int>& arr, int k, bool smallest) {
        /*
        Heap Based Approach
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        */
        if (smallest) {
            priority_queue<int> max_heap;
            for (int num : arr) {
                if (max_heap.size() < k) {
                    max_heap.push(num);
                } else if (num < max_heap.top()) {
                    max_heap.pop();
                    max_heap.push(num);
                }
            }
            return max_heap.top();
        } else {
            priority_queue<int, vector<int>, greater<int>> min_heap;
            for (int num : arr) {
                if (min_heap.size() < k) {
                    min_heap.push(num);
                } else if (num > min_heap.top()) {
                    min_heap.pop();
                    min_heap.push(num);
                }
            }
            return min_heap.top();
        }
    }

    int Kth_Element_QuickSelect(vector<int>& arr, int k, bool smallest) {
        /*
        Randomized QuickSelect
        Time Complexity: O(n) average, O(n^2) worst case
        Space Complexity: O(1)
        */
        vector<int> arr_copy = arr;
        if (smallest) {
            return QuickSelect_Smallest(arr_copy, 0, arr_copy.size() - 1, k - 1);
        } else {
            return QuickSelect_Largest(arr_copy, 0, arr_copy.size() - 1, k - 1);
        }
    }

    int Kth_Element_Sort(vector<int>& arr, int k, bool smallest) {
        /*
        Sort and Return Kth Element
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        vector<int> sorted_arr = arr;
        sort(sorted_arr.begin(), sorted_arr.end());
        
        if (smallest) {
            return sorted_arr[k - 1];
        } else {
            return sorted_arr[sorted_arr.size() - k];
        }
    }

private:
    int Partition(vector<int>& arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[high]);
        return i + 1;
    }

    int QuickSelect_Smallest(vector<int>& arr, int low, int high, int k) {
        if (low == high) {
            return arr[low];
        }
        
        int pivot_idx = Partition(arr, low, high);
        
        if (pivot_idx == k) {
            return arr[pivot_idx];
        } else if (pivot_idx > k) {
            return QuickSelect_Smallest(arr, low, pivot_idx - 1, k);
        } else {
            return QuickSelect_Smallest(arr, pivot_idx + 1, high, k);
        }
    }

    int QuickSelect_Largest(vector<int>& arr, int low, int high, int k) {
        int target = arr.size() - 1 - k;
        return QuickSelect_Smallest(arr, low, high, target);
    }
};

void Test_Kth_Element() {
    Solution solution;
    
    vector<int> arr1 = {7, 10, 4, 3, 20, 15};
    int k1 = 3;
    
    cout << "Array: ";
    for (int x : arr1) cout << x << " ";
    cout << ", k = " << k1 << endl;
    
    int kth_smallest_heap = solution.Kth_Element_Heap(arr1, k1, true);
    int kth_largest_heap = solution.Kth_Element_Heap(arr1, k1, false);
    cout << "Heap - Kth Smallest: " << kth_smallest_heap << ", Kth Largest: " << kth_largest_heap << endl;
    
    int kth_smallest_qs = solution.Kth_Element_QuickSelect(arr1, k1, true);
    int kth_largest_qs = solution.Kth_Element_QuickSelect(arr1, k1, false);
    cout << "QuickSelect - Kth Smallest: " << kth_smallest_qs << ", Kth Largest: " << kth_largest_qs << endl;
    
    int kth_smallest_sort = solution.Kth_Element_Sort(arr1, k1, true);
    int kth_largest_sort = solution.Kth_Element_Sort(arr1, k1, false);
    cout << "Sort - Kth Smallest: " << kth_smallest_sort << ", Kth Largest: " << kth_largest_sort << endl;
    
    vector<int> arr2 = {3, 2, 1, 5, 6, 4};
    int k2 = 2;
    
    cout << "\nArray: ";
    for (int x : arr2) cout << x << " ";
    cout << ", k = " << k2 << endl;
    
    int kth_smallest2 = solution.Kth_Element_Heap(arr2, k2, true);
    int kth_largest2 = solution.Kth_Element_Heap(arr2, k2, false);
    cout << "Kth Smallest: " << kth_smallest2 << ", Kth Largest: " << kth_largest2 << endl;
    
    vector<int> arr3 = {1, 5, 2, 8, 3, 9, 4};
    int k3 = 4;
    
    cout << "\nArray: ";
    for (int x : arr3) cout << x << " ";
    cout << ", k = " << k3 << endl;
    
    int kth_smallest3 = solution.Kth_Element_Heap(arr3, k3, true);
    int kth_largest3 = solution.Kth_Element_Heap(arr3, k3, false);
    cout << "Kth Smallest: " << kth_smallest3 << ", Kth Largest: " << kth_largest3 << endl;
}

int main() {
    Test_Kth_Element();
    return 0;
}
