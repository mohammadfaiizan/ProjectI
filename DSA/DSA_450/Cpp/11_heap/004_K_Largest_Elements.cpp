/*
Problem: K Largest Elements in Array
URL: https://practice.geeksforgeeks.org/problems/k-largest-elements3736/1

Problem Statement:
Find K largest elements from an unsorted array.

Sample Input/Output:
Input: [1,23,12,9,30,2,50], k=3
Output: [50,30,23]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> K_Largest_Min_Heap(vector<int>& arr, int k) {
        /*
        Min Heap of Size K
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        */
        priority_queue<int, vector<int>, greater<int>> min_heap;
        
        for (int num : arr) {
            if (min_heap.size() < k) {
                min_heap.push(num);
            } else if (num > min_heap.top()) {
                min_heap.pop();
                min_heap.push(num);
            }
        }
        
        vector<int> result;
        while (!min_heap.empty()) {
            result.push_back(min_heap.top());
            min_heap.pop();
        }
        
        reverse(result.begin(), result.end());
        return result;
    }

    vector<int> K_Largest_Sort(vector<int>& arr, int k) {
        /*
        Sort Descending and Take First K
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        vector<int> sorted_arr = arr;
        sort(sorted_arr.begin(), sorted_arr.end(), greater<int>());
        
        vector<int> result;
        for (int i = 0; i < k && i < sorted_arr.size(); i++) {
            result.push_back(sorted_arr[i]);
        }
        
        return result;
    }
};

void Test_K_Largest() {
    Solution solution;
    
    vector<int> arr1 = {1, 23, 12, 9, 30, 2, 50};
    int k1 = 3;
    
    cout << "Array: ";
    for (int x : arr1) cout << x << " ";
    cout << ", k = " << k1 << endl;
    
    vector<int> res1 = solution.K_Largest_Min_Heap(arr1, k1);
    cout << "Min Heap Result: ";
    for (int x : res1) cout << x << " ";
    cout << endl;
    
    vector<int> res2 = solution.K_Largest_Sort(arr1, k1);
    cout << "Sort Result: ";
    for (int x : res2) cout << x << " ";
    cout << endl;
    
    vector<int> arr2 = {12, 5, 787, 1, 23};
    int k2 = 2;
    
    cout << "\nArray: ";
    for (int x : arr2) cout << x << " ";
    cout << ", k = " << k2 << endl;
    
    vector<int> res3 = solution.K_Largest_Min_Heap(arr2, k2);
    cout << "Min Heap Result: ";
    for (int x : res3) cout << x << " ";
    cout << endl;
    
    vector<int> arr3 = {7, 10, 4, 3, 20, 15};
    int k3 = 3;
    
    cout << "\nArray: ";
    for (int x : arr3) cout << x << " ";
    cout << ", k = " << k3 << endl;
    
    vector<int> res4 = solution.K_Largest_Min_Heap(arr3, k3);
    cout << "Min Heap Result: ";
    for (int x : res4) cout << x << " ";
    cout << endl;
}

int main() {
    Test_K_Largest();
    return 0;
}
