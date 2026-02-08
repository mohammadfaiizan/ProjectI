/*
Problem: Convert Min Heap to Max Heap
URL: https://www.geeksforgeeks.org/convert-min-heap-to-max-heap/

Problem Statement:
Given an array representing a min heap, convert it to a max heap.

Sample Input/Output:
Input: [3,5,9,6,8,20,10,12,18,9]
Output: valid max heap
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Convert_Min_To_Max_Heapify(vector<int>& arr) {
        /*
        Heapify Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = arr.size();
        for (int i = n / 2 - 1; i >= 0; i--) {
            MaxHeapify(arr, i, n);
        }
    }
    
private:
    void MaxHeapify(vector<int>& arr, int i, int n) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        
        if (left < n && arr[left] > arr[largest]) {
            largest = left;
        }
        
        if (right < n && arr[right] > arr[largest]) {
            largest = right;
        }
        
        if (largest != i) {
            swap(arr[i], arr[largest]);
            MaxHeapify(arr, largest, n);
        }
    }
};

bool IsMaxHeap(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        if (left < n && arr[left] > arr[i]) return false;
        if (right < n && arr[right] > arr[i]) return false;
    }
    return true;
}

void Test_Convert_Min_To_Max() {
    Solution solution;
    
    vector<int> arr1 = {3, 5, 9, 6, 8, 20, 10, 12, 18, 9};
    cout << "Original Min Heap: ";
    for (int num : arr1) cout << num << " ";
    cout << endl;
    
    solution.Convert_Min_To_Max_Heapify(arr1);
    cout << "Converted Max Heap: ";
    for (int num : arr1) cout << num << " ";
    cout << endl;
    cout << "Is Valid Max Heap: " << IsMaxHeap(arr1) << endl;
    
    vector<int> arr2 = {1, 2, 3, 4, 5};
    cout << "Original Min Heap 2: ";
    for (int num : arr2) cout << num << " ";
    cout << endl;
    
    solution.Convert_Min_To_Max_Heapify(arr2);
    cout << "Converted Max Heap 2: ";
    for (int num : arr2) cout << num << " ";
    cout << endl;
    cout << "Is Valid Max Heap 2: " << IsMaxHeap(arr2) << endl;
    
    vector<int> arr3 = {10, 8, 9, 5, 6, 7, 4};
    cout << "Original Min Heap 3: ";
    for (int num : arr3) cout << num << " ";
    cout << endl;
    
    solution.Convert_Min_To_Max_Heapify(arr3);
    cout << "Converted Max Heap 3: ";
    for (int num : arr3) cout << num << " ";
    cout << endl;
    cout << "Is Valid Max Heap 3: " << IsMaxHeap(arr3) << endl;
}

int main() {
    Test_Convert_Min_To_Max();
    return 0;
}
