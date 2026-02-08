/*
Problem: Heap Sort
URL: https://www.geeksforgeeks.org/heap-sort/

Problem Statement:
Sort an array using heap sort algorithm. Build max heap, then repeatedly extract max.

Sample Input/Output:
Input: [12,11,13,5,6,7]
Output: [5,6,7,11,12,13]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Heap_Sort_Max_Heap(vector<int>& arr) {
        /*
        Heap Sort Max Heap (Ascending Order)
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        int n = arr.size();
        
        for (int i = n / 2 - 1; i >= 0; i--) {
            Max_Heapify(arr, n, i);
        }
        
        for (int i = n - 1; i > 0; i--) {
            swap(arr[0], arr[i]);
            Max_Heapify(arr, i, 0);
        }
    }

    void Heap_Sort_Min_Heap(vector<int>& arr) {
        /*
        Heap Sort Min Heap (Descending Order)
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        int n = arr.size();
        
        for (int i = n / 2 - 1; i >= 0; i--) {
            Min_Heapify(arr, n, i);
        }
        
        for (int i = n - 1; i > 0; i--) {
            swap(arr[0], arr[i]);
            Min_Heapify(arr, i, 0);
        }
    }

private:
    void Max_Heapify(vector<int>& arr, int n, int i) {
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
            Max_Heapify(arr, n, largest);
        }
    }

    void Min_Heapify(vector<int>& arr, int n, int i) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < n && arr[left] < arr[smallest]) {
            smallest = left;
        }

        if (right < n && arr[right] < arr[smallest]) {
            smallest = right;
        }

        if (smallest != i) {
            swap(arr[i], arr[smallest]);
            Min_Heapify(arr, n, smallest);
        }
    }
};

void Test_Heap_Sort() {
    Solution solution;
    
    vector<int> arr1 = {12, 11, 13, 5, 6, 7};
    cout << "Original: ";
    for (int x : arr1) cout << x << " ";
    cout << endl;
    
    solution.Heap_Sort_Max_Heap(arr1);
    cout << "Sorted (Ascending): ";
    for (int x : arr1) cout << x << " ";
    cout << endl;
    
    vector<int> arr2 = {4, 10, 3, 5, 1};
    cout << "\nOriginal: ";
    for (int x : arr2) cout << x << " ";
    cout << endl;
    
    solution.Heap_Sort_Max_Heap(arr2);
    cout << "Sorted (Ascending): ";
    for (int x : arr2) cout << x << " ";
    cout << endl;
    
    vector<int> arr3 = {64, 34, 25, 12, 22, 11, 90};
    cout << "\nOriginal: ";
    for (int x : arr3) cout << x << " ";
    cout << endl;
    
    solution.Heap_Sort_Max_Heap(arr3);
    cout << "Sorted (Ascending): ";
    for (int x : arr3) cout << x << " ";
    cout << endl;
    
    vector<int> arr4 = {5, 2, 8, 1, 9};
    cout << "\nOriginal: ";
    for (int x : arr4) cout << x << " ";
    cout << endl;
    
    solution.Heap_Sort_Min_Heap(arr4);
    cout << "Sorted (Descending): ";
    for (int x : arr4) cout << x << " ";
    cout << endl;
}

int main() {
    Test_Heap_Sort();
    return 0;
}
