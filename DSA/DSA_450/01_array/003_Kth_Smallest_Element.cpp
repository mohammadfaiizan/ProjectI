/*
Problem: Kth Smallest Element
URL: https://practice.geeksforgeeks.org/problems/kth-smallest-element5635/1

Problem Statement:
Given an array arr[] and an integer K where K is smaller than size of array,
the task is to find the Kth smallest element in the given array.

Sample Input/Output:
Input: arr = [7, 10, 4, 3, 20, 15], K = 3
Output: 7
Explanation: 3rd smallest element is 7.

Input: arr = [7, 10, 4, 20, 15], K = 4
Output: 15
Explanation: 4th smallest element is 15.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Kth_Smallest_Sorting(vector<int> arr, int k) {
        /*
        Sorting Approach - Sort and return kth element
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        sort(arr.begin(), arr.end());
        return arr[k - 1];
    }

    int Kth_Smallest_Min_Heap(vector<int>& arr, int k) {
        /*
        Min Heap - Build min heap and extract k times
        Time Complexity: O(n + k log n)
        Space Complexity: O(n)
        */
        priority_queue<int, vector<int>, greater<int>> min_heap(arr.begin(), arr.end());
        for (int i = 0; i < k - 1; i++) min_heap.pop();
        return min_heap.top();
    }

    int Kth_Smallest_Max_Heap_Optimal(vector<int>& arr, int k) {
        /*
        Max Heap of Size K - Maintain heap of k smallest seen so far
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        */
        priority_queue<int> max_heap;
        for (int i = 0; i < (int)arr.size(); i++) {
            max_heap.push(arr[i]);
            if ((int)max_heap.size() > k) max_heap.pop();
        }
        return max_heap.top();
    }

    int Kth_Smallest_Quickselect(vector<int>& arr, int k) {
        /*
        Quickselect (Hoare's Selection) - Partition-based selection
        Time Complexity: O(n) average, O(n^2) worst
        Space Complexity: O(log n) - recursion stack
        */
        vector<int> temp = arr;
        return Quickselect_Helper(temp, 0, temp.size() - 1, k - 1);
    }

    int Kth_Smallest_STL_Nth_Element(vector<int> arr, int k) {
        /*
        STL nth_element - Uses introselect algorithm
        Time Complexity: O(n) average
        Space Complexity: O(1)
        */
        nth_element(arr.begin(), arr.begin() + k - 1, arr.end());
        return arr[k - 1];
    }

private:
    int Quickselect_Helper(vector<int>& arr, int low, int high, int k) {
        if (low <= high) {
            int pivot = Partition(arr, low, high);
            if (pivot == k) return arr[pivot];
            if (pivot > k) return Quickselect_Helper(arr, low, pivot - 1, k);
            return Quickselect_Helper(arr, pivot + 1, high, k);
        }
        return -1;
    }

    int Partition(vector<int>& arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                swap(arr[++i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[high]);
        return i + 1;
    }
};

void Test_Kth_Smallest_Element() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        int k;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{7, 10, 4, 3, 20, 15}, 3, 7},
        {{7, 10, 4, 20, 15}, 4, 15},
        {{1}, 1, 1},
        {{12, 3, 5, 7, 19}, 2, 5}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", K=" << tc.k << ", Expected=" << tc.expected << endl;

        cout << "Sorting: " << solution.Kth_Smallest_Sorting(tc.arr, tc.k) << endl;
        cout << "Min Heap: " << solution.Kth_Smallest_Min_Heap(tc.arr, tc.k) << endl;
        cout << "Max Heap: " << solution.Kth_Smallest_Max_Heap_Optimal(tc.arr, tc.k) << endl;
        cout << "Quickselect: " << solution.Kth_Smallest_Quickselect(tc.arr, tc.k) << endl;
        cout << "STL nth_element: " << solution.Kth_Smallest_STL_Nth_Element(tc.arr, tc.k) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Kth_Smallest_Element();
    return 0;
}
