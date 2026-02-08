/*
Problem: Next Greater Element
URL: https://practice.geeksforgeeks.org/problems/next-larger-element-1587115620/1

Problem Statement:
For each element in the array, find the next greater element to its right. If no greater element exists, return -1.

Sample Input/Output:
Input: [4,5,2,25]
Output: [5,25,25,-1]
Input: [13,7,6,12]
Output: [-1,12,12,-1]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Next_Greater_BruteForce(vector<int>& arr) {
        int n = arr.size();
        vector<int> result(n, -1);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (arr[j] > arr[i]) {
                    result[i] = arr[j];
                    break;
                }
            }
        }
        return result;
    }

    vector<int> Next_Greater_Stack(vector<int>& arr) {
        int n = arr.size();
        vector<int> result(n, -1);
        stack<int> st;
        for (int i = n - 1; i >= 0; i--) {
            while (!st.empty() && st.top() <= arr[i]) {
                st.pop();
            }
            if (!st.empty()) {
                result[i] = st.top();
            }
            st.push(arr[i]);
        }
        return result;
    }

    vector<int> Next_Greater_Stack_LeftToRight(vector<int>& arr) {
        int n = arr.size();
        vector<int> result(n, -1);
        stack<int> st;
        for (int i = 0; i < n; i++) {
            while (!st.empty() && arr[st.top()] < arr[i]) {
                result[st.top()] = arr[i];
                st.pop();
            }
            st.push(i);
        }
        return result;
    }
};

void Test_Next_Greater_Element() {
    Solution solution;
    cout << "Next Greater Element Tests:" << endl;
    
    vector<int> arr1 = {4, 5, 2, 25};
    vector<int> result1 = solution.Next_Greater_Stack(arr1);
    cout << "Input: [4,5,2,25]" << endl;
    cout << "Output: [";
    for (int i = 0; i < result1.size(); i++) {
        cout << result1[i];
        if (i < result1.size() - 1) cout << ",";
    }
    cout << "]" << endl;
    
    vector<int> arr2 = {13, 7, 6, 12};
    vector<int> result2 = solution.Next_Greater_Stack(arr2);
    cout << "\nInput: [13,7,6,12]" << endl;
    cout << "Output: [";
    for (int i = 0; i < result2.size(); i++) {
        cout << result2[i];
        if (i < result2.size() - 1) cout << ",";
    }
    cout << "]" << endl;
    
    vector<int> arr3 = {1, 3, 2, 4};
    vector<int> result3 = solution.Next_Greater_Stack(arr3);
    cout << "\nInput: [1,3,2,4]" << endl;
    cout << "Output: [";
    for (int i = 0; i < result3.size(); i++) {
        cout << result3[i];
        if (i < result3.size() - 1) cout << ",";
    }
    cout << "]" << endl;
}

int main() {
    Test_Next_Greater_Element();
    return 0;
}
