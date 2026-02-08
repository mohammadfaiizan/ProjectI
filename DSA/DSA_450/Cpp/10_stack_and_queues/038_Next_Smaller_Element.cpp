/*
Problem: Next Smaller Element
URL: https://www.geeksforgeeks.org/next-smaller-element/

Problem Statement:
Given an array, print the Next Smaller Element (NSE) for every element. The Next smaller Element for an element x is the first smaller element on the right side of x in array. Elements for which no smaller element exist, consider next smaller element as -1.

Sample Input/Output:
Input: arr[] = [4,8,5,2,25]
Output: [2,5,2,-1,-1]
Explanation: Next smaller element for 4 is 2, for 8 is 5, for 5 is 2, for 2 is -1, for 25 is -1.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Next_Smaller_Element_Brute_Force(vector<int>& arr) {
        /*
        Brute force
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int n = arr.size();
        vector<int> result(n, -1);
        
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[i]) {
                    result[i] = arr[j];
                    break;
                }
            }
        }
        
        return result;
    }
    
    vector<int> Next_Smaller_Element_Stack_Right_To_Left(vector<int>& arr) {
        /*
        Stack-based right-to-left
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        int n = arr.size();
        vector<int> result(n, -1);
        stack<int> st;
        
        for (int i = n - 1; i >= 0; i--) {
            while (!st.empty() && st.top() >= arr[i]) {
                st.pop();
            }
            
            if (!st.empty()) {
                result[i] = st.top();
            }
            
            st.push(arr[i]);
        }
        
        return result;
    }
    
    vector<int> Next_Smaller_Element_Stack_Left_To_Right(vector<int>& arr) {
        /*
        Stack-based left-to-right with map
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        int n = arr.size();
        vector<int> result(n, -1);
        stack<int> st;
        unordered_map<int, int> nextSmaller;
        
        for (int i = 0; i < n; i++) {
            while (!st.empty() && arr[st.top()] > arr[i]) {
                nextSmaller[st.top()] = arr[i];
                st.pop();
            }
            st.push(i);
        }
        
        for (int i = 0; i < n; i++) {
            if (nextSmaller.find(i) != nextSmaller.end()) {
                result[i] = nextSmaller[i];
            }
        }
        
        return result;
    }
};

void Test_Next_Smaller_Element() {
    Solution solution;
    
    vector<int> arr1 = {4, 8, 5, 2, 25};
    vector<int> result1 = solution.Next_Smaller_Element_Stack_Right_To_Left(arr1);
    cout << "Test 1 - Stack Right to Left: ";
    for (int val : result1) cout << val << " ";
    cout << endl;
    
    vector<int> arr2 = {13, 7, 6, 12};
    vector<int> result2 = solution.Next_Smaller_Element_Stack_Right_To_Left(arr2);
    cout << "Test 2 - Stack Right to Left: ";
    for (int val : result2) cout << val << " ";
    cout << endl;
    
    vector<int> arr3 = {11, 13, 21, 3};
    vector<int> result3 = solution.Next_Smaller_Element_Stack_Right_To_Left(arr3);
    cout << "Test 3 - Stack Right to Left: ";
    for (int val : result3) cout << val << " ";
    cout << endl;
    
    vector<int> arr4 = {4, 8, 5, 2, 25};
    vector<int> result4 = solution.Next_Smaller_Element_Brute_Force(arr4);
    cout << "Test 4 - Brute Force: ";
    for (int val : result4) cout << val << " ";
    cout << endl;
}

int main() {
    Test_Next_Smaller_Element();
    return 0;
}
