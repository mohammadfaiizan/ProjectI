/*
Problem: The Celebrity Problem
URL: https://practice.geeksforgeeks.org/problems/the-celebrity-problem/1

Problem Statement:
In a party of n people, find the celebrity. A celebrity is someone who is known by everyone but knows nobody.

Sample Input/Output:
Input: Matrix representing who knows whom
Output: Index of celebrity or -1 if none exists
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Find_Celebrity_Stack(vector<vector<int>>& M, int n) {
        stack<int> st;
        for (int i = 0; i < n; i++) {
            st.push(i);
        }
        
        while (st.size() > 1) {
            int a = st.top();
            st.pop();
            int b = st.top();
            st.pop();
            
            if (M[a][b] == 1) {
                st.push(b);
            } else {
                st.push(a);
            }
        }
        
        int candidate = st.top();
        
        for (int i = 0; i < n; i++) {
            if (i != candidate) {
                if (M[candidate][i] == 1 || M[i][candidate] == 0) {
                    return -1;
                }
            }
        }
        
        return candidate;
    }

    int Find_Celebrity_TwoPointer(vector<vector<int>>& M, int n) {
        int left = 0;
        int right = n - 1;
        
        while (left < right) {
            if (M[left][right] == 1) {
                left++;
            } else {
                right--;
            }
        }
        
        int candidate = left;
        
        for (int i = 0; i < n; i++) {
            if (i != candidate) {
                if (M[candidate][i] == 1 || M[i][candidate] == 0) {
                    return -1;
                }
            }
        }
        
        return candidate;
    }

    int Find_Celebrity_BruteForce(vector<vector<int>>& M, int n) {
        for (int i = 0; i < n; i++) {
            bool isCelebrity = true;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    if (M[i][j] == 1 || M[j][i] == 0) {
                        isCelebrity = false;
                        break;
                    }
                }
            }
            if (isCelebrity) {
                return i;
            }
        }
        return -1;
    }
};

void Test_Celebrity_Problem() {
    Solution solution;
    cout << "Celebrity Problem Tests:" << endl;
    
    vector<vector<int>> M1 = {
        {0, 1, 0},
        {0, 0, 0},
        {0, 1, 0}
    };
    cout << "\nTest Case 1 (Celebrity exists at index 1):" << endl;
    cout << "Stack approach: " << solution.Find_Celebrity_Stack(M1, 3) << endl;
    cout << "Two-pointer approach: " << solution.Find_Celebrity_TwoPointer(M1, 3) << endl;
    cout << "Brute force approach: " << solution.Find_Celebrity_BruteForce(M1, 3) << endl;
    
    vector<vector<int>> M2 = {
        {0, 1},
        {1, 0}
    };
    cout << "\nTest Case 2 (No celebrity):" << endl;
    cout << "Stack approach: " << solution.Find_Celebrity_Stack(M2, 2) << endl;
    cout << "Two-pointer approach: " << solution.Find_Celebrity_TwoPointer(M2, 2) << endl;
    cout << "Brute force approach: " << solution.Find_Celebrity_BruteForce(M2, 2) << endl;
    
    vector<vector<int>> M3 = {
        {0, 0, 1, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 0},
        {0, 0, 1, 0}
    };
    cout << "\nTest Case 3 (Celebrity exists at index 2):" << endl;
    cout << "Stack approach: " << solution.Find_Celebrity_Stack(M3, 4) << endl;
    cout << "Two-pointer approach: " << solution.Find_Celebrity_TwoPointer(M3, 4) << endl;
    cout << "Brute force approach: " << solution.Find_Celebrity_BruteForce(M3, 4) << endl;
}

int main() {
    Test_Celebrity_Problem();
    return 0;
}
