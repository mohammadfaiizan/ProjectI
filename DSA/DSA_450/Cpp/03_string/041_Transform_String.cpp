/*
Problem: Transform One String to Another
URL: https://www.geeksforgeeks.org/transform-one-string-to-another-using-minimum-number-of-given-operation/

Problem Statement:
Given two strings A and B, find the minimum number of operations required to
transform A to B. The only allowed operation is to pick a character from A and
insert it at the front.

Sample Input/Output:
Input: A = "EACBD", B = "EABCD"
Output: 3

Input: A = "ABC", B = "BCA"
Output: 2
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Ops_Greedy(string A, string B) {
        /*
        Greedy - count mismatches from the end
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int m = A.length(), n = B.length();
        if (m != n) return -1;

        int count[256] = {0};
        for (int i = 0; i < n; i++) count[(int)B[i]]++;
        for (int i = 0; i < n; i++) count[(int)A[i]]--;
        for (int i = 0; i < 256; i++)
            if (count[i]) return -1;

        int res = 0;
        int i = n - 1, j = n - 1;
        while (i >= 0) {
            while (i >= 0 && A[i] != B[j]) {
                i--;
                res++;
            }
            if (i >= 0) {
                i--;
                j--;
            }
        }
        return res;
    }

    int Min_Ops_Simulation(string A, string B) {
        /*
        Simulate the process using deque
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        int n = A.length();
        if (n != (int)B.length()) return -1;

        int countA[256] = {0}, countB[256] = {0};
        for (char c : A) countA[(int)c]++;
        for (char c : B) countB[(int)c]++;
        for (int i = 0; i < 256; i++)
            if (countA[i] != countB[i]) return -1;

        deque<char> dq(A.begin(), A.end());
        int ops = 0;
        int j = n - 1;

        while (j >= 0) {
            if (dq.back() == B[j]) {
                dq.pop_back();
                j--;
            } else {
                char back = dq.back();
                dq.pop_back();
                dq.push_front(back);
                ops++;
            }
        }
        return ops;
    }
};

void Test_Transform_String() {
    Solution sol;
    struct TestCase { string A, B; };
    vector<TestCase> tests = {
        {"EACBD", "EABCD"},
        {"ABC", "BCA"},
        {"ABCD", "ABCD"},
        {"ABC", "DEF"}
    };

    for (auto& t : tests) {
        cout << "A: " << t.A << ", B: " << t.B << endl;
        cout << "Greedy: " << sol.Min_Ops_Greedy(t.A, t.B) << endl;
        cout << "Simulation: " << sol.Min_Ops_Simulation(t.A, t.B) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Transform_String();
    return 0;
}
