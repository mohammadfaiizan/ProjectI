/*
Problem: Count and Say
URL: https://leetcode.com/problems/count-and-say/

Problem Statement:
The count-and-say sequence is a sequence of digit strings defined by the recursive formula:
- countAndSay(1) = "1"
- countAndSay(n) is the way you would "say" the digit string from countAndSay(n-1)

Sample Input/Output:
Input: n = 1 -> Output: "1"
Input: n = 2 -> Output: "11" (one 1)
Input: n = 3 -> Output: "21" (two 1s)
Input: n = 4 -> Output: "1211" (one 2, one 1)
Input: n = 5 -> Output: "111221"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Count_And_Say_Recursive(int n) {
        /*
        Recursive - get previous result and describe it
        Time Complexity: O(2^n) total characters generated
        Space Complexity: O(2^n)
        */
        if (n == 1) return "1";
        string prev = Count_And_Say_Recursive(n - 1);
        string ans = "";
        for (int i = 0; i < (int)prev.size();) {
            char c = prev[i];
            int count = 0;
            while (i < (int)prev.size() && prev[i] == c) {
                count++;
                i++;
            }
            ans += to_string(count) + c;
        }
        return ans;
    }

    string Count_And_Say_Iterative(int n) {
        /*
        Iterative - build each term from the previous
        Time Complexity: O(2^n)
        Space Complexity: O(2^n)
        */
        string result = "1";
        for (int k = 2; k <= n; k++) {
            string next = "";
            for (int i = 0; i < (int)result.size();) {
                char c = result[i];
                int count = 0;
                while (i < (int)result.size() && result[i] == c) {
                    count++;
                    i++;
                }
                next += to_string(count) + c;
            }
            result = next;
        }
        return result;
    }
};

void Test_Count_And_Say() {
    Solution sol;

    for (int n = 1; n <= 8; n++) {
        string r1 = sol.Count_And_Say_Recursive(n);
        string r2 = sol.Count_And_Say_Iterative(n);
        cout << "n=" << n << " Recursive: " << r1 << " Iterative: " << r2;
        cout << (r1 == r2 ? " [MATCH]" : " [MISMATCH]") << endl;
    }
}

int main() {
    Test_Count_And_Say();
    return 0;
}
