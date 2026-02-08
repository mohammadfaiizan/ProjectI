/*
Problem: Minimum Swaps for Bracket Balancing
URL: https://practice.geeksforgeeks.org/problems/minimum-swaps-for-bracket-balancing2704/1

Problem Statement:
Given a string of 2N characters consisting of N '[' brackets and N ']' brackets,
find the minimum number of swaps to make the string balanced. You can swap
adjacent characters only.

Sample Input/Output:
Input: "[]][]["
Output: 2

Input: "[[][]]"
Output: 0
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long Min_Swaps_Position_Track(string s) {
        /*
        Track positions of '[' and swap when imbalanced
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> pos;
        for (int i = 0; i < (int)s.length(); i++)
            if (s[i] == '[') pos.push_back(i);

        int count = 0, p = 0;
        long sum = 0;
        for (int i = 0; i < (int)s.length(); i++) {
            if (s[i] == '[') {
                count++;
                p++;
            } else if (s[i] == ']') {
                count--;
            }

            if (count < 0) {
                sum += pos[p] - i;
                swap(s[i], s[pos[p]]);
                p++;
                count = 1;
            }
        }
        return sum;
    }

    long Min_Swaps_Imbalance_Counter(string s) {
        /*
        Counter approach tracking imbalance
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int countLeft = 0, countRight = 0;
        int ans = 0, imbalance = 0;

        for (int i = 0; i < (int)s.length(); i++) {
            if (s[i] == '[') {
                countLeft++;
                if (imbalance > 0) {
                    ans += imbalance;
                    imbalance--;
                }
            } else if (s[i] == ']') {
                countRight++;
                imbalance = countRight - countLeft;
            }
        }
        return ans;
    }
};

void Test_Min_Swaps_Bracket_Balancing() {
    Solution sol;
    vector<string> tests = {"[]][][", "[[][]]", "][][", "[[[]]]", "][]["};

    for (auto s : tests) {
        cout << "Input: " << s << endl;
        string s1 = s;
        cout << "Position Track: " << sol.Min_Swaps_Position_Track(s1) << endl;
        cout << "Imbalance Counter: " << sol.Min_Swaps_Imbalance_Counter(s) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Min_Swaps_Bracket_Balancing();
    return 0;
}
