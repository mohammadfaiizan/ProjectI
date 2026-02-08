/*
Problem: Rabin-Karp Algorithm for Pattern Searching
URL: https://www.geeksforgeeks.org/rabin-karp-algorithm-for-pattern-searching/

Problem Statement:
Given a text string and a pattern string, find all occurrences of the pattern
in the text using Rabin-Karp algorithm with rolling hash.

Sample Input/Output:
Input: text = "GEEKS FOR GEEKS", pattern = "GEEK"
Output: Pattern found at index 0, Pattern found at index 10
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Rabin_Karp_Search(string txt, string pat, int q = 101) {
        /*
        Rabin-Karp with rolling hash
        Time Complexity: O(n + m) average, O(n * m) worst case
        Space Complexity: O(1)
        */
        vector<int> result;
        int d = 256;
        int M = pat.size();
        int N = txt.size();
        int p = 0, t = 0, h = 1;

        for (int i = 0; i < M - 1; i++)
            h = (h * d) % q;

        for (int i = 0; i < M; i++) {
            p = (d * p + pat[i]) % q;
            t = (d * t + txt[i]) % q;
        }

        for (int i = 0; i <= N - M; i++) {
            if (p == t) {
                bool match = true;
                for (int j = 0; j < M; j++) {
                    if (txt[i + j] != pat[j]) { match = false; break; }
                }
                if (match) result.push_back(i);
            }

            if (i < N - M) {
                t = (d * (t - txt[i] * h) + txt[i + M]) % q;
                if (t < 0) t += q;
            }
        }
        return result;
    }

    vector<int> Naive_Search(string txt, string pat) {
        /*
        Naive pattern matching
        Time Complexity: O(n * m)
        Space Complexity: O(1)
        */
        vector<int> result;
        int N = txt.size(), M = pat.size();
        for (int i = 0; i <= N - M; i++) {
            int j;
            for (j = 0; j < M; j++) {
                if (txt[i + j] != pat[j]) break;
            }
            if (j == M) result.push_back(i);
        }
        return result;
    }

    vector<int> Rabin_Karp_Multiple_Hash(string txt, string pat) {
        /*
        Double hashing to reduce spurious hits
        Time Complexity: O(n + m) average
        Space Complexity: O(1)
        */
        vector<int> result;
        int q1 = 101, q2 = 103, d = 256;
        int M = pat.size(), N = txt.size();
        if (M > N) return result;

        int p1 = 0, t1 = 0, h1 = 1;
        int p2 = 0, t2 = 0, h2 = 1;

        for (int i = 0; i < M - 1; i++) {
            h1 = (h1 * d) % q1;
            h2 = (h2 * d) % q2;
        }

        for (int i = 0; i < M; i++) {
            p1 = (d * p1 + pat[i]) % q1;
            t1 = (d * t1 + txt[i]) % q1;
            p2 = (d * p2 + pat[i]) % q2;
            t2 = (d * t2 + txt[i]) % q2;
        }

        for (int i = 0; i <= N - M; i++) {
            if (p1 == t1 && p2 == t2) result.push_back(i);
            if (i < N - M) {
                t1 = (d * (t1 - txt[i] * h1) + txt[i + M]) % q1;
                if (t1 < 0) t1 += q1;
                t2 = (d * (t2 - txt[i] * h2) + txt[i + M]) % q2;
                if (t2 < 0) t2 += q2;
            }
        }
        return result;
    }
};

void Test_Rabin_Karp() {
    Solution sol;
    struct TestCase { string txt, pat; };
    vector<TestCase> tests = {
        {"GEEKS FOR GEEKS", "GEEK"},
        {"AABAACAADAABAABA", "AABA"},
        {"ABABABAB", "ABA"},
        {"hello world", "world"}
    };

    for (auto& t : tests) {
        cout << "Text: \"" << t.txt << "\", Pattern: \"" << t.pat << "\"" << endl;

        auto r1 = sol.Rabin_Karp_Search(t.txt, t.pat);
        cout << "Rabin-Karp: ";
        for (int idx : r1) cout << idx << " ";
        cout << endl;

        auto r2 = sol.Naive_Search(t.txt, t.pat);
        cout << "Naive: ";
        for (int idx : r2) cout << idx << " ";
        cout << endl;

        auto r3 = sol.Rabin_Karp_Multiple_Hash(t.txt, t.pat);
        cout << "Double Hash: ";
        for (int idx : r3) cout << idx << " ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Rabin_Karp();
    return 0;
}
