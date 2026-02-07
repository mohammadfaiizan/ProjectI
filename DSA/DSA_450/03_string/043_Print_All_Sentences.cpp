/*
Problem: Recursively Print All Sentences from List of Word Lists
URL: https://www.geeksforgeeks.org/recursively-print-all-sentences-that-can-be-formed-from-list-of-word-lists/

Problem Statement:
Given a list of word lists, recursively print all possible sentences that can
be formed by taking one word from each list.

Sample Input/Output:
Input: {{"you", "we"}, {"have", "are"}, {"sleep", "eat", "drink"}}
Output:
  you have sleep
  you have eat
  you have drink
  you are sleep
  you are eat
  you are drink
  we have sleep
  ... (and so on)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Print_Sentences_Recursive(vector<vector<string>>& words, int row,
                                    vector<string>& current, vector<vector<string>>& result) {
        /*
        Recursive backtracking
        Time Complexity: O(product of all list sizes)
        Space Complexity: O(R) recursion depth where R = number of word lists
        */
        if (row == (int)words.size()) {
            result.push_back(current);
            return;
        }
        for (auto& word : words[row]) {
            if (word.empty()) continue;
            current.push_back(word);
            Print_Sentences_Recursive(words, row + 1, current, result);
            current.pop_back();
        }
    }

    vector<vector<string>> Print_Sentences_Iterative(vector<vector<string>>& words) {
        /*
        Iterative - build sentences incrementally
        Time Complexity: O(product of all list sizes)
        Space Complexity: O(product of all list sizes)
        */
        vector<vector<string>> result = {{}};
        for (auto& wordList : words) {
            vector<vector<string>> newResult;
            for (auto& sentence : result) {
                for (auto& word : wordList) {
                    if (word.empty()) continue;
                    vector<string> newSentence = sentence;
                    newSentence.push_back(word);
                    newResult.push_back(newSentence);
                }
            }
            result = newResult;
        }
        return result;
    }

    vector<vector<string>> Print_Sentences_Index(vector<vector<string>>& words) {
        /*
        Using index array to enumerate all combinations
        Time Complexity: O(product of all list sizes)
        Space Complexity: O(R) for index array
        */
        int R = words.size();
        vector<int> indices(R, 0);
        vector<vector<string>> result;

        while (true) {
            vector<string> sentence;
            for (int i = 0; i < R; i++)
                sentence.push_back(words[i][indices[i]]);
            result.push_back(sentence);

            int i = R - 1;
            while (i >= 0) {
                indices[i]++;
                if (indices[i] < (int)words[i].size()) break;
                indices[i] = 0;
                i--;
            }
            if (i < 0) break;
        }
        return result;
    }
};

void Test_Print_All_Sentences() {
    Solution sol;
    vector<vector<string>> words = {
        {"you", "we"},
        {"have", "are"},
        {"sleep", "eat", "drink"}
    };

    cout << "=== Recursive ===" << endl;
    vector<string> current;
    vector<vector<string>> r1;
    sol.Print_Sentences_Recursive(words, 0, current, r1);
    for (auto& sentence : r1) {
        for (auto& w : sentence) cout << w << " ";
        cout << endl;
    }

    cout << "=== Iterative ===" << endl;
    auto r2 = sol.Print_Sentences_Iterative(words);
    for (auto& sentence : r2) {
        for (auto& w : sentence) cout << w << " ";
        cout << endl;
    }

    cout << "=== Index ===" << endl;
    auto r3 = sol.Print_Sentences_Index(words);
    for (auto& sentence : r3) {
        for (auto& w : sentence) cout << w << " ";
        cout << endl;
    }

    cout << "Total sentences: " << r1.size() << endl;
}

int main() {
    Test_Print_All_Sentences();
    return 0;
}
