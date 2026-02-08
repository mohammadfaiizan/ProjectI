/*
Problem: Print All Anagrams Together
URL: https://practice.geeksforgeeks.org/problems/print-anagrams-together/1

Problem Statement:
Given an array of strings, group all anagrams together.

Sample Input/Output:
Input: ["eat", "tea", "tan", "ate", "nat", "bat"]
Output: [["eat","tea","ate"], ["tan","nat"], ["bat"]]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<string>> Anagrams_Sort_Key(vector<string>& words) {
        /*
        Use sorted string as key in hashmap
        Time Complexity: O(n * k log k) where k = max word length
        Space Complexity: O(n * k)
        */
        unordered_map<string, vector<string>> mp;
        for (auto& word : words) {
            string key = word;
            sort(key.begin(), key.end());
            mp[key].push_back(word);
        }

        vector<vector<string>> result;
        for (auto& p : mp) result.push_back(p.second);
        return result;
    }

    vector<vector<string>> Anagrams_Count_Key(vector<string>& words) {
        /*
        Use character count as key (frequency string)
        Time Complexity: O(n * k) where k = max word length
        Space Complexity: O(n * k)
        */
        unordered_map<string, vector<string>> mp;
        for (auto& word : words) {
            int count[26] = {0};
            for (char c : word) count[c - 'a']++;
            string key = "";
            for (int i = 0; i < 26; i++) key += "#" + to_string(count[i]);
            mp[key].push_back(word);
        }

        vector<vector<string>> result;
        for (auto& p : mp) result.push_back(p.second);
        return result;
    }

    vector<vector<string>> Anagrams_Prime_Hash(vector<string>& words) {
        /*
        Map each char to a prime number, product as key
        Time Complexity: O(n * k)
        Space Complexity: O(n)
        */
        int primes[26] = {2,3,5,7,11,13,17,19,23,29,31,37,41,
                          43,47,53,59,61,67,71,73,79,83,89,97,101};
        unordered_map<long long, vector<string>> mp;
        for (auto& word : words) {
            long long hash = 1;
            for (char c : word) hash *= primes[c - 'a'];
            mp[hash].push_back(word);
        }

        vector<vector<string>> result;
        for (auto& p : mp) result.push_back(p.second);
        return result;
    }
};

void Test_Print_All_Anagrams() {
    Solution sol;
    vector<string> words = {"eat", "tea", "tan", "ate", "nat", "bat"};

    cout << "Input: ";
    for (auto& w : words) cout << w << " ";
    cout << endl;

    auto r1 = sol.Anagrams_Sort_Key(words);
    cout << "Sort Key:" << endl;
    for (auto& group : r1) {
        cout << "  [";
        for (int i = 0; i < (int)group.size(); i++) {
            cout << group[i] << (i < (int)group.size()-1 ? ", " : "");
        }
        cout << "]" << endl;
    }

    auto r2 = sol.Anagrams_Count_Key(words);
    cout << "Count Key:" << endl;
    for (auto& group : r2) {
        cout << "  [";
        for (int i = 0; i < (int)group.size(); i++) {
            cout << group[i] << (i < (int)group.size()-1 ? ", " : "");
        }
        cout << "]" << endl;
    }

    auto r3 = sol.Anagrams_Prime_Hash(words);
    cout << "Prime Hash:" << endl;
    for (auto& group : r3) {
        cout << "  [";
        for (int i = 0; i < (int)group.size(); i++) {
            cout << group[i] << (i < (int)group.size()-1 ? ", " : "");
        }
        cout << "]" << endl;
    }
}

int main() {
    Test_Print_All_Anagrams();
    return 0;
}
