/*
Problem: Convert Sentence to Numeric Keypad Sequence
URL: https://www.geeksforgeeks.org/convert-sentence-equivalent-mobile-numeric-keypad-sequence/

Problem Statement:
Given a sentence in the form of a string, convert it into its equivalent
mobile numeric keypad sequence. (Old phone keypad: ABC=2, DEF=3, ...)

Sample Input/Output:
Input: "GEEKSFORGEEKS"
Output: "4333355777733366677743333557777"

Input: "HELLO WORLD"
Output: "4433555555666096667775553"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Keypad_Sequence_Array(string input) {
        /*
        Using precomputed mapping array
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        string arr[] = {
            "2","22","222",
            "3","33","333",
            "4","44","444",
            "5","55","555",
            "6","66","666",
            "7","77","777","7777",
            "8","88","888",
            "9","99","999","9999"
        };

        string output = "";
        for (char c : input) {
            if (c == ' ')
                output += "0";
            else
                output += arr[c - 'A'];
        }
        return output;
    }

    string Keypad_Sequence_Map(string input) {
        /*
        Using unordered_map for mapping
        Time Complexity: O(n)
        Space Complexity: O(1) - fixed map size
        */
        unordered_map<char, string> mp = {
            {'A',"2"}, {'B',"22"}, {'C',"222"},
            {'D',"3"}, {'E',"33"}, {'F',"333"},
            {'G',"4"}, {'H',"44"}, {'I',"444"},
            {'J',"5"}, {'K',"55"}, {'L',"555"},
            {'M',"6"}, {'N',"66"}, {'O',"666"},
            {'P',"7"}, {'Q',"77"}, {'R',"777"}, {'S',"7777"},
            {'T',"8"}, {'U',"88"}, {'V',"888"},
            {'W',"9"}, {'X',"99"}, {'Y',"999"}, {'Z',"9999"},
            {' ',"0"}
        };

        string output = "";
        for (char c : input) output += mp[c];
        return output;
    }
};

void Test_Numeric_Keypad_Sequence() {
    Solution sol;
    vector<string> tests = {"GEEKSFORGEEKS", "HELLO WORLD", "ABC", "HI THERE"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;
        cout << "Array: " << sol.Keypad_Sequence_Array(s) << endl;
        cout << "Map: " << sol.Keypad_Sequence_Map(s) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Numeric_Keypad_Sequence();
    return 0;
}
