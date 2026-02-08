"""
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
"""


class Solution:
    def Keypad_Sequence_Array(self, input_str):
        """
        Using precomputed mapping array
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        arr = [
            "2", "22", "222",
            "3", "33", "333",
            "4", "44", "444",
            "5", "55", "555",
            "6", "66", "666",
            "7", "77", "777", "7777",
            "8", "88", "888",
            "9", "99", "999", "9999"
        ]

        output = ""
        for c in input_str:
            if c == ' ':
                output += "0"
            else:
                output += arr[ord(c) - ord('A')]
        return output

    def Keypad_Sequence_Map(self, input_str):
        """
        Using dictionary for mapping
        Time Complexity: O(n)
        Space Complexity: O(1) - fixed map size
        """
        mp = {
            'A': "2", 'B': "22", 'C': "222",
            'D': "3", 'E': "33", 'F': "333",
            'G': "4", 'H': "44", 'I': "444",
            'J': "5", 'K': "55", 'L': "555",
            'M': "6", 'N': "66", 'O': "666",
            'P': "7", 'Q': "77", 'R': "777", 'S': "7777",
            'T': "8", 'U': "88", 'V': "888",
            'W': "9", 'X': "99", 'Y': "999", 'Z': "9999",
            ' ': "0"
        }

        output = ""
        for c in input_str:
            output += mp[c]
        return output


def Test_Numeric_Keypad_Sequence():
    sol = Solution()
    tests = ["GEEKSFORGEEKS", "HELLO WORLD", "ABC", "HI THERE"]

    for s in tests:
        print(f"Input: {s}")
        print(f"Array: {sol.Keypad_Sequence_Array(s)}")
        print(f"Map: {sol.Keypad_Sequence_Map(s)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Numeric_Keypad_Sequence()
