"""
Problem: Isomorphic Strings
URL: https://practice.geeksforgeeks.org/problems/isomorphic-strings-1587115620/1

Problem Statement:
Two strings str1 and str2 are called isomorphic if there is a one-to-one mapping
possible for every character of str1 to every character of str2.

Sample Input/Output:
Input: str1 = "egg", str2 = "add"
Output: true (e->a, g->d)

Input: str1 = "foo", str2 = "bar"
Output: false
"""


class Solution:
    def Isomorphic_Two_Maps(self, str1, str2):
        """
        Two hashmaps for bidirectional mapping
        Time Complexity: O(n)
        Space Complexity: O(k) where k = unique chars
        """
        if len(str1) != len(str2):
            return False
        mp1 = {}
        mp2 = {}

        for i in range(len(str1)):
            if str1[i] not in mp1:
                mp1[str1[i]] = str2[i]
            if str2[i] not in mp2:
                mp2[str2[i]] = str1[i]

            if mp1[str1[i]] != str2[i] or mp2[str2[i]] != str1[i]:
                return False

        return True

    def Isomorphic_Array(self, str1, str2):
        """
        Using array for mapping + marked array
        Time Complexity: O(n)
        Space Complexity: O(1) - fixed 256
        """
        if len(str1) != len(str2):
            return False
        map1 = [-1] * 256
        map2 = [-1] * 256

        for i in range(len(str1)):
            if map1[ord(str1[i])] == -1 and map2[ord(str2[i])] == -1:
                map1[ord(str1[i])] = ord(str2[i])
                map2[ord(str2[i])] = ord(str1[i])
            elif (map1[ord(str1[i])] != ord(str2[i]) or
                  map2[ord(str2[i])] != ord(str1[i])):
                return False

        return True

    def Isomorphic_Transform(self, str1, str2):
        """
        Transform both strings to canonical form and compare
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if len(str1) != len(str2):
            return False

        def Transform(s):
            mp = {}
            result = ""
            counter = 0
            for c in s:
                if c not in mp:
                    mp[c] = counter
                    counter += 1
                result += str(mp[c]) + " "
            return result

        return Transform(str1) == Transform(str2)


def Test_Isomorphic_Strings():
    sol = Solution()
    tests = [
        ("egg", "add"),
        ("foo", "bar"),
        ("paper", "title"),
        ("ab", "aa"),
        ("abc", "abc"),
        ("", "")
    ]

    for s1, s2 in tests:
        print(f's1: "{s1}", s2: "{s2}"')
        print(f"Two Maps: {sol.Isomorphic_Two_Maps(s1, s2)}")
        print(f"Array: {sol.Isomorphic_Array(s1, s2)}")
        print(f"Transform: {sol.Isomorphic_Transform(s1, s2)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Isomorphic_Strings()
