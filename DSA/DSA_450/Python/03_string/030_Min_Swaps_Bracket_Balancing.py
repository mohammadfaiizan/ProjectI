"""
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
"""


class Solution:
    def Min_Swaps_Position_Track(self, s):
        """
        Track positions of '[' and swap when imbalanced
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        pos = []
        for i in range(len(s)):
            if s[i] == '[':
                pos.append(i)

        s_list = list(s)
        count = p = 0
        sum_val = 0
        for i in range(len(s_list)):
            if s_list[i] == '[':
                count += 1
                p += 1
            elif s_list[i] == ']':
                count -= 1

            if count < 0:
                sum_val += pos[p] - i
                s_list[i], s_list[pos[p]] = s_list[pos[p]], s_list[i]
                p += 1
                count = 1

        return sum_val

    def Min_Swaps_Imbalance_Counter(self, s):
        """
        Counter approach tracking imbalance
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        countLeft = countRight = 0
        ans = imbalance = 0

        for i in range(len(s)):
            if s[i] == '[':
                countLeft += 1
                if imbalance > 0:
                    ans += imbalance
                    imbalance -= 1
            elif s[i] == ']':
                countRight += 1
                imbalance = countRight - countLeft

        return ans


def Test_Min_Swaps_Bracket_Balancing():
    sol = Solution()
    tests = ["[]][][", "[[][]]", "][][", "[[[]]]", "][]["]

    for s in tests:
        print(f"Input: {s}")
        s1 = list(s)
        print(f"Position Track: {sol.Min_Swaps_Position_Track(''.join(s1))}")
        print(f"Imbalance Counter: {sol.Min_Swaps_Imbalance_Counter(s)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Min_Swaps_Bracket_Balancing()
