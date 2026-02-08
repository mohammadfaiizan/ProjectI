"""
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
"""


class Solution:
    def Count_And_Say_Recursive(self, n):
        """
        Recursive - get previous result and describe it
        Time Complexity: O(2^n) total characters generated
        Space Complexity: O(2^n)
        """
        if n == 1:
            return "1"
        prev = self.Count_And_Say_Recursive(n - 1)
        ans = ""
        i = 0
        while i < len(prev):
            c = prev[i]
            count = 0
            while i < len(prev) and prev[i] == c:
                count += 1
                i += 1
            ans += str(count) + c
        return ans

    def Count_And_Say_Iterative(self, n):
        """
        Iterative - build each term from the previous
        Time Complexity: O(2^n)
        Space Complexity: O(2^n)
        """
        result = "1"
        for k in range(2, n + 1):
            next_str = ""
            i = 0
            while i < len(result):
                c = result[i]
                count = 0
                while i < len(result) and result[i] == c:
                    count += 1
                    i += 1
                next_str += str(count) + c
            result = next_str
        return result


def Test_Count_And_Say():
    sol = Solution()

    for n in range(1, 9):
        r1 = sol.Count_And_Say_Recursive(n)
        r2 = sol.Count_And_Say_Iterative(n)
        match = " [MATCH]" if r1 == r2 else " [MISMATCH]"
        print(f"n={n} Recursive: {r1} Iterative: {r2}{match}")


if __name__ == "__main__":
    Test_Count_And_Say()
