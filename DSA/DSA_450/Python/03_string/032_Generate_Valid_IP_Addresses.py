"""
Problem: Generate All Valid IP Addresses
URL: https://www.geeksforgeeks.org/program-generate-possible-valid-ip-addresses-given-string/

Problem Statement:
Given a string containing only digits, restore it by returning all possible
valid IP address combinations.

Sample Input/Output:
Input: "25525511135"
Output: ["255.255.11.135", "255.255.111.35"]

Input: "0000"
Output: ["0.0.0.0"]
"""


class Solution:
    def Generate_IP_Backtrack(self, s):
        """
        Backtracking - try placing dots at all valid positions
        Time Complexity: O(1) - max 27 combinations (3^3)
        Space Complexity: O(1)
        """
        result = []
        n = len(s)
        if n < 4 or n > 12:
            return result
        self.Backtrack(s, 0, 0, "", result)
        return result

    def Generate_IP_Three_Loops(self, s):
        """
        Three nested loops for three dot positions
        Time Complexity: O(n^3) but n <= 12
        Space Complexity: O(1) excluding result
        """
        result = []
        n = len(s)
        if n < 4 or n > 12:
            return result

        for i in range(1, min(4, n)):
            for j in range(i + 1, min(i + 4, n)):
                for k in range(j + 1, min(j + 4, n)):
                    p1 = s[:i]
                    p2 = s[i:j]
                    p3 = s[j:k]
                    p4 = s[k:]

                    if (self.Is_Valid_Part(p1) and self.Is_Valid_Part(p2) and
                        self.Is_Valid_Part(p3) and self.Is_Valid_Part(p4)):
                        result.append(p1 + "." + p2 + "." + p3 + "." + p4)

        return result

    def Is_Valid_Part(self, s):
        if not s or len(s) > 3:
            return False
        if len(s) > 1 and s[0] == '0':
            return False
        val = int(s)
        return 0 <= val <= 255

    def Backtrack(self, s, start, parts, current, result):
        if parts == 4:
            if start == len(s):
                result.append(current)
            return

        for length in range(1, min(4, len(s) - start + 1)):
            part = s[start:start + length]
            if not self.Is_Valid_Part(part):
                continue
            next_str = part if not current else current + "." + part
            self.Backtrack(s, start + length, parts + 1, next_str, result)


def Test_Generate_Valid_IP():
    sol = Solution()
    tests = ["25525511135", "0000", "1111", "101023", "255255255255"]

    for s in tests:
        print(f"Input: {s}")

        r1 = sol.Generate_IP_Backtrack(s)
        print(f"Backtrack ({len(r1)}): ", end="")
        for ip in r1:
            print(f"{ip} | ", end="")
        print()

        r2 = sol.Generate_IP_Three_Loops(s)
        print(f"Three Loops ({len(r2)}): ", end="")
        for ip in r2:
            print(f"{ip} | ", end="")
        print()

        print('-' * 50)


if __name__ == "__main__":
    Test_Generate_Valid_IP()
