"""
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
"""


class Solution:
    def Print_Sentences_Recursive(self, words, row, current, result):
        """
        Recursive backtracking
        Time Complexity: O(product of all list sizes)
        Space Complexity: O(R) recursion depth where R = number of word lists
        """
        if row == len(words):
            result.append(current[:])
            return
        for word in words[row]:
            if not word:
                continue
            current.append(word)
            self.Print_Sentences_Recursive(words, row + 1, current, result)
            current.pop()

    def Print_Sentences_Iterative(self, words):
        """
        Iterative - build sentences incrementally
        Time Complexity: O(product of all list sizes)
        Space Complexity: O(product of all list sizes)
        """
        result = [[]]
        for wordList in words:
            newResult = []
            for sentence in result:
                for word in wordList:
                    if not word:
                        continue
                    newSentence = sentence[:]
                    newSentence.append(word)
                    newResult.append(newSentence)
            result = newResult
        return result

    def Print_Sentences_Index(self, words):
        """
        Using index array to enumerate all combinations
        Time Complexity: O(product of all list sizes)
        Space Complexity: O(R) for index array
        """
        R = len(words)
        indices = [0] * R
        result = []

        while True:
            sentence = []
            for i in range(R):
                sentence.append(words[i][indices[i]])
            result.append(sentence)

            i = R - 1
            while i >= 0:
                indices[i] += 1
                if indices[i] < len(words[i]):
                    break
                indices[i] = 0
                i -= 1
            if i < 0:
                break

        return result


def Test_Print_All_Sentences():
    sol = Solution()
    words = [
        ["you", "we"],
        ["have", "are"],
        ["sleep", "eat", "drink"]
    ]

    print("=== Recursive ===")
    current = []
    r1 = []
    sol.Print_Sentences_Recursive(words, 0, current, r1)
    for sentence in r1:
        print(' '.join(sentence))

    print("=== Iterative ===")
    r2 = sol.Print_Sentences_Iterative(words)
    for sentence in r2:
        print(' '.join(sentence))

    print("=== Index ===")
    r3 = sol.Print_Sentences_Index(words)
    for sentence in r3:
        print(' '.join(sentence))

    print(f"Total sentences: {len(r1)}")


if __name__ == "__main__":
    Test_Print_All_Sentences()
