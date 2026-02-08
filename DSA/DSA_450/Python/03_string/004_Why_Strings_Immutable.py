"""
Problem: Why are Strings Immutable in Java
URL: https://www.geeksforgeeks.org/java-string-is-immutable-what-exactly-is-the-meaning/
URL: https://www.javatpoint.com/immutable-string

Problem Statement:
Understand why strings are immutable in Java and how C++ handles strings differently.
In Java, String objects are immutable - once created, their value cannot be changed.
In C++, std::string is mutable - its content can be modified in place.

Key Points:
1. Java: String pool optimization - multiple references can share the same string object
2. Java: Thread safety - immutable objects are inherently thread-safe
3. Java: Security - prevents modification of sensitive data like DB connections, URLs
4. Java: Caching hashcode - since string is immutable, hashcode is cached at creation
5. C++: std::string is mutable, modification is done in-place
6. C++: For immutable behavior, use const string& or string_view (C++17)
"""


class Solution:
    def Demonstrate_String_Mutability(self):
        """
        Python strings are immutable unlike C++ strings
        Time Complexity: O(1) for individual char modification
        Space Complexity: O(n) for storing string
        """
        s = "hello"
        print(f"Original: {s}")

        s_list = list(s)
        s_list[0] = 'H'
        s = ''.join(s_list)
        print(f"After s[0]='H': {s}")

        s += " world"
        print(f"After append: {s}")

        s_list = list(s)
        s_list.insert(5, ',')
        s = ''.join(s_list)
        print(f"After insert: {s}")

        s = s[:5] + s[6:]
        print(f"After erase: {s}")

        s = s.replace("Hello", "Hi", 1)
        print(f"After replace: {s}")

    def Demonstrate_Const_Immutability(self):
        """
        Using const for immutable behavior in C++
        Time Complexity: O(1)
        Space Complexity: O(n)
        """
        s = "immutable"
        print(f"Const string: {s}")
        print(f"Length: {len(s)}")
        print(f"Char at 0: {s[0]}")

    def Demonstrate_Copy_On_Modify(self, s):
        """
        Simulating Java-like immutability by returning new string
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = s.upper()
        return result


def Test_Why_Strings_Immutable():
    sol = Solution()

    print("=== Python String Immutability ===")
    sol.Demonstrate_String_Mutability()
    print('-' * 50)

    print("=== Const Immutability ===")
    sol.Demonstrate_Const_Immutability()
    print('-' * 50)

    print("=== Copy On Modify ===")
    original = "hello"
    modified = sol.Demonstrate_Copy_On_Modify(original)
    print(f"Original: {original}")
    print(f"Modified: {modified}")
    print('-' * 50)


if __name__ == "__main__":
    Test_Why_Strings_Immutable()
