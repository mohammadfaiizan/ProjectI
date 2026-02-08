/*
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
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Demonstrate_String_Mutability() {
        /*
        C++ strings are mutable unlike Java strings
        Time Complexity: O(1) for individual char modification
        Space Complexity: O(n) for storing string
        */
        string s = "hello";
        cout << "Original: " << s << endl;

        s[0] = 'H';
        cout << "After s[0]='H': " << s << endl;

        s += " world";
        cout << "After append: " << s << endl;

        s.insert(5, ",");
        cout << "After insert: " << s << endl;

        s.erase(5, 1);
        cout << "After erase: " << s << endl;

        s.replace(0, 5, "Hi");
        cout << "After replace: " << s << endl;
    }

    void Demonstrate_Const_Immutability() {
        /*
        Using const for immutable behavior in C++
        Time Complexity: O(1)
        Space Complexity: O(n)
        */
        const string s = "immutable";
        cout << "Const string: " << s << endl;
        cout << "Length: " << s.size() << endl;
        cout << "Char at 0: " << s[0] << endl;
    }

    string Demonstrate_Copy_On_Modify(string s) {
        /*
        Simulating Java-like immutability by returning new string
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        string result = s;
        for (char& c : result) c = toupper(c);
        return result;
    }
};

void Test_Why_Strings_Immutable() {
    Solution sol;

    cout << "=== C++ String Mutability ===" << endl;
    sol.Demonstrate_String_Mutability();
    cout << string(50, '-') << endl;

    cout << "=== Const Immutability ===" << endl;
    sol.Demonstrate_Const_Immutability();
    cout << string(50, '-') << endl;

    cout << "=== Copy On Modify ===" << endl;
    string original = "hello";
    string modified = sol.Demonstrate_Copy_On_Modify(original);
    cout << "Original: " << original << endl;
    cout << "Modified: " << modified << endl;
    cout << string(50, '-') << endl;
}

int main() {
    Test_Why_Strings_Immutable();
    return 0;
}
