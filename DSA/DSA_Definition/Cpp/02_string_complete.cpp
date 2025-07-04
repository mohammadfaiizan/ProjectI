/*
 * =============================================================================
 * COMPLETE STRING GUIDE - All Operations & Algorithms
 * =============================================================================
 * 
 * This file covers:
 * 1. Different ways to initialize and create strings
 * 2. All string operations (access, modify, search, compare)
 * 3. String algorithms (palindrome, anagram, pattern matching)
 * 4. String manipulation techniques
 * 5. STL string functions and algorithms
 * 6. String parsing and processing
 * 
 * =============================================================================
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <unordered_map>
#include <unordered_set>
#include <regex>
using namespace std;

// =============================================================================
// STRING INITIALIZATION AND CREATION
// =============================================================================

void demonstrate_string_initialization() {
    cout << "\n=== STRING INITIALIZATION METHODS ===" << endl;
    
    // Method 1: Direct initialization
    string str1 = "Hello World";
    cout << "Method 1 - Direct: \"" << str1 << "\"" << endl;
    
    // Method 2: Constructor with repetition
    string str2(5, 'A');
    cout << "Method 2 - Repetition: \"" << str2 << "\"" << endl;
    
    // Method 3: Copy constructor
    string str3(str1);
    cout << "Method 3 - Copy: \"" << str3 << "\"" << endl;
    
    // Method 4: Substring constructor
    string str4(str1, 6, 5);
    cout << "Method 4 - Substring(6,5): \"" << str4 << "\"" << endl;
    
    // Method 5: From C-string
    const char* c_str = "C-Style String";
    string str5(c_str);
    cout << "Method 5 - From C-string: \"" << str5 << "\"" << endl;
    
    // Method 6: From character array
    char char_array[] = {'H', 'e', 'l', 'l', 'o', '\0'};
    string str6(char_array);
    cout << "Method 6 - From char array: \"" << str6 << "\"" << endl;
    
    // Method 7: Using assign
    string str7;
    str7.assign("Assigned String");
    cout << "Method 7 - Using assign: \"" << str7 << "\"" << endl;
    
    // Method 8: From vector of chars
    vector<char> char_vec = {'V', 'e', 'c', 't', 'o', 'r'};
    string str8(char_vec.begin(), char_vec.end());
    cout << "Method 8 - From vector: \"" << str8 << "\"" << endl;
    
    // Method 9: Using string literal
    string str9 = R"(Raw string literal with "quotes" and \backslashes)";
    cout << "Method 9 - Raw literal: \"" << str9 << "\"" << endl;
    
    // Method 10: Empty string
    string str10;
    cout << "Method 10 - Empty string length: " << str10.length() << endl;
}

// =============================================================================
// STRING ACCESS AND MODIFICATION
// =============================================================================

void demonstrate_string_access() {
    cout << "\n=== STRING ACCESS & MODIFICATION ===" << endl;
    
    string str = "Programming";
    cout << "Original string: \"" << str << "\"" << endl;
    
    // Different access methods
    cout << "\n--- Access Methods ---" << endl;
    cout << "str[0] = '" << str[0] << "'" << endl;           // No bounds checking
    cout << "str.at(1) = '" << str.at(1) << "'" << endl;     // With bounds checking
    cout << "str.front() = '" << str.front() << "'" << endl; // First character
    cout << "str.back() = '" << str.back() << "'" << endl;   // Last character
    
    // Using iterators
    cout << "\n--- Iterator Access ---" << endl;
    cout << "*(str.begin()) = '" << *(str.begin()) << "'" << endl;
    cout << "*(str.end()-1) = '" << *(str.end()-1) << "'" << endl;
    
    // Modification methods
    cout << "\n--- Modification Methods ---" << endl;
    
    // Direct assignment
    str[0] = 'p';
    cout << "After str[0] = 'p': \"" << str << "\"" << endl;
    
    // Using at() for safe modification
    str.at(1) = 'R';
    cout << "After str.at(1) = 'R': \"" << str << "\"" << endl;
    
    // Append operations
    str += " Language";
    cout << "After += \" Language\": \"" << str << "\"" << endl;
    
    str.append(" C++");
    cout << "After append(\" C++\"): \"" << str << "\"" << endl;
    
    // Insert operations
    str.insert(0, "Modern ");
    cout << "After insert(0, \"Modern \"): \"" << str << "\"" << endl;
    
    str.insert(str.find("C++"), "Beautiful ");
    cout << "After inserting \"Beautiful \" before \"C++\": \"" << str << "\"" << endl;
    
    // Replace operations
    str.replace(str.find("pRogramming"), 11, "Programming");
    cout << "After replace fix: \"" << str << "\"" << endl;
    
    // Erase operations
    string temp_str = str;
    temp_str.erase(0, 7);  // Remove "Modern "
    cout << "After erase(0, 7): \"" << temp_str << "\"" << endl;
    
    temp_str.erase(temp_str.find(" Beautiful"), 10);
    cout << "After erasing \" Beautiful\": \"" << temp_str << "\"" << endl;
}

// =============================================================================
// STRING SEARCH AND FIND OPERATIONS
// =============================================================================

void demonstrate_string_search() {
    cout << "\n=== STRING SEARCH OPERATIONS ===" << endl;
    
    string text = "The quick brown fox jumps over the lazy dog";
    cout << "Text: \"" << text << "\"" << endl;
    
    // Basic find operations
    cout << "\n--- Basic Find Operations ---" << endl;
    
    size_t pos = text.find("fox");
    cout << "find(\"fox\"): " << (pos != string::npos ? to_string(pos) : "not found") << endl;
    
    pos = text.find("cat");
    cout << "find(\"cat\"): " << (pos != string::npos ? to_string(pos) : "not found") << endl;
    
    pos = text.find('o');
    cout << "find('o'): " << pos << endl;
    
    pos = text.find('o', 15);  // Start search from position 15
    cout << "find('o', 15): " << pos << endl;
    
    // Reverse find operations
    cout << "\n--- Reverse Find Operations ---" << endl;
    
    pos = text.rfind('o');
    cout << "rfind('o'): " << pos << endl;
    
    pos = text.rfind("the");
    cout << "rfind(\"the\"): " << pos << endl;
    
    // Find first/last of character sets
    cout << "\n--- Character Set Operations ---" << endl;
    
    pos = text.find_first_of("aeiou");
    cout << "find_first_of(\"aeiou\"): " << pos << " ('" << text[pos] << "')" << endl;
    
    pos = text.find_last_of("aeiou");
    cout << "find_last_of(\"aeiou\"): " << pos << " ('" << text[pos] << "')" << endl;
    
    pos = text.find_first_not_of("The ");
    cout << "find_first_not_of(\"The \"): " << pos << " ('" << text[pos] << "')" << endl;
    
    pos = text.find_last_not_of("dog ");
    cout << "find_last_not_of(\"dog \"): " << pos << " ('" << text[pos] << "')" << endl;
    
    // Count occurrences
    cout << "\n--- Count Occurrences ---" << endl;
    
    int count = 0;
    pos = 0;
    while ((pos = text.find('o', pos)) != string::npos) {
        count++;
        pos++;
    }
    cout << "Count of 'o': " << count << endl;
    
    // Find all occurrences of a substring
    vector<size_t> positions;
    pos = 0;
    string target = "the";
    while ((pos = text.find(target, pos)) != string::npos) {
        positions.push_back(pos);
        pos += target.length();
    }
    cout << "Positions of \"the\": ";
    for (size_t p : positions) cout << p << " ";
    cout << endl;
}

// =============================================================================
// STRING COMPARISON AND SORTING
// =============================================================================

void demonstrate_string_comparison() {
    cout << "\n=== STRING COMPARISON OPERATIONS ===" << endl;
    
    string str1 = "apple";
    string str2 = "banana";
    string str3 = "apple";
    string str4 = "Apple";
    
    cout << "str1: \"" << str1 << "\"" << endl;
    cout << "str2: \"" << str2 << "\"" << endl;
    cout << "str3: \"" << str3 << "\"" << endl;
    cout << "str4: \"" << str4 << "\"" << endl;
    
    // Basic comparison operators
    cout << "\n--- Comparison Operators ---" << endl;
    cout << "str1 == str2: " << (str1 == str2 ? "true" : "false") << endl;
    cout << "str1 == str3: " << (str1 == str3 ? "true" : "false") << endl;
    cout << "str1 != str2: " << (str1 != str2 ? "true" : "false") << endl;
    cout << "str1 < str2: " << (str1 < str2 ? "true" : "false") << endl;
    cout << "str1 > str2: " << (str1 > str2 ? "true" : "false") << endl;
    
    // Compare method
    cout << "\n--- Compare Method ---" << endl;
    int result = str1.compare(str2);
    cout << "str1.compare(str2): " << result << " (";
    if (result < 0) cout << "str1 < str2";
    else if (result > 0) cout << "str1 > str2";
    else cout << "str1 == str2";
    cout << ")" << endl;
    
    result = str1.compare(str3);
    cout << "str1.compare(str3): " << result << " (equal)" << endl;
    
    // Case-insensitive comparison
    cout << "\n--- Case-Insensitive Comparison ---" << endl;
    
    auto to_lower = [](string s) {
        transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    };
    
    cout << "Case-insensitive str1 == str4: " 
         << (to_lower(str1) == to_lower(str4) ? "true" : "false") << endl;
    
    // Sorting strings
    cout << "\n--- Sorting Strings ---" << endl;
    vector<string> words = {"banana", "apple", "cherry", "date", "elderberry"};
    cout << "Before sorting: ";
    for (const string& word : words) cout << word << " ";
    cout << endl;
    
    sort(words.begin(), words.end());
    cout << "After sorting: ";
    for (const string& word : words) cout << word << " ";
    cout << endl;
    
    // Custom sorting (by length)
    sort(words.begin(), words.end(), [](const string& a, const string& b) {
        return a.length() < b.length();
    });
    cout << "Sorted by length: ";
    for (const string& word : words) cout << word << " ";
    cout << endl;
}

// =============================================================================
// STRING MANIPULATION AND PROCESSING
// =============================================================================

void demonstrate_string_manipulation() {
    cout << "\n=== STRING MANIPULATION ===" << endl;
    
    // Substring operations
    cout << "\n--- Substring Operations ---" << endl;
    string text = "Hello, World! How are you?";
    cout << "Original: \"" << text << "\"" << endl;
    
    string sub1 = text.substr(7, 5);  // From position 7, length 5
    cout << "substr(7, 5): \"" << sub1 << "\"" << endl;
    
    string sub2 = text.substr(7);     // From position 7 to end
    cout << "substr(7): \"" << sub2 << "\"" << endl;
    
    // Case conversion
    cout << "\n--- Case Conversion ---" << endl;
    string mixed_case = "Hello World";
    string upper_case = mixed_case;
    string lower_case = mixed_case;
    
    transform(upper_case.begin(), upper_case.end(), upper_case.begin(), ::toupper);
    transform(lower_case.begin(), lower_case.end(), lower_case.begin(), ::tolower);
    
    cout << "Original: \"" << mixed_case << "\"" << endl;
    cout << "Upper: \"" << upper_case << "\"" << endl;
    cout << "Lower: \"" << lower_case << "\"" << endl;
    
    // Reverse string
    cout << "\n--- String Reversal ---" << endl;
    string original = "Programming";
    string reversed = original;
    reverse(reversed.begin(), reversed.end());
    cout << "Original: \"" << original << "\"" << endl;
    cout << "Reversed: \"" << reversed << "\"" << endl;
    
    // Trim whitespace
    cout << "\n--- Trimming Whitespace ---" << endl;
    string whitespace_str = "   Hello World   ";
    cout << "With whitespace: \"" << whitespace_str << "\"" << endl;
    
    // Left trim
    string left_trimmed = whitespace_str;
    left_trimmed.erase(left_trimmed.begin(), 
                      find_if(left_trimmed.begin(), left_trimmed.end(), 
                              [](unsigned char ch) { return !isspace(ch); }));
    cout << "Left trimmed: \"" << left_trimmed << "\"" << endl;
    
    // Right trim
    string right_trimmed = whitespace_str;
    right_trimmed.erase(find_if(right_trimmed.rbegin(), right_trimmed.rend(),
                               [](unsigned char ch) { return !isspace(ch); }).base(),
                       right_trimmed.end());
    cout << "Right trimmed: \"" << right_trimmed << "\"" << endl;
    
    // Both sides trim
    string trimmed = whitespace_str;
    trimmed.erase(trimmed.begin(), 
                 find_if(trimmed.begin(), trimmed.end(), 
                         [](unsigned char ch) { return !isspace(ch); }));
    trimmed.erase(find_if(trimmed.rbegin(), trimmed.rend(),
                         [](unsigned char ch) { return !isspace(ch); }).base(),
                 trimmed.end());
    cout << "Both trimmed: \"" << trimmed << "\"" << endl;
    
    // String replacement
    cout << "\n--- String Replacement ---" << endl;
    string replace_text = "The cat in the hat";
    cout << "Original: \"" << replace_text << "\"" << endl;
    
    // Replace first occurrence
    size_t pos = replace_text.find("cat");
    if (pos != string::npos) {
        replace_text.replace(pos, 3, "dog");
    }
    cout << "Replace 'cat' with 'dog': \"" << replace_text << "\"" << endl;
    
    // Replace all occurrences
    string replace_all = "The cat and the cat are cats";
    cout << "Original: \"" << replace_all << "\"" << endl;
    
    string target = "cat";
    string replacement = "dog";
    pos = 0;
    while ((pos = replace_all.find(target, pos)) != string::npos) {
        replace_all.replace(pos, target.length(), replacement);
        pos += replacement.length();
    }
    cout << "Replace all 'cat' with 'dog': \"" << replace_all << "\"" << endl;
}

// =============================================================================
// STRING SPLITTING AND JOINING
// =============================================================================

void demonstrate_string_splitting() {
    cout << "\n=== STRING SPLITTING & JOINING ===" << endl;
    
    // Split by delimiter
    cout << "\n--- Splitting Strings ---" << endl;
    
    string csv_data = "apple,banana,cherry,date,elderberry";
    cout << "CSV data: \"" << csv_data << "\"" << endl;
    
    // Method 1: Using stringstream
    vector<string> tokens1;
    stringstream ss(csv_data);
    string token;
    
    while (getline(ss, token, ',')) {
        tokens1.push_back(token);
    }
    
    cout << "Split by comma (stringstream): ";
    for (const string& t : tokens1) cout << "\"" << t << "\" ";
    cout << endl;
    
    // Method 2: Manual splitting
    vector<string> tokens2;
    size_t start = 0;
    size_t end = csv_data.find(',');
    
    while (end != string::npos) {
        tokens2.push_back(csv_data.substr(start, end - start));
        start = end + 1;
        end = csv_data.find(',', start);
    }
    tokens2.push_back(csv_data.substr(start));  // Last token
    
    cout << "Split by comma (manual): ";
    for (const string& t : tokens2) cout << "\"" << t << "\" ";
    cout << endl;
    
    // Split by whitespace
    string sentence = "The quick brown fox jumps";
    cout << "\nSentence: \"" << sentence << "\"" << endl;
    
    vector<string> words;
    stringstream word_stream(sentence);
    string word;
    
    while (word_stream >> word) {
        words.push_back(word);
    }
    
    cout << "Split by whitespace: ";
    for (const string& w : words) cout << "\"" << w << "\" ";
    cout << endl;
    
    // Split by multiple delimiters
    cout << "\n--- Split by Multiple Delimiters ---" << endl;
    string multi_delim = "apple;banana,cherry:date|elderberry";
    cout << "Multi-delimiter string: \"" << multi_delim << "\"" << endl;
    
    vector<string> multi_tokens;
    string delimiters = ";,:| ";
    start = 0;
    
    while (start < multi_delim.length()) {
        end = multi_delim.find_first_of(delimiters, start);
        if (end == string::npos) {
            multi_tokens.push_back(multi_delim.substr(start));
            break;
        }
        if (end != start) {  // Skip empty tokens
            multi_tokens.push_back(multi_delim.substr(start, end - start));
        }
        start = end + 1;
    }
    
    cout << "Split by multiple delimiters: ";
    for (const string& t : multi_tokens) cout << "\"" << t << "\" ";
    cout << endl;
    
    // Joining strings
    cout << "\n--- Joining Strings ---" << endl;
    
    vector<string> fruits = {"apple", "banana", "cherry", "date"};
    
    // Method 1: Simple concatenation
    string joined1;
    for (size_t i = 0; i < fruits.size(); i++) {
        joined1 += fruits[i];
        if (i < fruits.size() - 1) joined1 += ", ";
    }
    cout << "Joined with \", \": \"" << joined1 << "\"" << endl;
    
    // Method 2: Using stringstream
    stringstream join_stream;
    for (size_t i = 0; i < fruits.size(); i++) {
        join_stream << fruits[i];
        if (i < fruits.size() - 1) join_stream << " | ";
    }
    string joined2 = join_stream.str();
    cout << "Joined with \" | \": \"" << joined2 << "\"" << endl;
    
    // Method 3: Custom join function
    auto join = [](const vector<string>& vec, const string& delimiter) {
        if (vec.empty()) return string();
        
        string result = vec[0];
        for (size_t i = 1; i < vec.size(); i++) {
            result += delimiter + vec[i];
        }
        return result;
    };
    
    string joined3 = join(fruits, " -> ");
    cout << "Joined with \" -> \": \"" << joined3 << "\"" << endl;
}

// =============================================================================
// STRING ALGORITHMS
// =============================================================================

class StringAlgorithms {
public:
    // Check if string is palindrome
    static bool isPalindrome(const string& str) {
        int left = 0, right = str.length() - 1;
        
        while (left < right) {
            if (tolower(str[left]) != tolower(str[right])) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
    
    // Check if two strings are anagrams
    static bool areAnagrams(string str1, string str2) {
        if (str1.length() != str2.length()) {
            return false;
        }
        
        sort(str1.begin(), str1.end());
        sort(str2.begin(), str2.end());
        
        return str1 == str2;
    }
    
    // Check if two strings are anagrams using frequency counting
    static bool areAnagramsFreq(const string& str1, const string& str2) {
        if (str1.length() != str2.length()) {
            return false;
        }
        
        unordered_map<char, int> freq;
        
        for (char c : str1) {
            freq[c]++;
        }
        
        for (char c : str2) {
            freq[c]--;
            if (freq[c] < 0) {
                return false;
            }
        }
        
        return true;
    }
    
    // Find longest palindromic substring
    static string longestPalindrome(const string& str) {
        if (str.empty()) return "";
        
        int start = 0, max_len = 1;
        
        auto expandAroundCenter = [&](int left, int right) {
            while (left >= 0 && right < str.length() && str[left] == str[right]) {
                int current_len = right - left + 1;
                if (current_len > max_len) {
                    start = left;
                    max_len = current_len;
                }
                left--;
                right++;
            }
        };
        
        for (int i = 0; i < str.length(); i++) {
            expandAroundCenter(i, i);      // Odd length palindromes
            expandAroundCenter(i, i + 1);  // Even length palindromes
        }
        
        return str.substr(start, max_len);
    }
    
    // Count character frequency
    static unordered_map<char, int> characterFrequency(const string& str) {
        unordered_map<char, int> freq;
        for (char c : str) {
            freq[c]++;
        }
        return freq;
    }
    
    // Find first non-repeating character
    static char firstNonRepeatingChar(const string& str) {
        unordered_map<char, int> freq;
        
        for (char c : str) {
            freq[c]++;
        }
        
        for (char c : str) {
            if (freq[c] == 1) {
                return c;
            }
        }
        
        return '\0';  // No non-repeating character found
    }
    
    // KMP pattern matching algorithm
    static vector<int> KMPSearch(const string& text, const string& pattern) {
        vector<int> matches;
        if (pattern.empty()) return matches;
        
        // Build failure function
        vector<int> failure(pattern.length(), 0);
        int j = 0;
        for (int i = 1; i < pattern.length(); i++) {
            while (j > 0 && pattern[i] != pattern[j]) {
                j = failure[j - 1];
            }
            if (pattern[i] == pattern[j]) {
                j++;
            }
            failure[i] = j;
        }
        
        // Search for pattern
        j = 0;
        for (int i = 0; i < text.length(); i++) {
            while (j > 0 && text[i] != pattern[j]) {
                j = failure[j - 1];
            }
            if (text[i] == pattern[j]) {
                j++;
            }
            if (j == pattern.length()) {
                matches.push_back(i - j + 1);
                j = failure[j - 1];
            }
        }
        
        return matches;
    }
    
    // Longest common subsequence
    static string longestCommonSubsequence(const string& str1, const string& str2) {
        int m = str1.length(), n = str2.length();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
        
        // Build DP table
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str1[i - 1] == str2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        // Reconstruct LCS
        string lcs;
        int i = m, j = n;
        while (i > 0 && j > 0) {
            if (str1[i - 1] == str2[j - 1]) {
                lcs = str1[i - 1] + lcs;
                i--;
                j--;
            } else if (dp[i - 1][j] > dp[i][j - 1]) {
                i--;
            } else {
                j--;
            }
        }
        
        return lcs;
    }
    
    // Edit distance (Levenshtein distance)
    static int editDistance(const string& str1, const string& str2) {
        int m = str1.length(), n = str2.length();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1));
        
        // Initialize base cases
        for (int i = 0; i <= m; i++) dp[i][0] = i;
        for (int j = 0; j <= n; j++) dp[0][j] = j;
        
        // Fill DP table
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str1[i - 1] == str2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + min({dp[i - 1][j],      // Deletion
                                        dp[i][j - 1],      // Insertion
                                        dp[i - 1][j - 1]}); // Substitution
                }
            }
        }
        
        return dp[m][n];
    }
};

// =============================================================================
// REGULAR EXPRESSIONS
// =============================================================================

void demonstrate_regex() {
    cout << "\n=== REGULAR EXPRESSIONS ===" << endl;
    
    string text = "Contact us at: john@example.com or call 123-456-7890";
    cout << "Text: \"" << text << "\"" << endl;
    
    // Email validation
    cout << "\n--- Email Validation ---" << endl;
    regex email_pattern(R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})");
    
    if (regex_search(text, email_pattern)) {
        cout << "Email found in text" << endl;
        
        smatch match;
        regex_search(text, match, email_pattern);
        cout << "Email: " << match[0] << endl;
    }
    
    // Phone number extraction
    cout << "\n--- Phone Number Extraction ---" << endl;
    regex phone_pattern(R"(\d{3}-\d{3}-\d{4})");
    
    smatch phone_match;
    if (regex_search(text, phone_match, phone_pattern)) {
        cout << "Phone number: " << phone_match[0] << endl;
    }
    
    // Find all matches
    cout << "\n--- Find All Numbers ---" << endl;
    regex number_pattern(R"(\d+)");
    
    sregex_iterator numbers_begin(text.begin(), text.end(), number_pattern);
    sregex_iterator numbers_end;
    
    cout << "All numbers found: ";
    for (auto it = numbers_begin; it != numbers_end; ++it) {
        cout << it->str() << " ";
    }
    cout << endl;
    
    // String replacement with regex
    cout << "\n--- Regex Replacement ---" << endl;
    string sensitive_text = "My SSN is 123-45-6789 and credit card is 1234-5678-9012-3456";
    cout << "Original: " << sensitive_text << endl;
    
    // Replace SSN pattern
    regex ssn_pattern(R"(\d{3}-\d{2}-\d{4})");
    string masked_ssn = regex_replace(sensitive_text, ssn_pattern, "XXX-XX-XXXX");
    cout << "SSN masked: " << masked_ssn << endl;
    
    // Replace credit card pattern
    regex cc_pattern(R"(\d{4}-\d{4}-\d{4}-\d{4})");
    string fully_masked = regex_replace(masked_ssn, cc_pattern, "XXXX-XXXX-XXXX-XXXX");
    cout << "Fully masked: " << fully_masked << endl;
}

// =============================================================================
// STRING PERFORMANCE AND MEMORY
// =============================================================================

void demonstrate_string_performance() {
    cout << "\n=== STRING PERFORMANCE & MEMORY ===" << endl;
    
    string str = "Hello";
    cout << "Original string: \"" << str << "\"" << endl;
    cout << "Length: " << str.length() << endl;
    cout << "Size: " << str.size() << endl;
    cout << "Capacity: " << str.capacity() << endl;
    cout << "Max size: " << str.max_size() << endl;
    
    // Reserve capacity
    cout << "\n--- Reserve Capacity ---" << endl;
    str.reserve(100);
    cout << "After reserve(100) - Capacity: " << str.capacity() << endl;
    
    // Efficient string building
    cout << "\n--- Efficient String Building ---" << endl;
    
    // Method 1: Inefficient - multiple concatenations
    string inefficient;
    for (int i = 0; i < 5; i++) {
        inefficient += "Word" + to_string(i) + " ";
    }
    cout << "Inefficient building: \"" << inefficient << "\"" << endl;
    
    // Method 2: Efficient - reserve space first
    string efficient;
    efficient.reserve(50);  // Pre-allocate space
    for (int i = 0; i < 5; i++) {
        efficient += "Word" + to_string(i) + " ";
    }
    cout << "Efficient building: \"" << efficient << "\"" << endl;
    
    // Method 3: Using stringstream
    stringstream ss;
    for (int i = 0; i < 5; i++) {
        ss << "Word" << i << " ";
    }
    string stream_built = ss.str();
    cout << "Stream building: \"" << stream_built << "\"" << endl;
    
    // Shrink to fit
    cout << "\n--- Shrink to Fit ---" << endl;
    string large_str(1000, 'A');
    cout << "Large string capacity: " << large_str.capacity() << endl;
    
    large_str.resize(10);
    cout << "After resize(10) - Capacity: " << large_str.capacity() << endl;
    
    large_str.shrink_to_fit();
    cout << "After shrink_to_fit() - Capacity: " << large_str.capacity() << endl;
    
    // String views (C++17)
    cout << "\n--- String Views (C++17) ---" << endl;
    string original_text = "This is a long string for demonstration";
    string_view view1(original_text.data() + 5, 4);  // "is a"
    string_view view2(original_text.data() + 10, 4); // "long"
    
    cout << "Original: \"" << original_text << "\"" << endl;
    cout << "View 1: \"" << view1 << "\"" << endl;
    cout << "View 2: \"" << view2 << "\"" << endl;
}

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_string_algorithms() {
    cout << "\n=== STRING ALGORITHMS DEMONSTRATION ===" << endl;
    
    // Palindrome check
    cout << "\n--- Palindrome Check ---" << endl;
    vector<string> palindrome_tests = {"racecar", "hello", "A man a plan a canal Panama", "race"};
    
    for (const string& test : palindrome_tests) {
        bool is_pal = StringAlgorithms::isPalindrome(test);
        cout << "\"" << test << "\" is " << (is_pal ? "" : "not ") << "a palindrome" << endl;
    }
    
    // Anagram check
    cout << "\n--- Anagram Check ---" << endl;
    vector<pair<string, string>> anagram_tests = {
        {"listen", "silent"},
        {"hello", "world"},
        {"evil", "vile"},
        {"astronomer", "moon starer"}
    };
    
    for (const auto& test : anagram_tests) {
        bool are_anag = StringAlgorithms::areAnagrams(test.first, test.second);
        cout << "\"" << test.first << "\" and \"" << test.second << "\" are " 
             << (are_anag ? "" : "not ") << "anagrams" << endl;
    }
    
    // Longest palindrome
    cout << "\n--- Longest Palindromic Substring ---" << endl;
    string pal_text = "babad";
    string longest_pal = StringAlgorithms::longestPalindrome(pal_text);
    cout << "Longest palindrome in \"" << pal_text << "\": \"" << longest_pal << "\"" << endl;
    
    // Character frequency
    cout << "\n--- Character Frequency ---" << endl;
    string freq_text = "programming";
    auto freq = StringAlgorithms::characterFrequency(freq_text);
    cout << "Character frequencies in \"" << freq_text << "\":" << endl;
    for (const auto& pair : freq) {
        cout << "'" << pair.first << "': " << pair.second << endl;
    }
    
    // First non-repeating character
    cout << "\n--- First Non-Repeating Character ---" << endl;
    string non_rep_text = "abccba";
    char first_non_rep = StringAlgorithms::firstNonRepeatingChar(non_rep_text);
    cout << "First non-repeating in \"" << non_rep_text << "\": '" 
         << (first_non_rep ? first_non_rep : '0') << "'" << endl;
    
    // KMP pattern matching
    cout << "\n--- KMP Pattern Matching ---" << endl;
    string text = "ABABDABACDABABCABCABCABCABC";
    string pattern = "ABABCABCABCABC";
    auto matches = StringAlgorithms::KMPSearch(text, pattern);
    cout << "Pattern \"" << pattern << "\" found at positions: ";
    for (int pos : matches) cout << pos << " ";
    cout << endl;
    
    // Longest common subsequence
    cout << "\n--- Longest Common Subsequence ---" << endl;
    string lcs1 = "ABCDGH";
    string lcs2 = "AEDFHR";
    string lcs = StringAlgorithms::longestCommonSubsequence(lcs1, lcs2);
    cout << "LCS of \"" << lcs1 << "\" and \"" << lcs2 << "\": \"" << lcs << "\"" << endl;
    
    // Edit distance
    cout << "\n--- Edit Distance ---" << endl;
    string edit1 = "kitten";
    string edit2 = "sitting";
    int edit_dist = StringAlgorithms::editDistance(edit1, edit2);
    cout << "Edit distance between \"" << edit1 << "\" and \"" << edit2 << "\": " << edit_dist << endl;
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main() {
    cout << "=== COMPLETE STRING GUIDE ===" << endl;
    
    demonstrate_string_initialization();
    demonstrate_string_access();
    demonstrate_string_search();
    demonstrate_string_comparison();
    demonstrate_string_manipulation();
    demonstrate_string_splitting();
    demonstrate_string_algorithms();
    demonstrate_regex();
    demonstrate_string_performance();
    
    cout << "\n=== SUMMARY ===" << endl;
    cout << "1. Initialization: Multiple ways to create and initialize strings" << endl;
    cout << "2. Access: Safe and unsafe methods to access/modify characters" << endl;
    cout << "3. Search: Find, rfind, and various search operations" << endl;
    cout << "4. Manipulation: Substring, case conversion, trimming, replacement" << endl;
    cout << "5. Parsing: Splitting and joining strings efficiently" << endl;
    cout << "6. Algorithms: Palindromes, anagrams, pattern matching, LCS" << endl;
    cout << "7. Regex: Pattern matching and replacement using regular expressions" << endl;
    cout << "8. Performance: Memory management and efficient string operations" << endl;
    
    return 0;
} 