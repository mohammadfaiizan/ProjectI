/*
 * =============================================================================
 * TYPES AND LITERALS - Comprehensive C++ Tutorial
 * =============================================================================
 * 
 * This file demonstrates:
 * 1. Fundamental types and their properties
 * 2. Literal types and representations
 * 3. Type deduction and inference
 * 4. Memory layout and size analysis
 * 5. Type traits and introspection
 * 6. Common pitfalls and debugging techniques
 * 
 * Compile with: g++ -std=c++17 -Wall -Wextra -g -O0 types_and_literals.cpp -o types_and_literals
 * Run with debugging: gdb ./types_and_literals
 * =============================================================================
 */

#include <iostream>
#include <iomanip>
#include <limits>
#include <type_traits>
#include <typeinfo>
#include <string>
#include <bitset>
#include <cstdint>
#include <climits>
#include <cfloat>

// Debug macro for detailed output
#define DEBUG_TYPE(type, var) \
    std::cout << "=== " << #type << " Analysis ===" << std::endl; \
    std::cout << "Variable: " << #var << " = " << var << std::endl; \
    std::cout << "Size: " << sizeof(type) << " bytes (" << sizeof(type) * 8 << " bits)" << std::endl; \
    std::cout << "Min: " << std::numeric_limits<type>::min() << std::endl; \
    std::cout << "Max: " << std::numeric_limits<type>::max() << std::endl; \
    std::cout << "Is signed: " << std::boolalpha << std::numeric_limits<type>::is_signed << std::endl; \
    std::cout << "Memory address: " << &var << std::endl; \
    std::cout << "Type name: " << typeid(type).name() << std::endl; \
    std::cout << std::string(50, '-') << std::endl;

// Template function to show binary representation
template<typename T>
void show_binary_representation(const T& value, const std::string& description) {
    std::cout << description << ": ";
    std::cout << std::bitset<sizeof(T) * 8>(reinterpret_cast<const unsigned long long&>(value)) << std::endl;
}

// Function to demonstrate integer types and literals
void demonstrate_integer_types() {
    std::cout << "\n🔢 INTEGER TYPES AND LITERALS DEMONSTRATION\n";
    std::cout << std::string(60, '=') << std::endl;
    
    // === FUNDAMENTAL INTEGER TYPES ===
    
    // char type (implementation-defined signedness)
    char c1 = 'A';                    // Character literal
    char c2 = 65;                     // ASCII value
    char c3 = '\x41';                 // Hexadecimal escape
    char c4 = '\101';                 // Octal escape
    DEBUG_TYPE(char, c1);
    std::cout << "Character representations: '" << c1 << "' " << static_cast<int>(c2) << " " << c3 << " " << c4 << std::endl;
    
    // Explicitly signed/unsigned char
    signed char sc = -128;
    unsigned char uc = 255;
    DEBUG_TYPE(signed char, sc);
    DEBUG_TYPE(unsigned char, uc);
    
    // Short integers
    short s1 = 32767;                 // Maximum short value
    short s2 = -32768;                // Minimum short value
    unsigned short us = 65535;        // Maximum unsigned short
    DEBUG_TYPE(short, s1);
    DEBUG_TYPE(unsigned short, us);
    
    // Regular integers with different literal representations
    int decimal = 42;                 // Decimal literal
    int octal = 052;                  // Octal literal (starts with 0)
    int hex = 0x2A;                   // Hexadecimal literal
    int binary = 0b101010;            // Binary literal (C++14)
    DEBUG_TYPE(int, decimal);
    std::cout << "Same value in different bases: " << decimal << " = " << octal << " = " << hex << " = " << binary << std::endl;
    
    // Long integers
    long l1 = 2147483647L;            // Long literal suffix
    long long ll1 = 9223372036854775807LL; // Long long literal suffix
    unsigned long ul = 4294967295UL;  // Unsigned long literal
    unsigned long long ull = 18446744073709551615ULL; // Unsigned long long literal
    DEBUG_TYPE(long, l1);
    DEBUG_TYPE(long long, ll1);
    DEBUG_TYPE(unsigned long long, ull);
    
    // === FIXED-WIDTH INTEGER TYPES (C++11) ===
    std::cout << "\n📏 FIXED-WIDTH INTEGER TYPES\n";
    std::cout << std::string(40, '-') << std::endl;
    
    std::int8_t i8 = 127;
    std::int16_t i16 = 32767;
    std::int32_t i32 = 2147483647;
    std::int64_t i64 = 9223372036854775807LL;
    
    std::uint8_t ui8 = 255;
    std::uint16_t ui16 = 65535;
    std::uint32_t ui32 = 4294967295U;
    std::uint64_t ui64 = 18446744073709551615ULL;
    
    DEBUG_TYPE(std::int32_t, i32);
    DEBUG_TYPE(std::uint64_t, ui64);
    
    // === LITERAL SUFFIXES DEMONSTRATION ===
    std::cout << "\n🏷️ LITERAL SUFFIXES\n";
    std::cout << std::string(30, '-') << std::endl;
    
    auto no_suffix = 100;              // int
    auto u_suffix = 100U;              // unsigned int
    auto l_suffix = 100L;              // long
    auto ul_suffix = 100UL;            // unsigned long
    auto ll_suffix = 100LL;            // long long
    auto ull_suffix = 100ULL;          // unsigned long long
    
    std::cout << "Type deduction with literals:" << std::endl;
    std::cout << "no_suffix type: " << typeid(no_suffix).name() << std::endl;
    std::cout << "u_suffix type: " << typeid(u_suffix).name() << std::endl;
    std::cout << "l_suffix type: " << typeid(l_suffix).name() << std::endl;
    std::cout << "ll_suffix type: " << typeid(ll_suffix).name() << std::endl;
    
    // === DIGIT SEPARATORS (C++14) ===
    std::cout << "\n📊 DIGIT SEPARATORS (C++14)\n";
    std::cout << std::string(35, '-') << std::endl;
    
    int million = 1'000'000;           // Much more readable!
    long long big_num = 1'234'567'890'123LL;
    int hex_with_sep = 0xFF'EE'DD'CC;
    int binary_with_sep = 0b1010'1100'1111'0000;
    
    std::cout << "Million: " << million << std::endl;
    std::cout << "Big number: " << big_num << std::endl;
    std::cout << "Hex with separators: " << std::hex << hex_with_sep << std::dec << std::endl;
    std::cout << "Binary with separators: " << binary_with_sep << std::endl;
    
    // === OVERFLOW AND UNDERFLOW DEMONSTRATION ===
    std::cout << "\n⚠️ OVERFLOW/UNDERFLOW DEMONSTRATION\n";
    std::cout << std::string(45, '-') << std::endl;
    
    unsigned char uc_max = 255;
    std::cout << "Before overflow: " << static_cast<int>(uc_max) << std::endl;
    uc_max++;  // This will overflow (wrap around to 0)
    std::cout << "After overflow: " << static_cast<int>(uc_max) << std::endl;
    
    signed char sc_min = -128;
    std::cout << "Before underflow: " << static_cast<int>(sc_min) << std::endl;
    sc_min--;  // This will underflow (wrap around to 127)
    std::cout << "After underflow: " << static_cast<int>(sc_min) << std::endl;
}

// Function to demonstrate floating-point types and literals
void demonstrate_floating_point_types() {
    std::cout << "\n🔢 FLOATING-POINT TYPES AND LITERALS\n";
    std::cout << std::string(50, '=') << std::endl;
    
    // === BASIC FLOATING-POINT TYPES ===
    
    float f1 = 3.14159f;              // float literal (suffix f)
    float f2 = 3.14159F;              // float literal (suffix F)
    float f3 = 1.23e-4f;              // Scientific notation
    float f4 = 0x1.2p3f;              // Hexadecimal floating literal (C++17)
    
    double d1 = 3.141592653589793;    // double literal (default)
    double d2 = 2.718281828459045;    // Another double
    double d3 = 1.23e-10;             // Scientific notation
    double d4 = 0x1.921fb54442d18p+1; // Hexadecimal floating literal
    
    long double ld1 = 3.141592653589793238L; // long double literal
    long double ld2 = 1.23e-15L;      // Scientific notation
    
    DEBUG_TYPE(float, f1);
    DEBUG_TYPE(double, d1);
    DEBUG_TYPE(long double, ld1);
    
    // === PRECISION DEMONSTRATION ===
    std::cout << "\n🎯 PRECISION COMPARISON\n";
    std::cout << std::string(30, '-') << std::endl;
    
    std::cout << std::fixed << std::setprecision(20);
    std::cout << "float precision:       " << f1 << std::endl;
    std::cout << "double precision:      " << d1 << std::endl;
    std::cout << "long double precision: " << ld1 << std::endl;
    
    // === SPECIAL FLOATING-POINT VALUES ===
    std::cout << "\n🔮 SPECIAL FLOATING-POINT VALUES\n";
    std::cout << std::string(40, '-') << std::endl;
    
    double positive_infinity = std::numeric_limits<double>::infinity();
    double negative_infinity = -std::numeric_limits<double>::infinity();
    double not_a_number = std::numeric_limits<double>::quiet_NaN();
    double signaling_nan = std::numeric_limits<double>::signaling_NaN();
    
    std::cout << "Positive infinity: " << positive_infinity << std::endl;
    std::cout << "Negative infinity: " << negative_infinity << std::endl;
    std::cout << "Quiet NaN: " << not_a_number << std::endl;
    std::cout << "Signaling NaN: " << signaling_nan << std::endl;
    
    // Testing for special values
    std::cout << "\nTesting special values:" << std::endl;
    std::cout << "Is positive_infinity infinite? " << std::boolalpha << std::isinf(positive_infinity) << std::endl;
    std::cout << "Is not_a_number NaN? " << std::isnan(not_a_number) << std::endl;
    std::cout << "Is 3.14 finite? " << std::isfinite(3.14) << std::endl;
    
    // === FLOATING-POINT ARITHMETIC PITFALLS ===
    std::cout << "\n⚠️ FLOATING-POINT ARITHMETIC PITFALLS\n";
    std::cout << std::string(45, '-') << std::endl;
    
    double a = 0.1;
    double b = 0.2;
    double c = a + b;
    
    std::cout << std::setprecision(20);
    std::cout << "0.1 + 0.2 = " << c << std::endl;
    std::cout << "Is 0.1 + 0.2 == 0.3? " << std::boolalpha << (c == 0.3) << std::endl;
    
    // Proper floating-point comparison
    const double epsilon = 1e-9;
    bool approximately_equal = std::abs(c - 0.3) < epsilon;
    std::cout << "Approximately equal to 0.3? " << approximately_equal << std::endl;
    
    // === FLOATING-POINT LITERAL FORMATS ===
    std::cout << "\n📝 FLOATING-POINT LITERAL FORMATS\n";
    std::cout << std::string(40, '-') << std::endl;
    
    double literals[] = {
        123.456,        // Standard decimal
        1.23456e2,      // Scientific notation (positive exponent)
        1.23456e-2,     // Scientific notation (negative exponent)
        1.23456E2,      // Scientific notation (capital E)
        123.,           // Trailing decimal point
        .456,           // Leading decimal point
        123.456e0       // Exponent of zero
    };
    
    for (size_t i = 0; i < sizeof(literals)/sizeof(literals[0]); ++i) {
        std::cout << "Literal " << i << ": " << std::setprecision(6) << literals[i] << std::endl;
    }
}

// Function to demonstrate boolean and character types
void demonstrate_boolean_and_character_types() {
    std::cout << "\n🔤 BOOLEAN AND CHARACTER TYPES\n";
    std::cout << std::string(40, '=') << std::endl;
    
    // === BOOLEAN TYPE ===
    std::cout << "\n✅ BOOLEAN TYPE\n";
    std::cout << std::string(20, '-') << std::endl;
    
    bool b1 = true;
    bool b2 = false;
    bool b3 = 42;                     // Non-zero converts to true
    bool b4 = 0;                      // Zero converts to false
    bool b5 = nullptr;                // nullptr converts to false
    
    DEBUG_TYPE(bool, b1);
    std::cout << "Boolean values:" << std::endl;
    std::cout << "b1 (true): " << std::boolalpha << b1 << std::endl;
    std::cout << "b2 (false): " << b2 << std::endl;
    std::cout << "b3 (42): " << b3 << std::endl;
    std::cout << "b4 (0): " << b4 << std::endl;
    std::cout << "b5 (nullptr): " << b5 << std::endl;
    
    // Boolean arithmetic (not recommended!)
    std::cout << "\nBoolean arithmetic (avoid this!):" << std::endl;
    std::cout << "true + true = " << (true + true) << std::endl;
    std::cout << "true * false = " << (true * false) << std::endl;
    
    // === CHARACTER TYPES ===
    std::cout << "\n🔤 CHARACTER TYPES\n";
    std::cout << std::string(25, '-') << std::endl;
    
    char c1 = 'A';
    char c2 = '\n';                   // Newline escape sequence
    char c3 = '\t';                   // Tab escape sequence
    char c4 = '\\';                   // Backslash escape sequence
    char c5 = '\'';                   // Single quote escape sequence
    char c6 = '\0';                   // Null character
    
    std::cout << "Character literals:" << std::endl;
    std::cout << "c1 ('A'): '" << c1 << "' (ASCII: " << static_cast<int>(c1) << ")" << std::endl;
    std::cout << "c2 ('\\n'): ASCII " << static_cast<int>(c2) << std::endl;
    std::cout << "c3 ('\\t'): ASCII " << static_cast<int>(c3) << std::endl;
    std::cout << "c4 ('\\\\'): '" << c4 << "' (ASCII: " << static_cast<int>(c4) << ")" << std::endl;
    std::cout << "c5 ('\\''): '" << c5 << "' (ASCII: " << static_cast<int>(c5) << ")" << std::endl;
    std::cout << "c6 ('\\0'): ASCII " << static_cast<int>(c6) << std::endl;
    
    // === WIDE CHARACTERS (C++11 and later) ===
    std::cout << "\n🌐 WIDE CHARACTER TYPES\n";
    std::cout << std::string(30, '-') << std::endl;
    
    wchar_t wc = L'Ω';                // Wide character literal
    char16_t c16 = u'α';              // UTF-16 character literal
    char32_t c32 = U'🚀';             // UTF-32 character literal
    
    std::cout << "Wide character sizes:" << std::endl;
    std::cout << "wchar_t: " << sizeof(wchar_t) << " bytes" << std::endl;
    std::cout << "char16_t: " << sizeof(char16_t) << " bytes" << std::endl;
    std::cout << "char32_t: " << sizeof(char32_t) << " bytes" << std::endl;
    
    std::wcout << L"Wide character: " << wc << std::endl;
}

// Function to demonstrate string literals
void demonstrate_string_literals() {
    std::cout << "\n📝 STRING LITERALS\n";
    std::cout << std::string(30, '=') << std::endl;
    
    // === BASIC STRING LITERALS ===
    const char* str1 = "Hello, World!";           // C-style string
    const char* str2 = "Line 1\nLine 2\nLine 3";  // With escape sequences
    const char* str3 = "Quote: \"Hello\"";        // With escaped quotes
    const char* str4 = "Path: C:\\Users\\Name";   // With escaped backslashes
    
    std::cout << "Basic string literals:" << std::endl;
    std::cout << "str1: " << str1 << std::endl;
    std::cout << "str2:\n" << str2 << std::endl;
    std::cout << "str3: " << str3 << std::endl;
    std::cout << "str4: " << str4 << std::endl;
    
    // === RAW STRING LITERALS (C++11) ===
    std::cout << "\n📋 RAW STRING LITERALS (C++11)\n";
    std::cout << std::string(35, '-') << std::endl;
    
    const char* raw1 = R"(No escape sequences needed: \n \t \" \\)";
    const char* raw2 = R"delimiter(
    This is a multi-line
    raw string literal.
    No escaping needed for "quotes" or \backslashes\.
    )delimiter";
    
    const char* regex_pattern = R"(\d+\.\d+\.\d+\.\d+)"; // IP address regex
    const char* json_example = R"({"name": "John", "age": 30, "city": "New York"})";
    
    std::cout << "Raw string literals:" << std::endl;
    std::cout << "raw1: " << raw1 << std::endl;
    std::cout << "raw2: " << raw2 << std::endl;
    std::cout << "regex_pattern: " << regex_pattern << std::endl;
    std::cout << "json_example: " << json_example << std::endl;
    
    // === WIDE STRING LITERALS ===
    std::cout << "\n🌐 WIDE STRING LITERALS\n";
    std::cout << std::string(30, '-') << std::endl;
    
    const wchar_t* wide_str = L"Wide string with Unicode: αβγ";
    const char16_t* utf16_str = u"UTF-16 string: αβγ";
    const char32_t* utf32_str = U"UTF-32 string: αβγ🚀🌟";
    
    std::wcout << L"Wide string: " << wide_str << std::endl;
    std::cout << "UTF-16 string size: " << std::char_traits<char16_t>::length(utf16_str) << " characters" << std::endl;
    std::cout << "UTF-32 string size: " << std::char_traits<char32_t>::length(utf32_str) << " characters" << std::endl;
    
    // === STRING CONCATENATION ===
    std::cout << "\n🔗 STRING LITERAL CONCATENATION\n";
    std::cout << std::string(35, '-') << std::endl;
    
    const char* concatenated = "This is " "concatenated " "at compile time!";
    const char* multiline = "This is a very long string that "
                           "spans multiple lines for better "
                           "readability in source code.";
    
    std::cout << "Concatenated: " << concatenated << std::endl;
    std::cout << "Multiline: " << multiline << std::endl;
}

// Function to demonstrate type deduction and auto
void demonstrate_type_deduction() {
    std::cout << "\n🔍 TYPE DEDUCTION AND AUTO\n";
    std::cout << std::string(35, '=') << std::endl;
    
    // === AUTO KEYWORD ===
    std::cout << "\n🤖 AUTO KEYWORD\n";
    std::cout << std::string(20, '-') << std::endl;
    
    auto a1 = 42;                     // int
    auto a2 = 42U;                    // unsigned int  
    auto a3 = 42L;                    // long
    auto a4 = 42.0;                   // double
    auto a5 = 42.0f;                  // float
    auto a6 = 'c';                    // char
    auto a7 = "string";               // const char*
    auto a8 = true;                   // bool
    
    std::cout << "Type deduction with auto:" << std::endl;
    std::cout << "a1 (42): " << typeid(a1).name() << std::endl;
    std::cout << "a2 (42U): " << typeid(a2).name() << std::endl;
    std::cout << "a3 (42L): " << typeid(a3).name() << std::endl;
    std::cout << "a4 (42.0): " << typeid(a4).name() << std::endl;
    std::cout << "a5 (42.0f): " << typeid(a5).name() << std::endl;
    std::cout << "a6 ('c'): " << typeid(a6).name() << std::endl;
    std::cout << "a7 (\"string\"): " << typeid(a7).name() << std::endl;
    std::cout << "a8 (true): " << typeid(a8).name() << std::endl;
    
    // === AUTO WITH REFERENCES AND POINTERS ===
    std::cout << "\n🔗 AUTO WITH REFERENCES AND POINTERS\n";
    std::cout << std::string(40, '-') << std::endl;
    
    int x = 42;
    int& ref_x = x;
    const int& const_ref_x = x;
    int* ptr_x = &x;
    const int* const_ptr_x = &x;
    
    auto auto_x = x;                  // int (copy)
    auto auto_ref = ref_x;            // int (copy, not reference!)
    auto& auto_ref_proper = ref_x;    // int& (proper reference)
    auto auto_ptr = ptr_x;            // int*
    auto auto_const_ref = const_ref_x; // int (copy)
    const auto& auto_const_ref_proper = const_ref_x; // const int&
    
    std::cout << "Reference and pointer deduction:" << std::endl;
    std::cout << "auto_x type: " << typeid(auto_x).name() << std::endl;
    std::cout << "auto_ref type: " << typeid(auto_ref).name() << std::endl;
    std::cout << "auto_ref_proper type: " << typeid(auto_ref_proper).name() << std::endl;
    std::cout << "auto_ptr type: " << typeid(auto_ptr).name() << std::endl;
}

// Function to demonstrate nullptr and null pointer literal
void demonstrate_nullptr() {
    std::cout << "\n🚫 NULLPTR AND NULL POINTER LITERAL\n";
    std::cout << std::string(45, '=') << std::endl;
    
    // === NULLPTR VS NULL VS 0 ===
    int* ptr1 = nullptr;              // Preferred (C++11)
    int* ptr2 = NULL;                 // C-style (avoid in C++)
    int* ptr3 = 0;                    // Integer literal (avoid)
    
    std::cout << "Null pointer representations:" << std::endl;
    std::cout << "ptr1 (nullptr): " << ptr1 << std::endl;
    std::cout << "ptr2 (NULL): " << ptr2 << std::endl;
    std::cout << "ptr3 (0): " << ptr3 << std::endl;
    
    // Demonstrate why nullptr is better
    auto np1 = nullptr;               // std::nullptr_t
    auto np2 = NULL;                  // int (or long)
    auto np3 = 0;                     // int
    
    std::cout << "\nType deduction differences:" << std::endl;
    std::cout << "nullptr type: " << typeid(np1).name() << std::endl;
    std::cout << "NULL type: " << typeid(np2).name() << std::endl;
    std::cout << "0 type: " << typeid(np3).name() << std::endl;
    
    // === NULLPTR_T TYPE ===
    std::cout << "\n🔍 NULLPTR_T TYPE\n";
    std::cout << std::string(25, '-') << std::endl;
    
    std::nullptr_t null_ptr = nullptr;
    DEBUG_TYPE(std::nullptr_t, null_ptr);
    
    // Pointer safety checks
    if (ptr1 == nullptr) {
        std::cout << "ptr1 is null - safe to check!" << std::endl;
    }
    
    if (ptr2) {
        std::cout << "ptr2 is not null" << std::endl;
    } else {
        std::cout << "ptr2 is null" << std::endl;
    }
}

// Function to demonstrate memory layout and alignment
void demonstrate_memory_layout() {
    std::cout << "\n💾 MEMORY LAYOUT AND ALIGNMENT\n";
    std::cout << std::string(40, '=') << std::endl;
    
    // === STRUCTURE MEMORY LAYOUT ===
    struct SimpleStruct {
        char c;       // 1 byte
        int i;        // 4 bytes (typically)
        double d;     // 8 bytes (typically)
    };
    
    struct PackedStruct {
        char c1;      // 1 byte
        char c2;      // 1 byte  
        char c3;      // 1 byte
        char c4;      // 1 byte
        int i;        // 4 bytes
    };
    
    SimpleStruct simple;
    PackedStruct packed;
    
    std::cout << "Structure sizes and alignment:" << std::endl;
    std::cout << "SimpleStruct size: " << sizeof(SimpleStruct) << " bytes" << std::endl;
    std::cout << "PackedStruct size: " << sizeof(PackedStruct) << " bytes" << std::endl;
    
    std::cout << "\nSimpleStruct member offsets:" << std::endl;
    std::cout << "c offset: " << offsetof(SimpleStruct, c) << std::endl;
    std::cout << "i offset: " << offsetof(SimpleStruct, i) << std::endl;
    std::cout << "d offset: " << offsetof(SimpleStruct, d) << std::endl;
    
    // === ARRAY MEMORY LAYOUT ===
    int array[5] = {1, 2, 3, 4, 5};
    
    std::cout << "\nArray memory layout:" << std::endl;
    std::cout << "Array size: " << sizeof(array) << " bytes" << std::endl;
    std::cout << "Element size: " << sizeof(array[0]) << " bytes" << std::endl;
    std::cout << "Number of elements: " << sizeof(array)/sizeof(array[0]) << std::endl;
    
    std::cout << "Memory addresses:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "array[" << i << "]: " << &array[i] << " (value: " << array[i] << ")" << std::endl;
    }
    
    // === ALIGNMENT REQUIREMENTS ===
    std::cout << "\n🎯 ALIGNMENT REQUIREMENTS\n";
    std::cout << std::string(30, '-') << std::endl;
    
    std::cout << "Alignment requirements:" << std::endl;
    std::cout << "char: " << alignof(char) << " bytes" << std::endl;
    std::cout << "int: " << alignof(int) << " bytes" << std::endl;
    std::cout << "double: " << alignof(double) << " bytes" << std::endl;
    std::cout << "SimpleStruct: " << alignof(SimpleStruct) << " bytes" << std::endl;
}

// Main function demonstrating all concepts
int main() {
    std::cout << "🚀 C++ TYPES AND LITERALS - COMPREHENSIVE TUTORIAL\n";
    std::cout << std::string(65, '=') << std::endl;
    
    try {
        // Run all demonstrations
        demonstrate_integer_types();
        demonstrate_floating_point_types();
        demonstrate_boolean_and_character_types();
        demonstrate_string_literals();
        demonstrate_type_deduction();
        demonstrate_nullptr();
        demonstrate_memory_layout();
        
        std::cout << "\n✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!\n";
        std::cout << std::string(50, '=') << std::endl;
        
        // === DEBUGGING TIPS ===
        std::cout << "\n🐛 DEBUGGING TIPS:\n";
        std::cout << "1. Use 'gdb ./types_and_literals' to debug\n";
        std::cout << "2. Set breakpoints with 'break main' or 'break line_number'\n";
        std::cout << "3. Print variables with 'print variable_name'\n";
        std::cout << "4. Check memory with 'x/4xw &variable_name'\n";
        std::cout << "5. Step through code with 'step' or 'next'\n";
        std::cout << "6. Use 'info locals' to see all local variables\n";
        std::cout << "7. Use 'ptype variable_name' to see variable type\n";
        
        std::cout << "\n📚 UNDERSTANDING POINTS:\n";
        std::cout << "1. Integer literals have different suffixes (U, L, LL)\n";
        std::cout << "2. Floating-point precision varies between float/double/long double\n";
        std::cout << "3. Character literals can use escape sequences\n";
        std::cout << "4. Raw string literals avoid escape sequence issues\n";
        std::cout << "5. auto deduces type but may not preserve references\n";
        std::cout << "6. nullptr is preferred over NULL or 0 for pointers\n";
        std::cout << "7. Structure padding affects memory layout\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
