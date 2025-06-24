/*
 * =============================================================================
 * CONTROL STRUCTURES - Comprehensive C++ Tutorial
 * =============================================================================
 * 
 * This file demonstrates:
 * 1. Conditional statements (if, else if, else, switch)
 * 2. Loops (for, while, do-while, range-based for)
 * 3. Jump statements (break, continue, goto, return)
 * 4. Modern C++ control flow features
 * 5. Performance considerations and optimizations
 * 6. Common pitfalls and debugging techniques
 * 
 * Compile with: g++ -std=c++17 -Wall -Wextra -g -O0 control_structures.cpp -o control_structures
 * Run with debugging: gdb ./control_structures
 * =============================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <random>
#include <algorithm>
#include <cassert>
#include <iomanip>

// Debug macro for tracing execution flow
#define TRACE(msg) \
    std::cout << "[TRACE] " << __FUNCTION__ << ":" << __LINE__ << " - " << msg << std::endl;

// Performance measurement macro
#define MEASURE_TIME(code_block, description) \
    { \
        auto start = std::chrono::high_resolution_clock::now(); \
        code_block; \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); \
        std::cout << "[PERF] " << description << ": " << duration.count() << " microseconds" << std::endl; \
    }

// Function to demonstrate if-else statements
void demonstrate_conditional_statements() {
    std::cout << "\n🔀 CONDITIONAL STATEMENTS DEMONSTRATION\n";
    std::cout << std::string(55, '=') << std::endl;
    
    // === BASIC IF-ELSE ===
    TRACE("Starting basic if-else demonstration");
    
    int score = 85;
    char grade;
    
    // Traditional if-else chain
    if (score >= 90) {
        grade = 'A';
        std::cout << "Excellent! Grade: " << grade << std::endl;
    } else if (score >= 80) {
        grade = 'B';
        std::cout << "Good! Grade: " << grade << std::endl;
    } else if (score >= 70) {
        grade = 'C';
        std::cout << "Average. Grade: " << grade << std::endl;
    } else if (score >= 60) {
        grade = 'D';
        std::cout << "Below average. Grade: " << grade << std::endl;
    } else {
        grade = 'F';
        std::cout << "Fail. Grade: " << grade << std::endl;
    }
    
    // === CONDITIONAL OPERATOR (TERNARY) ===
    std::cout << "\n🎯 TERNARY OPERATOR\n";
    std::cout << std::string(25, '-') << std::endl;
    
    int a = 10, b = 20;
    int max_value = (a > b) ? a : b;
    std::cout << "Max of " << a << " and " << b << " is: " << max_value << std::endl;
    
    // Nested ternary (avoid in production code!)
    int x = 5, y = 10, z = 15;
    int max_of_three = (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
    std::cout << "Max of " << x << ", " << y << ", " << z << " is: " << max_of_three << std::endl;
    
    // === INITIALIZATION WITH IF STATEMENTS (C++17) ===
    std::cout << "\n🆕 IF WITH INITIALIZATION (C++17)\n";
    std::cout << std::string(35, '-') << std::endl;
    
    // Traditional way
    std::map<std::string, int> scores = {{"Alice", 95}, {"Bob", 87}, {"Charlie", 92}};
    auto it = scores.find("Alice");
    if (it != scores.end()) {
        std::cout << "Alice's score (traditional): " << it->second << std::endl;
    }
    
    // C++17 way - initialization in if statement
    if (auto it = scores.find("Bob"); it != scores.end()) {
        std::cout << "Bob's score (C++17 style): " << it->second << std::endl;
    }
    
    // === CONSTEXPR IF (C++17) ===
    std::cout << "\n⚡ CONSTEXPR IF (C++17)\n";
    std::cout << std::string(25, '-') << std::endl;
    
    auto process_value = [](auto value) {
        if constexpr (std::is_integral_v<decltype(value)>) {
            std::cout << "Processing integer: " << value << std::endl;
            return value * 2;
        } else if constexpr (std::is_floating_point_v<decltype(value)>) {
            std::cout << "Processing float: " << value << std::endl;
            return value * 1.5;
        } else {
            std::cout << "Processing other type" << std::endl;
            return value;
        }
    };
    
    auto result1 = process_value(42);      // integer path
    auto result2 = process_value(3.14);    // floating-point path
    auto result3 = process_value("text");  // other path
    
    std::cout << "Results: " << result1 << ", " << result2 << ", " << result3 << std::endl;
}

// Function to demonstrate switch statements
void demonstrate_switch_statements() {
    std::cout << "\n🔄 SWITCH STATEMENTS DEMONSTRATION\n";
    std::cout << std::string(45, '=') << std::endl;
    
    // === BASIC SWITCH ===
    TRACE("Starting switch statement demonstration");
    
    char operation = '+';
    double operand1 = 10.5, operand2 = 3.2;
    double result = 0;
    
    switch (operation) {
        case '+':
            result = operand1 + operand2;
            std::cout << operand1 << " + " << operand2 << " = " << result << std::endl;
            break;
        case '-':
            result = operand1 - operand2;
            std::cout << operand1 << " - " << operand2 << " = " << result << std::endl;
            break;
        case '*':
            result = operand1 * operand2;
            std::cout << operand1 << " * " << operand2 << " = " << result << std::endl;
            break;
        case '/':
            if (operand2 != 0) {
                result = operand1 / operand2;
                std::cout << operand1 << " / " << operand2 << " = " << result << std::endl;
            } else {
                std::cout << "Error: Division by zero!" << std::endl;
            }
            break;
        default:
            std::cout << "Unknown operation: " << operation << std::endl;
            break;
    }
    
    // === SWITCH WITH FALLTHROUGH ===
    std::cout << "\n⬇️ SWITCH FALLTHROUGH\n";
    std::cout << std::string(25, '-') << std::endl;
    
    int day = 3;
    std::cout << "Day " << day << ": ";
    
    switch (day) {
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
            std::cout << "Weekday";
            break;
        case 6:
        case 7:
            std::cout << "Weekend";
            break;
        default:
            std::cout << "Invalid day";
            break;
    }
    std::cout << std::endl;
    
    // === SWITCH WITH INITIALIZATION (C++17) ===
    std::cout << "\n🆕 SWITCH WITH INITIALIZATION (C++17)\n";
    std::cout << std::string(40, '-') << std::endl;
    
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    switch (auto size = numbers.size(); size) {
        case 0:
            std::cout << "Empty vector" << std::endl;
            break;
        case 1:
            std::cout << "Single element: " << numbers[0] << std::endl;
            break;
        default:
            std::cout << "Vector with " << size << " elements" << std::endl;
            std::cout << "First: " << numbers.front() << ", Last: " << numbers.back() << std::endl;
            break;
    }
    
    // === ENUM CLASS WITH SWITCH ===
    std::cout << "\n🏷️ ENUM CLASS WITH SWITCH\n";
    std::cout << std::string(30, '-') << std::endl;
    
    enum class Color { RED, GREEN, BLUE, YELLOW };
    
    auto describe_color = [](Color c) {
        switch (c) {
            case Color::RED:
                return "Warm and energetic";
            case Color::GREEN:
                return "Natural and calm";
            case Color::BLUE:
                return "Cool and peaceful";
            case Color::YELLOW:
                return "Bright and cheerful";
        }
        return "Unknown color"; // This should never be reached
    };
    
    Color current_color = Color::BLUE;
    std::cout << "Blue is: " << describe_color(current_color) << std::endl;
}

// Function to demonstrate loops
void demonstrate_loops() {
    std::cout << "\n🔄 LOOPS DEMONSTRATION\n";
    std::cout << std::string(30, '=') << std::endl;
    
    // === FOR LOOP ===
    std::cout << "\n🔢 TRADITIONAL FOR LOOP\n";
    std::cout << std::string(30, '-') << std::endl;
    
    TRACE("Starting for loop demonstration");
    
    // Classic for loop
    std::cout << "Counting from 1 to 5: ";
    for (int i = 1; i <= 5; ++i) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    
    // Multiple variables in for loop
    std::cout << "Multiple variables: ";
    for (int i = 0, j = 10; i < 5; ++i, --j) {
        std::cout << "(" << i << "," << j << ") ";
    }
    std::cout << std::endl;
    
    // Nested for loops
    std::cout << "Multiplication table (3x3):" << std::endl;
    for (int i = 1; i <= 3; ++i) {
        for (int j = 1; j <= 3; ++j) {
            std::cout << std::setw(4) << (i * j);
        }
        std::cout << std::endl;
    }
    
    // === RANGE-BASED FOR LOOP (C++11) ===
    std::cout << "\n🆕 RANGE-BASED FOR LOOP (C++11)\n";
    std::cout << std::string(35, '-') << std::endl;
    
    std::vector<std::string> fruits = {"apple", "banana", "cherry", "date"};
    
    // By value (creates copies)
    std::cout << "Fruits (by value): ";
    for (auto fruit : fruits) {
        std::cout << fruit << " ";
    }
    std::cout << std::endl;
    
    // By const reference (more efficient)
    std::cout << "Fruits (by const ref): ";
    for (const auto& fruit : fruits) {
        std::cout << fruit << " ";
    }
    std::cout << std::endl;
    
    // By reference (can modify)
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::cout << "Before modification: ";
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    // Modify elements
    for (auto& num : numbers) {
        num *= 2;
    }
    
    std::cout << "After modification: ";
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    // === WHILE LOOP ===
    std::cout << "\n🔄 WHILE LOOP\n";
    std::cout << std::string(15, '-') << std::endl;
    
    int countdown = 5;
    std::cout << "Countdown: ";
    while (countdown > 0) {
        std::cout << countdown << " ";
        --countdown;
    }
    std::cout << "Blast off!" << std::endl;
    
    // === DO-WHILE LOOP ===
    std::cout << "\n🔄 DO-WHILE LOOP\n";
    std::cout << std::string(20, '-') << std::endl;
    
    int input;
    do {
        std::cout << "Enter a number between 1 and 10 (0 to quit): ";
        // Simulate user input
        static int simulated_inputs[] = {15, -5, 7, 0};
        static int input_index = 0;
        input = simulated_inputs[input_index++];
        std::cout << input << std::endl;
        
        if (input < 0 || input > 10) {
            std::cout << "Invalid input! Try again." << std::endl;
        } else if (input != 0) {
            std::cout << "You entered: " << input << std::endl;
        }
    } while (input != 0 && input_index < 4);
    
    std::cout << "Goodbye!" << std::endl;
}

// Function to demonstrate jump statements
void demonstrate_jump_statements() {
    std::cout << "\n⏭️ JUMP STATEMENTS DEMONSTRATION\n";
    std::cout << std::string(40, '=') << std::endl;
    
    // === BREAK STATEMENT ===
    std::cout << "\n🛑 BREAK STATEMENT\n";
    std::cout << std::string(20, '-') << std::endl;
    
    TRACE("Demonstrating break statement");
    
    // Break in loop
    std::cout << "Finding first even number > 10: ";
    for (int i = 11; i <= 20; ++i) {
        if (i % 2 == 0) {
            std::cout << i << " (found!)" << std::endl;
            break; // Exit the loop
        }
        std::cout << i << "(odd) ";
    }
    
    // Break in nested loops
    std::cout << "Break in nested loops:" << std::endl;
    bool found = false;
    for (int i = 1; i <= 3 && !found; ++i) {
        for (int j = 1; j <= 3; ++j) {
            std::cout << "(" << i << "," << j << ") ";
            if (i == 2 && j == 2) {
                std::cout << "-> Breaking inner loop ";
                found = true;
                break; // Only breaks inner loop
            }
        }
        std::cout << std::endl;
    }
    
    // === CONTINUE STATEMENT ===
    std::cout << "\n⏭️ CONTINUE STATEMENT\n";
    std::cout << std::string(25, '-') << std::endl;
    
    std::cout << "Even numbers from 1 to 10: ";
    for (int i = 1; i <= 10; ++i) {
        if (i % 2 != 0) {
            continue; // Skip odd numbers
        }
        std::cout << i << " ";
    }
    std::cout << std::endl;
    
    // === GOTO STATEMENT (USE WITH CAUTION!) ===
    std::cout << "\n⚠️ GOTO STATEMENT (DEMONSTRATION ONLY)\n";
    std::cout << std::string(45, '-') << std::endl;
    
    int error_code = 0;
    
    // Simulate some operations
    std::cout << "Operation 1: ";
    if (std::rand() % 2 == 0) {
        error_code = 1;
        goto error_handler;
    }
    std::cout << "Success" << std::endl;
    
    std::cout << "Operation 2: ";
    if (std::rand() % 3 == 0) {
        error_code = 2;
        goto error_handler;
    }
    std::cout << "Success" << std::endl;
    
    std::cout << "All operations completed successfully!" << std::endl;
    goto cleanup;
    
error_handler:
    std::cout << "Failed with error code: " << error_code << std::endl;
    
cleanup:
    std::cout << "Cleanup completed." << std::endl;
    
    std::cout << "\n⚠️ NOTE: goto is generally discouraged in modern C++!" << std::endl;
    std::cout << "Better alternatives: exceptions, RAII, structured programming" << std::endl;
}

// Function to demonstrate performance considerations
void demonstrate_performance_considerations() {
    std::cout << "\n⚡ PERFORMANCE CONSIDERATIONS\n";
    std::cout << std::string(40, '=') << std::endl;
    
    const size_t SIZE = 1000000;
    std::vector<int> data(SIZE);
    
    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    
    for (size_t i = 0; i < SIZE; ++i) {
        data[i] = dis(gen);
    }
    
    // === TRADITIONAL FOR LOOP ===
    int sum1 = 0;
    MEASURE_TIME({
        for (size_t i = 0; i < data.size(); ++i) {
            sum1 += data[i];
        }
    }, "Traditional for loop");
    
    // === ITERATOR-BASED LOOP ===
    int sum2 = 0;
    MEASURE_TIME({
        for (auto it = data.begin(); it != data.end(); ++it) {
            sum2 += *it;
        }
    }, "Iterator-based loop");
    
    // === RANGE-BASED FOR LOOP ===
    int sum3 = 0;
    MEASURE_TIME({
        for (const auto& value : data) {
            sum3 += value;
        }
    }, "Range-based for loop");
    
    // === STL ALGORITHM ===
    int sum4 = 0;
    MEASURE_TIME({
        sum4 = std::accumulate(data.begin(), data.end(), 0);
    }, "STL accumulate");
    
    std::cout << "All sums equal? " << std::boolalpha 
              << (sum1 == sum2 && sum2 == sum3 && sum3 == sum4) << std::endl;
    std::cout << "Sum: " << sum1 << std::endl;
    
    // === LOOP OPTIMIZATION TIPS ===
    std::cout << "\n🎯 OPTIMIZATION TIPS\n";
    std::cout << std::string(25, '-') << std::endl;
    
    // Inefficient: size() called every iteration
    std::vector<int> test_data = {1, 2, 3, 4, 5};
    std::cout << "Inefficient loop: ";
    MEASURE_TIME({
        for (size_t i = 0; i < test_data.size(); ++i) {
            // size() is called every iteration!
            std::cout << test_data[i] << " ";
        }
    }, "size() in loop condition");
    std::cout << std::endl;
    
    // Efficient: size() called once
    std::cout << "Efficient loop: ";
    MEASURE_TIME({
        const size_t size = test_data.size();
        for (size_t i = 0; i < size; ++i) {
            std::cout << test_data[i] << " ";
        }
    }, "size() cached");
    std::cout << std::endl;
}

// Function to demonstrate common pitfalls
void demonstrate_common_pitfalls() {
    std::cout << "\n⚠️ COMMON PITFALLS AND DEBUGGING\n";
    std::cout << std::string(45, '=') << std::endl;
    
    // === INFINITE LOOP PREVENTION ===
    std::cout << "\n🔄 INFINITE LOOP PREVENTION\n";
    std::cout << std::string(35, '-') << std::endl;
    
    std::cout << "Safe loop with counter: ";
    int counter = 0;
    const int MAX_ITERATIONS = 10;
    
    while (true) {
        std::cout << counter << " ";
        ++counter;
        
        // Safety mechanism
        if (counter >= MAX_ITERATIONS) {
            std::cout << "(safety break)";
            break;
        }
        
        // Some condition that might never become true
        if (counter == 1000) { // This will never be reached
            break;
        }
    }
    std::cout << std::endl;
    
    // === OFF-BY-ONE ERRORS ===
    std::cout << "\n🎯 OFF-BY-ONE ERROR EXAMPLES\n";
    std::cout << std::string(35, '-') << std::endl;
    
    std::vector<int> array = {10, 20, 30, 40, 50};
    
    std::cout << "Correct loop (0 to size-1): ";
    for (size_t i = 0; i < array.size(); ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
    
    // Demonstrate bounds checking
    std::cout << "Bounds checking example: ";
    for (int i = -1; i <= static_cast<int>(array.size()); ++i) {
        if (i >= 0 && i < static_cast<int>(array.size())) {
            std::cout << array[i] << " ";
        } else {
            std::cout << "[OUT_OF_BOUNDS] ";
        }
    }
    std::cout << std::endl;
    
    // === SWITCH STATEMENT PITFALLS ===
    std::cout << "\n🔄 SWITCH STATEMENT PITFALLS\n";
    std::cout << std::string(35, '-') << std::endl;
    
    int value = 2;
    std::cout << "Missing breaks (intentional fallthrough): ";
    
    switch (value) {
        case 1:
            std::cout << "One ";
            [[fallthrough]]; // C++17 attribute to indicate intentional fallthrough
        case 2:
            std::cout << "Two ";
            [[fallthrough]];
        case 3:
            std::cout << "Three ";
            break;
        default:
            std::cout << "Other ";
            break;
    }
    std::cout << std::endl;
    
    // === VARIABLE SCOPE ISSUES ===
    std::cout << "\n🔍 VARIABLE SCOPE ISSUES\n";
    std::cout << std::string(30, '-') << std::endl;
    
    std::cout << "Variable scope demonstration:" << std::endl;
    
    int outer_var = 10;
    std::cout << "Outer variable: " << outer_var << std::endl;
    
    for (int i = 0; i < 3; ++i) {
        int loop_var = i * 10;
        std::cout << "Loop " << i << ": loop_var = " << loop_var 
                  << ", outer_var = " << outer_var << std::endl;
        
        if (i == 1) {
            int inner_var = 999;
            std::cout << "  Inner scope: inner_var = " << inner_var << std::endl;
        }
        // inner_var is not accessible here
    }
    // loop_var is not accessible here
    
    std::cout << "After loop: outer_var = " << outer_var << std::endl;
}

// Main function demonstrating all concepts
int main() {
    std::cout << "🚀 C++ CONTROL STRUCTURES - COMPREHENSIVE TUTORIAL\n";
    std::cout << std::string(65, '=') << std::endl;
    
    try {
        // Seed random number generator
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        
        // Run all demonstrations
        demonstrate_conditional_statements();
        demonstrate_switch_statements();
        demonstrate_loops();
        demonstrate_jump_statements();
        demonstrate_performance_considerations();
        demonstrate_common_pitfalls();
        
        std::cout << "\n✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!\n";
        std::cout << std::string(50, '=') << std::endl;
        
        // === DEBUGGING TIPS ===
        std::cout << "\n🐛 DEBUGGING TIPS:\n";
        std::cout << "1. Use TRACE macro to track execution flow\n";
        std::cout << "2. Set breakpoints on loop conditions and inside loops\n";
        std::cout << "3. Watch variables change with 'watch variable_name' in gdb\n";
        std::cout << "4. Use 'until' command to skip to end of loop\n";
        std::cout << "5. Check loop indices and bounds carefully\n";
        std::cout << "6. Verify switch case fallthrough behavior\n";
        
        std::cout << "\n📚 UNDERSTANDING POINTS:\n";
        std::cout << "1. if-else vs switch: use switch for multiple discrete values\n";
        std::cout << "2. Range-based for loops are cleaner and safer\n";
        std::cout << "3. Always prefer ++i over i++ in loops for iterators\n";
        std::cout << "4. Be careful with unsigned/signed comparisons in loops\n";
        std::cout << "5. Use [[fallthrough]] attribute for intentional switch fallthrough\n";
        std::cout << "6. constexpr if enables compile-time branching\n";
        std::cout << "7. C++17 init-if and init-switch reduce variable scope\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
