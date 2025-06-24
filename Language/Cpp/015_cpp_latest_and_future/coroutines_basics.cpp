/*
 * C++ LATEST AND FUTURE - COROUTINES BASICS
 * 
 * Comprehensive guide to C++20 coroutines including generators,
 * async operations, and practical applications.
 * 
 * Compilation: g++ -std=c++20 -fcoroutines -Wall -Wextra coroutines_basics.cpp -o coroutines_demo
 */

#include <iostream>
#include <coroutine>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <future>
#include <optional>

// =============================================================================
// BASIC GENERATOR COROUTINE
// =============================================================================

template<typename T>
class Generator {
public:
    struct promise_type {
        T current_value;
        
        Generator get_return_object() {
            return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        
        std::suspend_always yield_value(T value) {
            current_value = value;
            return {};
        }
        
        void return_void() {}
        void unhandled_exception() {}
    };
    
    std::coroutine_handle<promise_type> handle;
    
    explicit Generator(std::coroutine_handle<promise_type> h) : handle(h) {}
    
    ~Generator() {
        if (handle) {
            handle.destroy();
        }
    }
    
    // Move-only type
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;
    
    Generator(Generator&& other) noexcept : handle(other.handle) {
        other.handle = {};
    }
    
    Generator& operator=(Generator&& other) noexcept {
        if (this != &other) {
            if (handle) {
                handle.destroy();
            }
            handle = other.handle;
            other.handle = {};
        }
        return *this;
    }
    
    // Iterator interface
    class iterator {
    public:
        std::coroutine_handle<promise_type> handle;
        
        iterator(std::coroutine_handle<promise_type> h) : handle(h) {}
        
        iterator& operator++() {
            handle.resume();
            return *this;
        }
        
        T operator*() const {
            return handle.promise().current_value;
        }
        
        bool operator!=(const iterator& other) const {
            return handle != other.handle;
        }
    };
    
    iterator begin() {
        if (handle) {
            handle.resume();
            if (handle.done()) {
                return end();
            }
        }
        return iterator{handle};
    }
    
    iterator end() {
        return iterator{nullptr};
    }
};

// Simple generator examples
Generator<int> fibonacci_generator(int count) {
    std::cout << "Starting Fibonacci generator for " << count << " numbers" << std::endl;
    
    if (count <= 0) co_return;
    
    int a = 0, b = 1;
    
    for (int i = 0; i < count; ++i) {
        if (i == 0) {
            co_yield a;
        } else if (i == 1) {
            co_yield b;
        } else {
            int next = a + b;
            a = b;
            b = next;
            co_yield next;
        }
    }
    
    std::cout << "Fibonacci generator completed" << std::endl;
}

Generator<int> range_generator(int start, int end, int step = 1) {
    std::cout << "Range generator: " << start << " to " << end << " step " << step << std::endl;
    
    for (int i = start; i < end; i += step) {
        co_yield i;
    }
}

Generator<std::string> word_generator(const std::vector<std::string>& words) {
    std::cout << "Word generator with " << words.size() << " words" << std::endl;
    
    for (const auto& word : words) {
        co_yield word;
    }
}

// =============================================================================
// ASYNC TASK COROUTINE
// =============================================================================

template<typename T>
class Task {
public:
    struct promise_type {
        T result;
        std::exception_ptr exception;
        
        Task get_return_object() {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        
        void return_value(T value) {
            result = value;
        }
        
        void unhandled_exception() {
            exception = std::current_exception();
        }
    };
    
    std::coroutine_handle<promise_type> handle;
    
    explicit Task(std::coroutine_handle<promise_type> h) : handle(h) {}
    
    ~Task() {
        if (handle) {
            handle.destroy();
        }
    }
    
    // Move-only type
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    
    Task(Task&& other) noexcept : handle(other.handle) {
        other.handle = {};
    }
    
    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (handle) {
                handle.destroy();
            }
            handle = other.handle;
            other.handle = {};
        }
        return *this;
    }
    
    T get() {
        if (!handle.done()) {
            // In a real implementation, this would wait for completion
            std::cout << "Task not yet completed, waiting..." << std::endl;
        }
        
        if (handle.promise().exception) {
            std::rethrow_exception(handle.promise().exception);
        }
        
        return handle.promise().result;
    }
    
    bool is_ready() const {
        return handle && handle.done();
    }
};

// Async task examples
Task<int> async_computation(int value) {
    std::cout << "Starting async computation with value: " << value << std::endl;
    
    // Simulate async work
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    int result = value * value + 10;
    std::cout << "Async computation completed: " << result << std::endl;
    
    co_return result;
}

Task<std::string> async_string_processing(const std::string& input) {
    std::cout << "Processing string: \"" << input << "\"" << std::endl;
    
    // Simulate processing time
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    std::string result = "PROCESSED: " + input;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    
    std::cout << "String processing completed: \"" << result << "\"" << std::endl;
    co_return result;
}

// =============================================================================
// AWAITABLE TYPES
// =============================================================================

class Timer {
public:
    Timer(std::chrono::milliseconds duration) : duration_(duration) {}
    
    bool await_ready() const { return false; }
    
    void await_suspend(std::coroutine_handle<> handle) {
        std::thread([handle, duration = duration_]() {
            std::this_thread::sleep_for(duration);
            handle.resume();
        }).detach();
    }
    
    void await_resume() {
        std::cout << "Timer completed after " << duration_.count() << "ms" << std::endl;
    }
    
private:
    std::chrono::milliseconds duration_;
};

class ThreadPoolAwaiter {
public:
    ThreadPoolAwaiter(std::function<void()> work) : work_(std::move(work)) {}
    
    bool await_ready() const { return false; }
    
    void await_suspend(std::coroutine_handle<> handle) {
        std::thread([work = work_, handle]() {
            work();
            handle.resume();
        }).detach();
    }
    
    void await_resume() {
        std::cout << "Thread pool work completed" << std::endl;
    }
    
private:
    std::function<void()> work_;
};

// =============================================================================
// ADVANCED COROUTINE EXAMPLES
// =============================================================================

Task<void> timer_example() {
    std::cout << "Starting timer example" << std::endl;
    
    co_await Timer(std::chrono::milliseconds(500));
    std::cout << "First timer done" << std::endl;
    
    co_await Timer(std::chrono::milliseconds(300));
    std::cout << "Second timer done" << std::endl;
    
    std::cout << "Timer example completed" << std::endl;
}

Task<int> parallel_work_example() {
    std::cout << "Starting parallel work example" << std::endl;
    
    int result = 0;
    
    // Simulate parallel work
    co_await ThreadPoolAwaiter([&result]() {
        std::cout << "Worker 1: Computing..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        result += 100;
        std::cout << "Worker 1: Done" << std::endl;
    });
    
    co_await ThreadPoolAwaiter([&result]() {
        std::cout << "Worker 2: Computing..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        result += 200;
        std::cout << "Worker 2: Done" << std::endl;
    });
    
    std::cout << "Parallel work completed with result: " << result << std::endl;
    co_return result;
}

// =============================================================================
// COROUTINE PIPELINE
// =============================================================================

Generator<int> pipeline_stage1(int count) {
    std::cout << "Pipeline Stage 1: Generating numbers" << std::endl;
    for (int i = 1; i <= count; ++i) {
        std::cout << "  Stage 1 producing: " << i << std::endl;
        co_yield i;
    }
}

Generator<int> pipeline_stage2(Generator<int> input) {
    std::cout << "Pipeline Stage 2: Squaring numbers" << std::endl;
    for (auto value : input) {
        int squared = value * value;
        std::cout << "  Stage 2 transforming " << value << " -> " << squared << std::endl;
        co_yield squared;
    }
}

Generator<int> pipeline_stage3(Generator<int> input) {
    std::cout << "Pipeline Stage 3: Filtering even numbers" << std::endl;
    for (auto value : input) {
        if (value % 2 == 0) {
            std::cout << "  Stage 3 passing: " << value << std::endl;
            co_yield value;
        } else {
            std::cout << "  Stage 3 filtering out: " << value << std::endl;
        }
    }
}

// =============================================================================
// DEMONSTRATION FUNCTIONS
// =============================================================================

void demonstrate_generators() {
    std::cout << "\n=== Generator Coroutines ===" << std::endl;
    
    // Fibonacci generator
    std::cout << "\nFibonacci sequence:" << std::endl;
    auto fib = fibonacci_generator(8);
    for (auto value : fib) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    
    // Range generator
    std::cout << "\nRange 0 to 10 step 2:" << std::endl;
    auto range = range_generator(0, 10, 2);
    for (auto value : range) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    
    // Word generator
    std::cout << "\nWord sequence:" << std::endl;
    std::vector<std::string> words = {"Hello", "Coroutines", "World", "C++20"};
    auto word_gen = word_generator(words);
    for (const auto& word : word_gen) {
        std::cout << word << " ";
    }
    std::cout << std::endl;
}

void demonstrate_async_tasks() {
    std::cout << "\n=== Async Task Coroutines ===" << std::endl;
    
    // Simple async computation
    auto task1 = async_computation(5);
    auto result1 = task1.get();
    std::cout << "Async computation result: " << result1 << std::endl;
    
    // String processing
    auto task2 = async_string_processing("hello coroutines");
    auto result2 = task2.get();
    std::cout << "String processing result: " << result2 << std::endl;
}

void demonstrate_awaitables() {
    std::cout << "\n=== Awaitable Types ===" << std::endl;
    
    // Timer example
    auto timer_task = timer_example();
    // In a real implementation, we would wait for completion
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    // Parallel work example
    auto parallel_task = parallel_work_example();
    // In a real implementation, we would wait for completion
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

void demonstrate_pipeline() {
    std::cout << "\n=== Coroutine Pipeline ===" << std::endl;
    
    // Create pipeline: generate -> square -> filter even
    auto stage1 = pipeline_stage1(5);
    auto stage2 = pipeline_stage2(std::move(stage1));
    auto stage3 = pipeline_stage3(std::move(stage2));
    
    std::cout << "\nFinal pipeline results:" << std::endl;
    for (auto value : stage3) {
        std::cout << "Final result: " << value << std::endl;
    }
}

void demonstrate_coroutine_benefits() {
    std::cout << "\n=== Coroutine Benefits ===" << std::endl;
    
    std::cout << "1. Simplified Async Programming:" << std::endl;
    std::cout << "   • Linear code flow for async operations" << std::endl;
    std::cout << "   • No callback hell or complex state machines" << std::endl;
    
    std::cout << "\n2. Lazy Evaluation:" << std::endl;
    std::cout << "   • Generators produce values on demand" << std::endl;
    std::cout << "   • Memory efficient for large sequences" << std::endl;
    
    std::cout << "\n3. Composability:" << std::endl;
    std::cout << "   • Easy to chain and combine coroutines" << std::endl;
    std::cout << "   • Pipeline processing patterns" << std::endl;
    
    std::cout << "\n4. Performance:" << std::endl;
    std::cout << "   • Stackless coroutines are lightweight" << std::endl;
    std::cout << "   • Efficient context switching" << std::endl;
    
    std::cout << "\n5. Error Handling:" << std::endl;
    std::cout << "   • Standard exception propagation" << std::endl;
    std::cout << "   • Clean resource management with RAII" << std::endl;
}

void demonstrate_real_world_patterns() {
    std::cout << "\n=== Real-World Patterns ===" << std::endl;
    
    std::cout << "Common Coroutine Use Cases:" << std::endl;
    std::cout << "• Network I/O operations" << std::endl;
    std::cout << "• File processing pipelines" << std::endl;
    std::cout << "• Event stream processing" << std::endl;
    std::cout << "• Game state machines" << std::endl;
    std::cout << "• Data transformation pipelines" << std::endl;
    std::cout << "• Async web request handling" << std::endl;
    
    std::cout << "\nIntegration with:" << std::endl;
    std::cout << "• std::future and std::async" << std::endl;
    std::cout << "• Thread pools and executors" << std::endl;
    std::cout << "• Event loops (like epoll)" << std::endl;
    std::cout << "• Networking libraries (ASIO)" << std::endl;
}

int main() {
    std::cout << "C++20 COROUTINES BASICS\n";
    std::cout << "=======================\n";
    
    try {
        demonstrate_generators();
        demonstrate_async_tasks();
        demonstrate_awaitables();
        demonstrate_pipeline();
        demonstrate_coroutine_benefits();
        demonstrate_real_world_patterns();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nKey Coroutine Concepts:" << std::endl;
    std::cout << "• co_yield for generators and lazy evaluation" << std::endl;
    std::cout << "• co_await for asynchronous operations" << std::endl;
    std::cout << "• co_return for returning values" << std::endl;
    std::cout << "• Promise types define coroutine behavior" << std::endl;
    std::cout << "• Awaitable types enable custom suspension points" << std::endl;
    std::cout << "• Stackless design for efficient memory usage" << std::endl;
    
    return 0;
}

/*
COROUTINE SYNTAX SUMMARY:

1. Generator Function:
Generator<T> my_generator() {
    co_yield value1;
    co_yield value2;
    // ...
}

2. Async Function:
Task<T> my_async_function() {
    auto result = co_await some_awaitable;
    co_return result;
}

3. Promise Type Requirements:
struct promise_type {
    auto get_return_object();
    auto initial_suspend();
    auto final_suspend() noexcept;
    void unhandled_exception();
    
    // For generators:
    auto yield_value(T value);
    void return_void();
    
    // For tasks:
    void return_value(T value);
};

4. Awaitable Type Requirements:
struct MyAwaitable {
    bool await_ready();
    void await_suspend(std::coroutine_handle<>);
    T await_resume();
};

5. Usage Patterns:
// Generator usage
for (auto value : my_generator()) {
    process(value);
}

// Async usage
auto task = my_async_function();
auto result = co_await task;
*/ 