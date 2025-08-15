"""
Python Generators and Coroutines: Advanced Generator Patterns and Coroutine Programming
Implementation-focused with minimal comments, maximum functionality coverage
"""

import sys
import time
import random
from typing import Generator, Iterator, Any, Optional, Callable
from collections import deque
import inspect

# Basic generator patterns
def simple_generator():
    """Basic generator function"""
    yield 1
    yield 2
    yield 3

def infinite_counter(start=0, step=1):
    """Infinite generator"""
    current = start
    while True:
        yield current
        current += step

def fibonacci_generator():
    """Fibonacci sequence generator"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def range_generator(start, stop, step=1):
    """Custom range implementation"""
    current = start
    while current < stop:
        yield current
        current += step

def basic_generator_demo():
    # Test simple generator
    simple_results = list(simple_generator())
    
    # Test infinite counter (limited)
    counter = infinite_counter(10, 2)
    counter_results = [next(counter) for _ in range(5)]
    
    # Test fibonacci (limited)
    fib = fibonacci_generator()
    fib_results = [next(fib) for _ in range(10)]
    
    # Test custom range
    range_results = list(range_generator(0, 10, 2))
    
    return {
        "simple_generator": simple_results,
        "infinite_counter": counter_results,
        "fibonacci_sequence": fib_results,
        "custom_range": range_results
    }

# Generator state and methods
def stateful_generator():
    """Generator that demonstrates state management"""
    state = {"count": 0, "total": 0}
    
    while True:
        value = yield state["total"]
        if value is not None:
            state["count"] += 1
            state["total"] += value

def closable_generator():
    """Generator that handles close() properly"""
    try:
        for i in range(10):
            yield i
    except GeneratorExit:
        print("Generator is being closed")
        return "Cleanup completed"
    finally:
        print("Generator cleanup")

def exception_generator():
    """Generator that can receive exceptions"""
    try:
        for i in range(5):
            try:
                yield i
            except ValueError as e:
                yield f"Caught ValueError: {e}"
            except TypeError as e:
                yield f"Caught TypeError: {e}"
    except Exception as e:
        yield f"Caught general exception: {e}"

def generator_methods_demo():
    # Test send() method
    stateful = stateful_generator()
    next(stateful)  # Prime the generator
    
    send_results = []
    send_results.append(stateful.send(10))
    send_results.append(stateful.send(20))
    send_results.append(stateful.send(30))
    
    # Test close() method
    closable = closable_generator()
    close_results = [next(closable) for _ in range(3)]
    closable.close()
    
    # Test throw() method
    exception_gen = exception_generator()
    throw_results = []
    
    throw_results.append(next(exception_gen))  # 0
    throw_results.append(exception_gen.throw(ValueError, "Test error"))
    throw_results.append(next(exception_gen))  # 1
    throw_results.append(exception_gen.throw(TypeError, "Type error"))
    
    return {
        "send_method": send_results,
        "close_method": close_results,
        "throw_method": throw_results
    }

# Generator delegation with yield from
def sub_generator(n):
    """Sub-generator for delegation"""
    for i in range(n):
        yield f"sub_{i}"

def delegating_generator():
    """Generator that delegates to sub-generators"""
    yield "start"
    yield from sub_generator(3)
    yield "middle"
    yield from sub_generator(2)
    yield "end"

def flattening_generator(nested_list):
    """Flatten nested structures using yield from"""
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            yield from flattening_generator(item)
        else:
            yield item

def pipeline_generator():
    """Generator pipeline using yield from"""
    def source():
        for i in range(10):
            yield i
    
    def filter_even(gen):
        for value in gen:
            if value % 2 == 0:
                yield value
    
    def square(gen):
        for value in gen:
            yield value ** 2
    
    # Chain generators using yield from
    yield from square(filter_even(source()))

def yield_from_demo():
    # Test basic delegation
    delegation_results = list(delegating_generator())
    
    # Test flattening
    nested_data = [1, [2, 3], [4, [5, 6]], 7]
    flattened_results = list(flattening_generator(nested_data))
    
    # Test pipeline
    pipeline_results = list(pipeline_generator())
    
    return {
        "delegation": delegation_results,
        "flattening": flattened_results,
        "pipeline": pipeline_results
    }

# Coroutine patterns
def simple_coroutine():
    """Basic coroutine pattern"""
    result = None
    while True:
        value = yield result
        if value is not None:
            result = value * 2

def accumulator_coroutine():
    """Coroutine that accumulates values"""
    total = 0
    count = 0
    while True:
        value = yield {"total": total, "count": count, "average": total/count if count > 0 else 0}
        if value is not None:
            total += value
            count += 1

def coroutine_decorator(func):
    """Decorator to prime coroutines"""
    def wrapper(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)  # Prime the coroutine
        return gen
    return wrapper

@coroutine_decorator
def grep_coroutine(pattern):
    """Coroutine that filters lines matching a pattern"""
    print(f"Looking for pattern: {pattern}")
    while True:
        line = yield
        if pattern in line:
            print(f"Found: {line}")

@coroutine_decorator
def broadcast_coroutine(targets):
    """Coroutine that broadcasts to multiple targets"""
    while True:
        value = yield
        for target in targets:
            target.send(value)

def coroutine_demo():
    # Test simple coroutine
    simple = simple_coroutine()
    next(simple)  # Prime
    
    simple_results = []
    simple_results.append(simple.send(5))
    simple_results.append(simple.send(10))
    simple_results.append(simple.send(3))
    
    # Test accumulator
    accumulator = accumulator_coroutine()
    next(accumulator)  # Prime
    
    accumulator_results = []
    accumulator_results.append(accumulator.send(10))
    accumulator_results.append(accumulator.send(20))
    accumulator_results.append(accumulator.send(30))
    
    # Test grep coroutine (output captured in print statements)
    grep = grep_coroutine("python")
    test_lines = ["hello world", "python is great", "java programming", "I love python"]
    
    for line in test_lines:
        grep.send(line)
    
    return {
        "simple_coroutine": simple_results,
        "accumulator_coroutine": accumulator_results,
        "grep_test_completed": True
    }

# Generator expressions vs functions
def generator_expression_demo():
    # Generator expression
    squares_expr = (x**2 for x in range(10))
    
    # Equivalent generator function
    def squares_func():
        for x in range(10):
            yield x**2
    
    # Memory comparison (simplified)
    expr_memory = sys.getsizeof(squares_expr)
    func_memory = sys.getsizeof(squares_func())
    
    # Performance comparison
    import timeit
    
    expr_time = timeit.timeit(lambda: list(x**2 for x in range(100)), number=1000)
    func_time = timeit.timeit(lambda: list(squares_func()), number=1000)
    
    # Chained generator expressions
    chained = (x for x in (y**2 for y in range(10)) if x % 2 == 0)
    chained_results = list(chained)
    
    return {
        "memory_comparison": {
            "expression": expr_memory,
            "function": func_memory
        },
        "performance_comparison": {
            "expression_time": f"{expr_time:.6f}s",
            "function_time": f"{func_time:.6f}s"
        },
        "chained_expressions": chained_results
    }

# Advanced generator patterns
class GeneratorClass:
    """Class that implements generator protocol"""
    
    def __init__(self, max_value):
        self.max_value = max_value
    
    def __iter__(self):
        return self.generator()
    
    def generator(self):
        for i in range(self.max_value):
            yield i ** 2

def sliding_window(iterable, window_size):
    """Generator for sliding window over iterable"""
    iterator = iter(iterable)
    window = deque(maxlen=window_size)
    
    # Fill initial window
    for _ in range(window_size):
        try:
            window.append(next(iterator))
        except StopIteration:
            return
    
    yield tuple(window)
    
    # Slide the window
    for item in iterator:
        window.append(item)
        yield tuple(window)

def batched(iterable, batch_size):
    """Generator that yields batches of items"""
    iterator = iter(iterable)
    while True:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(iterator))
            except StopIteration:
                if batch:
                    yield batch
                return
        yield batch

def interleave(*iterables):
    """Generator that interleaves multiple iterables"""
    iterators = [iter(it) for it in iterables]
    while iterators:
        for it in list(iterators):
            try:
                yield next(it)
            except StopIteration:
                iterators.remove(it)

def cycle_limited(iterable, times):
    """Generator that cycles through iterable limited times"""
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    
    for _ in range(times - 1):
        for element in saved:
            yield element

def advanced_patterns_demo():
    # Test generator class
    gen_class = GeneratorClass(5)
    class_results = list(gen_class)
    
    # Test sliding window
    window_results = list(sliding_window(range(10), 3))
    
    # Test batching
    batch_results = list(batched(range(13), 3))
    
    # Test interleaving
    interleave_results = list(interleave([1, 2, 3], ['a', 'b', 'c', 'd'], [10, 20]))
    
    # Test limited cycling
    cycle_results = list(cycle_limited([1, 2, 3], 3))
    
    return {
        "generator_class": class_results,
        "sliding_window": window_results,
        "batching": batch_results,
        "interleaving": interleave_results,
        "limited_cycle": cycle_results
    }

# Generator pipelines and composition
def data_pipeline():
    """Demonstrate generator pipeline for data processing"""
    
    def data_source():
        """Generate sample data"""
        for i in range(100):
            yield {
                "id": i,
                "value": random.randint(1, 100),
                "category": random.choice(["A", "B", "C"])
            }
    
    def filter_by_category(data_stream, category):
        """Filter data by category"""
        for item in data_stream:
            if item["category"] == category:
                yield item
    
    def transform_value(data_stream, multiplier):
        """Transform values"""
        for item in data_stream:
            item = item.copy()
            item["value"] *= multiplier
            yield item
    
    def add_computed_field(data_stream):
        """Add computed field"""
        for item in data_stream:
            item = item.copy()
            item["computed"] = item["value"] ** 2
            yield item
    
    def take_first(data_stream, n):
        """Take first n items"""
        count = 0
        for item in data_stream:
            if count >= n:
                break
            yield item
            count += 1
    
    # Build pipeline
    pipeline = take_first(
        add_computed_field(
            transform_value(
                filter_by_category(data_source(), "A"),
                2
            )
        ),
        5
    )
    
    return list(pipeline)

def functional_pipeline(*functions):
    """Create a functional pipeline generator"""
    def pipeline(data):
        for func in functions:
            if inspect.isgeneratorfunction(func):
                # It's a generator function, apply it
                data = func(data)
            else:
                # It's a regular function, apply to each item
                data = (func(item) for item in data)
        return data
    return pipeline

def pipeline_demo():
    # Test data pipeline
    pipeline_results = data_pipeline()
    
    # Test functional pipeline
    def add_one(x):
        return x + 1
    
    def multiply_two(x):
        return x * 2
    
    def filter_even(data):
        for item in data:
            if item % 2 == 0:
                yield item
    
    functional_pipe = functional_pipeline(add_one, multiply_two, filter_even)
    functional_results = list(functional_pipe(range(10)))
    
    return {
        "data_pipeline": pipeline_results,
        "functional_pipeline": functional_results
    }

# Memory efficiency and lazy evaluation
def memory_efficient_processing():
    """Demonstrate memory-efficient data processing"""
    
    def large_dataset():
        """Simulate large dataset without loading into memory"""
        for i in range(1000000):
            yield {"id": i, "value": i ** 2}
    
    def process_chunk(chunk):
        """Process a chunk of data"""
        return sum(item["value"] for item in chunk)
    
    # Process in chunks to maintain constant memory usage
    chunk_size = 1000
    results = []
    chunk = []
    
    for item in large_dataset():
        chunk.append(item)
        if len(chunk) >= chunk_size:
            results.append(process_chunk(chunk))
            chunk = []
            if len(results) >= 10:  # Limit for demo
                break
    
    if chunk:  # Process remaining items
        results.append(process_chunk(chunk))
    
    return results

def lazy_evaluation_demo():
    """Demonstrate lazy evaluation benefits"""
    
    def expensive_computation(x):
        """Simulate expensive computation"""
        time.sleep(0.001)  # Simulate work
        return x ** 3
    
    # Eager evaluation (list comprehension)
    start_time = time.time()
    eager_results = [expensive_computation(x) for x in range(100)]
    eager_time = time.time() - start_time
    
    # Lazy evaluation (generator)
    start_time = time.time()
    lazy_gen = (expensive_computation(x) for x in range(100))
    lazy_creation_time = time.time() - start_time
    
    # Only consume first 5 items
    start_time = time.time()
    lazy_results = [next(lazy_gen) for _ in range(5)]
    lazy_consumption_time = time.time() - start_time
    
    return {
        "memory_efficient_processing": memory_efficient_processing(),
        "timing_comparison": {
            "eager_time": f"{eager_time:.4f}s",
            "lazy_creation_time": f"{lazy_creation_time:.4f}s",
            "lazy_consumption_time": f"{lazy_consumption_time:.4f}s",
            "lazy_advantage": lazy_creation_time < eager_time
        },
        "lazy_results": lazy_results[:5]
    }

# Generator-based state machines
class StateMachine:
    """Generator-based state machine"""
    
    def __init__(self):
        self.state_gen = self._state_machine()
        next(self.state_gen)  # Prime the generator
    
    def _state_machine(self):
        state = "START"
        
        while True:
            if state == "START":
                action = yield f"In START state"
                if action == "begin":
                    state = "RUNNING"
                elif action == "exit":
                    state = "END"
            
            elif state == "RUNNING":
                action = yield f"In RUNNING state"
                if action == "pause":
                    state = "PAUSED"
                elif action == "stop":
                    state = "END"
            
            elif state == "PAUSED":
                action = yield f"In PAUSED state"
                if action == "resume":
                    state = "RUNNING"
                elif action == "stop":
                    state = "END"
            
            elif state == "END":
                yield "In END state - machine stopped"
                break
    
    def send_action(self, action):
        try:
            return self.state_gen.send(action)
        except StopIteration:
            return "Machine stopped"

def generator_coroutine_protocol():
    """Implement a simple protocol using generators"""
    
    def server():
        """Server coroutine"""
        clients = []
        while True:
            message = yield
            if message["type"] == "connect":
                clients.append(message["client"])
                print(f"Client {message['client']} connected")
            elif message["type"] == "broadcast":
                for client in clients:
                    print(f"Sent to {client}: {message['data']}")
            elif message["type"] == "disconnect":
                if message["client"] in clients:
                    clients.remove(message["client"])
                    print(f"Client {message['client']} disconnected")
    
    # Setup server
    server_gen = server()
    next(server_gen)  # Prime
    
    # Simulate protocol
    messages = [
        {"type": "connect", "client": "client1"},
        {"type": "connect", "client": "client2"},
        {"type": "broadcast", "data": "Hello everyone!"},
        {"type": "disconnect", "client": "client1"},
        {"type": "broadcast", "data": "Goodbye!"}
    ]
    
    for message in messages:
        server_gen.send(message)
    
    return "Protocol simulation completed"

def state_machine_demo():
    # Test state machine
    sm = StateMachine()
    
    state_transitions = []
    actions = ["begin", "pause", "resume", "stop"]
    
    for action in actions:
        result = sm.send_action(action)
        state_transitions.append({"action": action, "result": result})
    
    # Test protocol
    protocol_result = generator_coroutine_protocol()
    
    return {
        "state_machine": state_transitions,
        "protocol_simulation": protocol_result
    }

# Real-world generator applications
def log_parser_generator(log_lines):
    """Parse log files using generators"""
    for line in log_lines:
        if line.strip():
            parts = line.strip().split(" ", 3)
            if len(parts) >= 4:
                yield {
                    "timestamp": parts[0] + " " + parts[1],
                    "level": parts[2],
                    "message": parts[3]
                }

def csv_reader_generator(csv_content):
    """CSV reader using generators"""
    lines = csv_content.strip().split('\n')
    if not lines:
        return
    
    headers = [h.strip() for h in lines[0].split(',')]
    
    for line in lines[1:]:
        values = [v.strip() for v in line.split(',')]
        if len(values) == len(headers):
            yield dict(zip(headers, values))

def web_scraper_simulator():
    """Simulate web scraping with generators"""
    def fetch_pages():
        """Simulate fetching pages"""
        urls = [f"http://example.com/page{i}" for i in range(1, 6)]
        for url in urls:
            # Simulate HTTP request
            time.sleep(0.01)  # Simulate network delay
            yield {
                "url": url,
                "content": f"Content from {url}",
                "status": 200
            }
    
    def extract_data(pages):
        """Extract data from pages"""
        for page in pages:
            if page["status"] == 200:
                # Simulate data extraction
                yield {
                    "url": page["url"],
                    "title": f"Title from {page['url']}",
                    "word_count": len(page["content"].split())
                }
    
    def filter_valid_data(data_stream):
        """Filter valid data"""
        for item in data_stream:
            if item["word_count"] > 2:
                yield item
    
    # Create scraping pipeline
    scraping_pipeline = filter_valid_data(extract_data(fetch_pages()))
    return list(scraping_pipeline)

def real_world_demo():
    # Test log parser
    log_data = """
    2023-01-01 10:00:00 INFO Application started
    2023-01-01 10:01:00 DEBUG Loading configuration
    2023-01-01 10:02:00 ERROR Failed to connect to database
    2023-01-01 10:03:00 INFO Retrying connection
    """
    
    log_results = list(log_parser_generator(log_data.strip().split('\n')))
    
    # Test CSV reader
    csv_data = """
    name,age,city
    Alice,30,New York
    Bob,25,Los Angeles
    Charlie,35,Chicago
    """
    
    csv_results = list(csv_reader_generator(csv_data))
    
    # Test web scraper
    scraper_results = web_scraper_simulator()
    
    return {
        "log_parsing": log_results,
        "csv_reading": csv_results,
        "web_scraping": scraper_results
    }

# Comprehensive testing
def run_all_generator_demos():
    """Execute all generator and coroutine demonstrations"""
    demo_functions = [
        ('basic_generators', basic_generator_demo),
        ('generator_methods', generator_methods_demo),
        ('yield_from', yield_from_demo),
        ('coroutines', coroutine_demo),
        ('generator_expressions', generator_expression_demo),
        ('advanced_patterns', advanced_patterns_demo),
        ('pipeline_composition', pipeline_demo),
        ('lazy_evaluation', lazy_evaluation_demo),
        ('state_machines', state_machine_demo),
        ('real_world_applications', real_world_demo)
    ]
    
    results = {}
    for name, func in demo_functions:
        try:
            result = func()
            results[name] = result
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    print("=== Python Generators and Coroutines Demo ===")
    
    # Run all demonstrations
    all_results = run_all_generator_demos()
    
    for category, data in all_results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        
        if 'error' in data:
            print(f"  Error: {data['error']}")
            continue
        
        # Display results
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and len(value) > 3:
                    print(f"  {key}: {dict(list(value.items())[:3])}... (truncated)")
                elif isinstance(value, list) and len(value) > 5:
                    print(f"  {key}: {value[:5]}... (showing first 5)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Result: {data}")
    
    print("\n=== GENERATOR CONCEPTS ===")
    
    concepts = {
        "Generator Functions": "Functions that use yield keyword to produce values lazily",
        "Generator Expressions": "Compact syntax for creating generators: (expr for item in iterable)",
        "yield": "Keyword that produces a value and suspends function execution",
        "yield from": "Delegate to another generator or iterable",
        "send()": "Send a value to a generator using two-way communication",
        "throw()": "Send an exception to a generator",
        "close()": "Close a generator and trigger GeneratorExit exception",
        "Coroutines": "Generators that can receive data via send() method"
    }
    
    for concept, description in concepts.items():
        print(f"  {concept}: {description}")
    
    print("\n=== GENERATOR BENEFITS ===")
    
    benefits = [
        "Memory Efficiency: Process large datasets without loading into memory",
        "Lazy Evaluation: Compute values only when needed",
        "Infinite Sequences: Generate unlimited sequences efficiently",
        "Pipeline Processing: Chain operations without intermediate storage",
        "State Management: Maintain state between function calls",
        "Cooperative Multitasking: Implement coroutines for async-like behavior",
        "Clean Syntax: More readable than iterator classes",
        "Composability: Easy to combine and chain generators"
    ]
    
    for benefit in benefits:
        print(f"  • {benefit}")
    
    print("\n=== BEST PRACTICES ===")
    
    best_practices = [
        "Use generators for large datasets to save memory",
        "Prime coroutines with next() or use a decorator",
        "Handle GeneratorExit in finally blocks for cleanup",
        "Use yield from for generator delegation",
        "Consider generator expressions for simple transformations",
        "Chain generators to create processing pipelines",
        "Use itertools for advanced generator operations",
        "Document generator behavior and expected usage",
        "Test both normal and exceptional flows",
        "Consider async/await for I/O-bound coroutines"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== Generators and Coroutines Complete! ===")
    print("  Advanced generator patterns and coroutine programming mastered")
