"""
Python String Manipulation: Methods, Formatting, Regular Expressions
Implementation-focused with minimal comments, maximum functionality coverage
"""

import re
import string
import time
from typing import List, Dict, Tuple, Optional, Pattern
import unicodedata

# String methods and formatting
def basic_string_methods():
    text = "  Hello, World! Programming with Python  "
    
    # Case methods
    case_methods = {
        'lower': text.lower(),
        'upper': text.upper(),
        'title': text.title(),
        'capitalize': text.capitalize(),
        'swapcase': text.swapcase(),
        'casefold': text.casefold()  # Aggressive lowercase
    }
    
    # Whitespace methods
    whitespace_methods = {
        'strip': text.strip(),
        'lstrip': text.lstrip(),
        'rstrip': text.rstrip(),
        'strip_specific': text.strip(' !'),
        'center': text.strip().center(50, '-'),
        'ljust': text.strip().ljust(30, '*'),
        'rjust': text.strip().rjust(30, '*')
    }
    
    # Search and check methods
    search_text = "Programming"
    search_methods = {
        'startswith': text.startswith("  Hello"),
        'endswith': text.endswith("Python  "),
        'find': text.find("World"),
        'rfind': text.rfind("o"),
        'index': text.index("Programming"),
        'count': text.count("o"),
        'in_operator': "Python" in text
    }
    
    # Character type checking
    test_strings = ["Hello123", "123", "hello", "HELLO", "   ", "Hello World"]
    char_checks = {}
    for s in test_strings:
        char_checks[s] = {
            'isalnum': s.isalnum(),
            'isalpha': s.isalpha(),
            'isdigit': s.isdigit(),
            'isspace': s.isspace(),
            'islower': s.islower(),
            'isupper': s.isupper(),
            'istitle': s.istitle()
        }
    
    return {
        'case_methods': case_methods,
        'whitespace_methods': whitespace_methods,
        'search_methods': search_methods,
        'character_checks': char_checks
    }

def string_splitting_joining():
    text = "apple,banana,cherry;orange:grape"
    csv_data = "name,age,city\nAlice,30,NYC\nBob,25,LA"
    
    # Splitting methods
    split_results = {
        'default_split': "hello world python".split(),
        'comma_split': text.split(','),
        'multi_delimiter': re.split('[,;:]', text),
        'maxsplit': "a-b-c-d-e".split('-', 2),
        'rsplit': "a.b.c.d".rsplit('.', 1),
        'splitlines': csv_data.splitlines(),
        'partition': "email@domain.com".partition('@'),
        'rpartition': "path/to/file.txt".rpartition('/')
    }
    
    # Joining methods
    words = ['Python', 'is', 'awesome']
    numbers = [1, 2, 3, 4, 5]
    
    join_results = {
        'space_join': ' '.join(words),
        'comma_join': ', '.join(words),
        'empty_join': ''.join(words),
        'newline_join': '\n'.join(words),
        'numbers_join': '-'.join(map(str, numbers)),
        'path_join': '/'.join(['home', 'user', 'documents', 'file.txt'])
    }
    
    return {
        'splitting': split_results,
        'joining': join_results
    }

def string_replacement_translation():
    text = "Hello World! Hello Python! Hello Programming!"
    
    # Replacement methods
    replacement_results = {
        'replace_basic': text.replace('Hello', 'Hi'),
        'replace_limited': text.replace('Hello', 'Hi', 2),
        'replace_case_sensitive': text.replace('hello', 'hi'),  # No match
        'multiple_replacements': text.replace('Hello', 'Hi').replace('!', '.')
    }
    
    # Translation tables
    # Remove vowels
    vowels = 'aeiouAEIOU'
    remove_vowels = str.maketrans('', '', vowels)
    
    # Caesar cipher (shift by 3)
    alphabet = string.ascii_lowercase
    shifted = alphabet[3:] + alphabet[:3]
    caesar_table = str.maketrans(alphabet, shifted)
    
    # Character substitution
    substitute_table = str.maketrans('aeiou', '12345')
    
    translation_results = {
        'remove_vowels': "Hello World".translate(remove_vowels),
        'caesar_cipher': "hello world".translate(caesar_table),
        'substitute_vowels': "hello world".translate(substitute_table),
        'remove_punctuation': "Hello, World!".translate(str.maketrans('', '', string.punctuation))
    }
    
    return {
        'replacements': replacement_results,
        'translations': translation_results
    }

# Advanced string formatting
def string_formatting_methods():
    name = "Alice"
    age = 30
    pi = 3.14159265359
    
    # Old-style formatting
    old_style = {
        'basic': "Name: %s, Age: %d" % (name, age),
        'float_precision': "Pi: %.2f" % pi,
        'zero_padding': "Number: %05d" % 42,
        'dictionary': "%(name)s is %(age)d years old" % {'name': name, 'age': age}
    }
    
    # .format() method
    format_method = {
        'positional': "Name: {}, Age: {}".format(name, age),
        'indexed': "Age: {1}, Name: {0}".format(name, age),
        'named': "Name: {name}, Age: {age}".format(name=name, age=age),
        'precision': "Pi: {:.3f}".format(pi),
        'padding': "Number: {:05d}".format(42),
        'alignment': "'{:>10}' '{:<10}' '{:^10}'".format('test', 'test', 'test')
    }
    
    # f-strings (Python 3.6+)
    f_strings = {
        'basic': f"Name: {name}, Age: {age}",
        'expressions': f"Next year: {age + 1}",
        'precision': f"Pi: {pi:.4f}",
        'padding': f"Number: {42:05d}",
        'alignment': f"'{name:>10}' '{name:<10}' '{name:^10}'",
        'date_format': f"Today: {time.strftime('%Y-%m-%d')}",
        'debug': f"{name=}, {age=}"  # Python 3.8+
    }
    
    return {
        'old_style': old_style,
        'format_method': format_method,
        'f_strings': f_strings
    }

def advanced_formatting_patterns():
    # Number formatting
    numbers = [1234567.89, 0.00123, 1000000, -45.678]
    
    number_formatting = {}
    for num in numbers:
        number_formatting[str(num)] = {
            'comma_separator': f"{num:,}",
            'percentage': f"{num:.1%}" if abs(num) <= 1 else f"{num/100:.1%}",
            'scientific': f"{num:.2e}",
            'fixed_width': f"{num:10.2f}",
            'sign_always': f"{num:+.2f}"
        }
    
    # Advanced alignment and formatting
    data = [('Alice', 30, 85000), ('Bob', 25, 75000), ('Charlie', 35, 95000)]
    
    table_formatting = {
        'header': f"{'Name':<10} {'Age':>5} {'Salary':>10}",
        'separator': '-' * 27,
        'rows': [f"{name:<10} {age:>5} ${salary:>9,}" for name, age, salary in data]
    }
    
    # Format specifications
    value = 123.456789
    format_specs = {
        'general': f"{value:g}",
        'binary': f"{int(value):b}",
        'octal': f"{int(value):o}",
        'hexadecimal': f"{int(value):x}",
        'uppercase_hex': f"{int(value):X}",
        'character': f"{65:c}",  # ASCII 'A'
        'repr': f"{value!r}",
        'str': f"{value!s}"
    }
    
    return {
        'number_formatting': number_formatting,
        'table_formatting': table_formatting,
        'format_specifications': format_specs
    }

# Regular expression patterns
def regex_basics():
    text = "Contact: john.doe@email.com, phone: 123-456-7890, date: 2023-12-25"
    
    # Basic patterns
    patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}-\d{3}-\d{4}\b',
        'date': r'\b\d{4}-\d{2}-\d{2}\b',
        'word': r'\b\w+\b',
        'digits': r'\d+',
        'uppercase': r'[A-Z]+',
        'email_parts': r'(\w+)\.(\w+)@(\w+)\.(\w+)'
    }
    
    regex_results = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        regex_results[name] = matches
    
    # Match objects and groups
    email_match = re.search(patterns['email_parts'], text)
    if email_match:
        match_info = {
            'full_match': email_match.group(0),
            'groups': email_match.groups(),
            'group_dict': email_match.groupdict() if hasattr(email_match, 'groupdict') else {},
            'start_end': (email_match.start(), email_match.end())
        }
    else:
        match_info = None
    
    return {
        'pattern_matches': regex_results,
        'match_details': match_info
    }

def regex_advanced_patterns():
    # Compiled patterns for efficiency
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE)
    url_pattern = re.compile(r'https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?')
    
    text = """
    Visit our website at https://example.com or email us at info@example.com
    Also check https://subdomain.example.org:8080/path?param=value#section
    Contact: admin@test.co.uk or support@company.net
    """
    
    # Pattern matching methods
    compiled_results = {
        'findall_emails': email_pattern.findall(text),
        'findall_urls': url_pattern.findall(text),
        'search_first_email': email_pattern.search(text).group() if email_pattern.search(text) else None,
        'finditer_emails': [match.group() for match in email_pattern.finditer(text)]
    }
    
    # Substitution patterns
    phone_text = "Call 123-456-7890 or 987-654-3210 for assistance"
    
    substitution_results = {
        'hide_phones': re.sub(r'\d{3}-\d{3}-\d{4}', 'XXX-XXX-XXXX', phone_text),
        'format_phones': re.sub(r'(\d{3})-(\d{3})-(\d{4})', r'(\1) \2-\3', phone_text),
        'remove_extra_spaces': re.sub(r'\s+', ' ', '  Multiple   spaces    here  '),
        'split_on_punctuation': re.split(r'[,;.!?]', "Hello, world! How are you? Fine.")
    }
    
    # Named groups
    date_pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
    date_text = "Important dates: 2023-12-25 and 2024-01-01"
    
    named_groups = []
    for match in re.finditer(date_pattern, date_text):
        named_groups.append({
            'full': match.group(),
            'year': match.group('year'),
            'month': match.group('month'),
            'day': match.group('day'),
            'as_dict': match.groupdict()
        })
    
    return {
        'compiled_patterns': compiled_results,
        'substitutions': substitution_results,
        'named_groups': named_groups
    }

def regex_validation_patterns():
    # Common validation patterns
    validation_patterns = {
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'phone_us': r'^\(\d{3}\) \d{3}-\d{4}$',
        'ssn': r'^\d{3}-\d{2}-\d{4}$',
        'credit_card': r'^\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}$',
        'ipv4': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
        'url': r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$',
        'password_strong': r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$',
        'hex_color': r'^#?([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'
    }
    
    # Test data
    test_data = {
        'email': ['test@example.com', 'invalid.email', 'user@domain.co.uk'],
        'phone_us': ['(123) 456-7890', '123-456-7890', '1234567890'],
        'ssn': ['123-45-6789', '123456789', '12-345-6789'],
        'ipv4': ['192.168.1.1', '256.1.1.1', '192.168.1'],
        'password_strong': ['Password123!', 'weak', 'StrongP@ss1']
    }
    
    validation_results = {}
    for pattern_name, pattern in validation_patterns.items():
        if pattern_name in test_data:
            results = []
            for test_value in test_data[pattern_name]:
                is_valid = bool(re.match(pattern, test_value))
                results.append({'value': test_value, 'valid': is_valid})
            validation_results[pattern_name] = results
    
    return validation_results

# Unicode and encoding handling
def unicode_operations():
    # Unicode examples
    unicode_text = "Héllo Wörld! 你好世界 🌍🐍"
    
    unicode_info = {
        'length': len(unicode_text),
        'encode_utf8': unicode_text.encode('utf-8'),
        'encode_ascii_errors': 'Error handling needed for ASCII',
        'normalize_nfc': unicodedata.normalize('NFC', unicode_text),
        'normalize_nfd': unicodedata.normalize('NFD', unicode_text)
    }
    
    # Character categories
    char_categories = {}
    for char in "A1α你🐍":
        char_categories[char] = {
            'category': unicodedata.category(char),
            'name': unicodedata.name(char, 'UNKNOWN'),
            'is_alpha': char.isalpha(),
            'is_digit': char.isdigit(),
            'is_ascii': char.isascii()
        }
    
    # Encoding/decoding examples
    text = "Python Programming"
    encoding_examples = {
        'utf8': text.encode('utf-8').decode('utf-8'),
        'latin1': text.encode('latin-1').decode('latin-1'),
        'ascii': text.encode('ascii').decode('ascii')
    }
    
    return {
        'unicode_info': unicode_info,
        'character_categories': char_categories,
        'encoding_examples': encoding_examples
    }

# String parsing and processing
def string_parsing_patterns():
    # CSV-like parsing
    csv_line = 'John,"Doe, Jr.",30,"New York, NY",Engineer'
    
    def parse_csv_line(line):
        # Simple CSV parser handling quotes
        result = []
        current = ""
        in_quotes = False
        
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                result.append(current.strip())
                current = ""
            else:
                current += char
        
        result.append(current.strip())
        return result
    
    # Configuration file parsing
    config_text = """
    [database]
    host = localhost
    port = 5432
    username = admin
    password = secret123
    
    [logging]
    level = DEBUG
    file = app.log
    """
    
    def parse_config(text):
        config = {}
        current_section = None
        
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            elif line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                config[current_section] = {}
            elif '=' in line and current_section:
                key, value = line.split('=', 1)
                config[current_section][key.strip()] = value.strip()
        
        return config
    
    # Log parsing
    log_line = '2023-12-25 10:30:45 [INFO] User alice logged in from 192.168.1.100'
    log_pattern = r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)'
    
    log_match = re.match(log_pattern, log_line)
    log_parsed = {
        'date': log_match.group(1),
        'time': log_match.group(2),
        'level': log_match.group(3),
        'message': log_match.group(4)
    } if log_match else None
    
    return {
        'csv_parsing': parse_csv_line(csv_line),
        'config_parsing': parse_config(config_text),
        'log_parsing': log_parsed
    }

# Interview problems using strings
def string_interview_problems():
    def is_palindrome(s: str) -> bool:
        """Check if string is palindrome (ignore non-alphanumeric)"""
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]
    
    def longest_substring_without_repeating(s: str) -> int:
        """Find length of longest substring without repeating characters"""
        char_index = {}
        max_length = 0
        start = 0
        
        for i, char in enumerate(s):
            if char in char_index and char_index[char] >= start:
                start = char_index[char] + 1
            char_index[char] = i
            max_length = max(max_length, i - start + 1)
        
        return max_length
    
    def group_anagrams(strs: List[str]) -> List[List[str]]:
        """Group strings that are anagrams"""
        anagram_groups = {}
        for s in strs:
            key = ''.join(sorted(s))
            if key not in anagram_groups:
                anagram_groups[key] = []
            anagram_groups[key].append(s)
        return list(anagram_groups.values())
    
    def longest_common_prefix(strs: List[str]) -> str:
        """Find longest common prefix among strings"""
        if not strs:
            return ""
        
        min_len = min(len(s) for s in strs)
        for i in range(min_len):
            char = strs[0][i]
            if not all(s[i] == char for s in strs):
                return strs[0][:i]
        
        return strs[0][:min_len]
    
    def string_to_integer(s: str) -> int:
        """Convert string to integer (like atoi)"""
        s = s.strip()
        if not s:
            return 0
        
        sign = 1
        i = 0
        if s[0] in ['+', '-']:
            sign = -1 if s[0] == '-' else 1
            i = 1
        
        result = 0
        while i < len(s) and s[i].isdigit():
            result = result * 10 + int(s[i])
            i += 1
        
        result *= sign
        return max(-2**31, min(2**31 - 1, result))
    
    def word_pattern(pattern: str, s: str) -> bool:
        """Check if string follows given pattern"""
        words = s.split()
        if len(pattern) != len(words):
            return False
        
        char_to_word = {}
        word_to_char = {}
        
        for char, word in zip(pattern, words):
            if char in char_to_word:
                if char_to_word[char] != word:
                    return False
            else:
                char_to_word[char] = word
            
            if word in word_to_char:
                if word_to_char[word] != char:
                    return False
            else:
                word_to_char[word] = char
        
        return True
    
    # Test problems
    test_results = {
        'palindrome': [
            is_palindrome("A man a plan a canal Panama"),
            is_palindrome("hello world")
        ],
        'longest_substring': longest_substring_without_repeating("abcabcbb"),
        'anagrams': group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]),
        'common_prefix': longest_common_prefix(["flower", "flow", "flight"]),
        'string_to_int': [
            string_to_integer("42"),
            string_to_integer("   -42"),
            string_to_integer("4193 with words")
        ],
        'word_pattern': [
            word_pattern("abba", "dog cat cat dog"),
            word_pattern("abba", "dog cat cat fish")
        ]
    }
    
    return test_results

# Performance optimizations
def string_performance_optimizations():
    # String concatenation performance
    def time_operation(operation, iterations=1000):
        start = time.time()
        for _ in range(iterations):
            operation()
        return (time.time() - start) * 1000
    
    # Different concatenation methods
    words = ['word'] * 1000
    
    def concatenate_plus():
        result = ""
        for word in words:
            result += word
        return result
    
    def concatenate_join():
        return ''.join(words)
    
    def concatenate_format():
        return ''.join(f"{word}" for word in words)
    
    # Performance comparison
    performance_results = {
        'plus_operator': time_operation(concatenate_plus, 10),
        'join_method': time_operation(concatenate_join, 100),
        'format_method': time_operation(concatenate_format, 100)
    }
    
    # String search performance
    large_text = "word " * 10000
    
    def search_in():
        return "target" in large_text
    
    def search_find():
        return large_text.find("target") != -1
    
    def search_regex():
        return bool(re.search("target", large_text))
    
    search_performance = {
        'in_operator': time_operation(search_in, 1000),
        'find_method': time_operation(search_find, 1000),
        'regex_search': time_operation(search_regex, 100)
    }
    
    return {
        'concatenation_performance_ms': performance_results,
        'search_performance_ms': search_performance,
        'optimization_tips': [
            "Use str.join() for multiple concatenations",
            "Use 'in' operator for simple substring checks",
            "Compile regex patterns for repeated use",
            "Use f-strings for readable formatting"
        ]
    }

# Comprehensive testing
def run_all_string_demos():
    """Execute all string manipulation demonstrations"""
    demo_functions = [
        ('basic_methods', basic_string_methods),
        ('split_join', string_splitting_joining),
        ('replacement', string_replacement_translation),
        ('formatting', string_formatting_methods),
        ('advanced_formatting', advanced_formatting_patterns),
        ('regex_basics', regex_basics),
        ('regex_advanced', regex_advanced_patterns),
        ('regex_validation', regex_validation_patterns),
        ('unicode', unicode_operations),
        ('parsing', string_parsing_patterns),
        ('interview_problems', string_interview_problems),
        ('performance', string_performance_optimizations)
    ]
    
    results = {}
    for name, func in demo_functions:
        try:
            start_time = time.time()
            result = func()
            execution_time = time.time() - start_time
            results[name] = {
                'result': result,
                'execution_time': f"{execution_time*1000:.2f}ms"
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    print("=== Python String Manipulation Demo ===")
    
    # Run all demonstrations
    all_results = run_all_string_demos()
    
    for category, data in all_results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        
        if 'error' in data:
            print(f"  Error: {data['error']}")
            continue
            
        result = data['result']
        print(f"  Execution time: {data['execution_time']}")
        
        # Display results with appropriate truncation
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, (str, bytes)) and len(str(value)) > 100:
                    print(f"  {key}: {str(value)[:100]}... (truncated)")
                elif isinstance(value, list) and len(value) > 5:
                    print(f"  {key}: {value[:5]}... (showing first 5)")
                elif isinstance(value, dict) and len(value) > 3:
                    items = list(value.items())[:3]
                    print(f"  {key}: {dict(items)}... (showing first 3)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Result: {result}")
    
    print("\n=== STRING OPERATION SUMMARY ===")
    
    # Quick reference
    string_operations = {
        'Case': 'lower(), upper(), title(), capitalize(), swapcase()',
        'Search': 'find(), index(), count(), startswith(), endswith(), in',
        'Modify': 'replace(), strip(), split(), join()',
        'Format': 'f-strings, .format(), % formatting',
        'Regex': 're.search(), re.findall(), re.sub(), re.split()',
        'Validate': 'isdigit(), isalpha(), isalnum(), regex patterns'
    }
    
    for category, methods in string_operations.items():
        print(f"  {category}: {methods}")
    
    print("\n=== PERFORMANCE SUMMARY ===")
    total_time = sum(float(data.get('execution_time', '0ms')[:-2]) 
                    for data in all_results.values() 
                    if 'execution_time' in data)
    print(f"  Total execution time: {total_time:.2f}ms")
    print(f"  Functions executed: {len(all_results)}")
    print(f"  Average per function: {total_time/len(all_results):.2f}ms")
