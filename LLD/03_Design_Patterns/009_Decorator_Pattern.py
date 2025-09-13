"""
DECORATOR PATTERN - Structural Design Pattern
=============================================

Problem Statement:
Implement the Decorator pattern to add behavior to objects dynamically
without altering their structure:
- Add responsibilities to objects at runtime
- Alternative to subclassing for extending functionality
- Compose behaviors by wrapping objects
- Chain multiple decorators for complex behavior
- Maintain interface compatibility

Learning Objectives:
- Understand when to use Decorator pattern
- Implement flexible object enhancement
- Design decorator chains and combinations
- Handle dynamic behavior composition
- Avoid class explosion through decoration
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
import time
import json
from datetime import datetime
from functools import wraps
import logging


# ============================================================================
# COMPONENT INTERFACES
# ============================================================================

class Coffee(ABC):
    """Abstract component for coffee beverages."""
    
    @abstractmethod
    def get_description(self) -> str:
        """Get coffee description."""
        pass
    
    @abstractmethod
    def get_cost(self) -> float:
        """Get coffee cost."""
        pass
    
    @abstractmethod
    def get_ingredients(self) -> List[str]:
        """Get list of ingredients."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get complete coffee information."""
        return {
            'description': self.get_description(),
            'cost': self.get_cost(),
            'ingredients': self.get_ingredients()
        }


class DataSource(ABC):
    """Abstract component for data sources."""
    
    @abstractmethod
    def read_data(self) -> str:
        """Read data from source."""
        pass
    
    @abstractmethod
    def write_data(self, data: str) -> bool:
        """Write data to source."""
        pass
    
    @abstractmethod
    def get_source_info(self) -> Dict[str, Any]:
        """Get source information."""
        pass


class TextProcessor(ABC):
    """Abstract component for text processing."""
    
    @abstractmethod
    def process(self, text: str) -> str:
        """Process text."""
        pass
    
    @abstractmethod
    def get_processor_info(self) -> Dict[str, str]:
        """Get processor information."""
        pass


# ============================================================================
# CONCRETE COMPONENTS
# ============================================================================

class SimpleCoffee(Coffee):
    """Basic coffee implementation."""
    
    def __init__(self):
        self.base_cost = 2.00
        self.base_description = "Simple Coffee"
    
    def get_description(self) -> str:
        return self.base_description
    
    def get_cost(self) -> float:
        return self.base_cost
    
    def get_ingredients(self) -> List[str]:
        return ["Coffee beans", "Water"]


class Espresso(Coffee):
    """Espresso coffee implementation."""
    
    def __init__(self):
        self.base_cost = 1.99
        self.base_description = "Espresso"
    
    def get_description(self) -> str:
        return self.base_description
    
    def get_cost(self) -> float:
        return self.base_cost
    
    def get_ingredients(self) -> List[str]:
        return ["Espresso beans", "Hot water"]


class DarkRoast(Coffee):
    """Dark roast coffee implementation."""
    
    def __init__(self):
        self.base_cost = 2.50
        self.base_description = "Dark Roast Coffee"
    
    def get_description(self) -> str:
        return self.base_description
    
    def get_cost(self) -> float:
        return self.base_cost
    
    def get_ingredients(self) -> List[str]:
        return ["Dark roast beans", "Water"]


class FileDataSource(DataSource):
    """File-based data source."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.data_cache = ""
    
    def read_data(self) -> str:
        """Simulate reading from file."""
        print(f"FileDataSource: Reading from {self.filename}")
        # Simulate file content
        self.data_cache = f"File content from {self.filename}\nTimestamp: {datetime.now()}"
        return self.data_cache
    
    def write_data(self, data: str) -> bool:
        """Simulate writing to file."""
        print(f"FileDataSource: Writing to {self.filename}")
        self.data_cache = data
        return True
    
    def get_source_info(self) -> Dict[str, Any]:
        return {
            'type': 'file',
            'filename': self.filename,
            'cached_data_length': len(self.data_cache)
        }


class DatabaseDataSource(DataSource):
    """Database-based data source."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.last_query_result = ""
    
    def read_data(self) -> str:
        """Simulate reading from database."""
        print(f"DatabaseDataSource: Querying database")
        # Simulate database query result
        self.last_query_result = f"Database result from {self.connection_string}\nQuery executed at: {datetime.now()}"
        return self.last_query_result
    
    def write_data(self, data: str) -> bool:
        """Simulate writing to database."""
        print(f"DatabaseDataSource: Inserting data into database")
        return True
    
    def get_source_info(self) -> Dict[str, Any]:
        return {
            'type': 'database',
            'connection': self.connection_string,
            'last_result_length': len(self.last_query_result)
        }


class BasicTextProcessor(TextProcessor):
    """Basic text processor that returns text as-is."""
    
    def process(self, text: str) -> str:
        """Return text unchanged."""
        return text
    
    def get_processor_info(self) -> Dict[str, str]:
        return {
            'type': 'basic',
            'description': 'Basic text processor - no modifications'
        }


# ============================================================================
# ABSTRACT DECORATORS
# ============================================================================

class CoffeeDecorator(Coffee):
    """Abstract decorator for coffee."""
    
    def __init__(self, coffee: Coffee):
        self.coffee = coffee
    
    def get_description(self) -> str:
        return self.coffee.get_description()
    
    def get_cost(self) -> float:
        return self.coffee.get_cost()
    
    def get_ingredients(self) -> List[str]:
        return self.coffee.get_ingredients()


class DataSourceDecorator(DataSource):
    """Abstract decorator for data sources."""
    
    def __init__(self, data_source: DataSource):
        self.data_source = data_source
    
    def read_data(self) -> str:
        return self.data_source.read_data()
    
    def write_data(self, data: str) -> bool:
        return self.data_source.write_data(data)
    
    def get_source_info(self) -> Dict[str, Any]:
        return self.data_source.get_source_info()


class TextProcessorDecorator(TextProcessor):
    """Abstract decorator for text processors."""
    
    def __init__(self, processor: TextProcessor):
        self.processor = processor
    
    def process(self, text: str) -> str:
        return self.processor.process(text)
    
    def get_processor_info(self) -> Dict[str, str]:
        return self.processor.get_processor_info()


# ============================================================================
# CONCRETE DECORATORS - COFFEE SYSTEM
# ============================================================================

class MilkDecorator(CoffeeDecorator):
    """Decorator that adds milk to coffee."""
    
    def __init__(self, coffee: Coffee):
        super().__init__(coffee)
        self.milk_cost = 0.60
    
    def get_description(self) -> str:
        return f"{self.coffee.get_description()}, Milk"
    
    def get_cost(self) -> float:
        return self.coffee.get_cost() + self.milk_cost
    
    def get_ingredients(self) -> List[str]:
        ingredients = self.coffee.get_ingredients().copy()
        ingredients.append("Milk")
        return ingredients


class SugarDecorator(CoffeeDecorator):
    """Decorator that adds sugar to coffee."""
    
    def __init__(self, coffee: Coffee, sugar_packets: int = 1):
        super().__init__(coffee)
        self.sugar_packets = sugar_packets
        self.sugar_cost_per_packet = 0.10
    
    def get_description(self) -> str:
        sugar_desc = f"{self.sugar_packets} Sugar" if self.sugar_packets == 1 else f"{self.sugar_packets} Sugars"
        return f"{self.coffee.get_description()}, {sugar_desc}"
    
    def get_cost(self) -> float:
        return self.coffee.get_cost() + (self.sugar_packets * self.sugar_cost_per_packet)
    
    def get_ingredients(self) -> List[str]:
        ingredients = self.coffee.get_ingredients().copy()
        ingredients.append(f"Sugar ({self.sugar_packets} packets)")
        return ingredients


class WhipDecorator(CoffeeDecorator):
    """Decorator that adds whipped cream to coffee."""
    
    def __init__(self, coffee: Coffee):
        super().__init__(coffee)
        self.whip_cost = 0.70
    
    def get_description(self) -> str:
        return f"{self.coffee.get_description()}, Whip"
    
    def get_cost(self) -> float:
        return self.coffee.get_cost() + self.whip_cost
    
    def get_ingredients(self) -> List[str]:
        ingredients = self.coffee.get_ingredients().copy()
        ingredients.append("Whipped cream")
        return ingredients


class VanillaDecorator(CoffeeDecorator):
    """Decorator that adds vanilla syrup to coffee."""
    
    def __init__(self, coffee: Coffee):
        super().__init__(coffee)
        self.vanilla_cost = 0.50
    
    def get_description(self) -> str:
        return f"{self.coffee.get_description()}, Vanilla"
    
    def get_cost(self) -> float:
        return self.coffee.get_cost() + self.vanilla_cost
    
    def get_ingredients(self) -> List[str]:
        ingredients = self.coffee.get_ingredients().copy()
        ingredients.append("Vanilla syrup")
        return ingredients


class SizeDecorator(CoffeeDecorator):
    """Decorator that changes coffee size."""
    
    def __init__(self, coffee: Coffee, size: str):
        super().__init__(coffee)
        self.size = size
        self.size_multipliers = {
            'small': 0.8,
            'medium': 1.0,
            'large': 1.3,
            'extra_large': 1.6
        }
    
    def get_description(self) -> str:
        return f"{self.size.title()} {self.coffee.get_description()}"
    
    def get_cost(self) -> float:
        multiplier = self.size_multipliers.get(self.size, 1.0)
        return self.coffee.get_cost() * multiplier
    
    def get_ingredients(self) -> List[str]:
        return self.coffee.get_ingredients()


# ============================================================================
# CONCRETE DECORATORS - DATA SOURCE SYSTEM
# ============================================================================

class EncryptionDecorator(DataSourceDecorator):
    """Decorator that adds encryption to data source."""
    
    def __init__(self, data_source: DataSource, encryption_key: str = "default_key"):
        super().__init__(data_source)
        self.encryption_key = encryption_key
    
    def read_data(self) -> str:
        """Read and decrypt data."""
        encrypted_data = self.data_source.read_data()
        decrypted_data = self._decrypt(encrypted_data)
        print("EncryptionDecorator: Data decrypted")
        return decrypted_data
    
    def write_data(self, data: str) -> bool:
        """Encrypt and write data."""
        encrypted_data = self._encrypt(data)
        print("EncryptionDecorator: Data encrypted")
        return self.data_source.write_data(encrypted_data)
    
    def get_source_info(self) -> Dict[str, Any]:
        info = self.data_source.get_source_info()
        info['encryption'] = {
            'enabled': True,
            'algorithm': 'AES-256',
            'key_length': len(self.encryption_key)
        }
        return info
    
    def _encrypt(self, data: str) -> str:
        """Simulate encryption."""
        return f"ENCRYPTED[{data}]"
    
    def _decrypt(self, encrypted_data: str) -> str:
        """Simulate decryption."""
        if encrypted_data.startswith("ENCRYPTED[") and encrypted_data.endswith("]"):
            return encrypted_data[10:-1]  # Remove ENCRYPTED[ and ]
        return encrypted_data


class CompressionDecorator(DataSourceDecorator):
    """Decorator that adds compression to data source."""
    
    def __init__(self, data_source: DataSource, compression_level: int = 5):
        super().__init__(data_source)
        self.compression_level = compression_level
        self.compression_ratio = 0.7  # Simulate 30% compression
    
    def read_data(self) -> str:
        """Read and decompress data."""
        compressed_data = self.data_source.read_data()
        decompressed_data = self._decompress(compressed_data)
        print(f"CompressionDecorator: Data decompressed (level {self.compression_level})")
        return decompressed_data
    
    def write_data(self, data: str) -> bool:
        """Compress and write data."""
        compressed_data = self._compress(data)
        print(f"CompressionDecorator: Data compressed (level {self.compression_level})")
        return self.data_source.write_data(compressed_data)
    
    def get_source_info(self) -> Dict[str, Any]:
        info = self.data_source.get_source_info()
        info['compression'] = {
            'enabled': True,
            'level': self.compression_level,
            'ratio': self.compression_ratio
        }
        return info
    
    def _compress(self, data: str) -> str:
        """Simulate compression."""
        return f"COMPRESSED[{data}]"
    
    def _decompress(self, compressed_data: str) -> str:
        """Simulate decompression."""
        if compressed_data.startswith("COMPRESSED[") and compressed_data.endswith("]"):
            return compressed_data[11:-1]  # Remove COMPRESSED[ and ]
        return compressed_data


class CachingDecorator(DataSourceDecorator):
    """Decorator that adds caching to data source."""
    
    def __init__(self, data_source: DataSource, cache_ttl: int = 300):
        super().__init__(data_source)
        self.cache_ttl = cache_ttl  # Time to live in seconds
        self.cache = {}
        self.cache_timestamps = {}
    
    def read_data(self) -> str:
        """Read data with caching."""
        cache_key = "read_data"
        current_time = time.time()
        
        # Check if cached data is still valid
        if (cache_key in self.cache and 
            cache_key in self.cache_timestamps and
            current_time - self.cache_timestamps[cache_key] < self.cache_ttl):
            print("CachingDecorator: Returning cached data")
            return self.cache[cache_key]
        
        # Cache miss or expired - fetch fresh data
        data = self.data_source.read_data()
        self.cache[cache_key] = data
        self.cache_timestamps[cache_key] = current_time
        print("CachingDecorator: Data cached")
        return data
    
    def write_data(self, data: str) -> bool:
        """Write data and invalidate cache."""
        result = self.data_source.write_data(data)
        if result:
            # Invalidate cache
            self.cache.clear()
            self.cache_timestamps.clear()
            print("CachingDecorator: Cache invalidated after write")
        return result
    
    def get_source_info(self) -> Dict[str, Any]:
        info = self.data_source.get_source_info()
        info['caching'] = {
            'enabled': True,
            'ttl_seconds': self.cache_ttl,
            'cached_items': len(self.cache),
            'cache_hit_ratio': self._calculate_hit_ratio()
        }
        return info
    
    def _calculate_hit_ratio(self) -> float:
        """Calculate cache hit ratio (simplified)."""
        return 0.75  # Simulated hit ratio


class LoggingDecorator(DataSourceDecorator):
    """Decorator that adds logging to data source operations."""
    
    def __init__(self, data_source: DataSource, log_level: str = "INFO"):
        super().__init__(data_source)
        self.log_level = log_level
        self.operation_count = 0
        self.operation_log = []
    
    def read_data(self) -> str:
        """Read data with logging."""
        start_time = time.time()
        self.operation_count += 1
        
        try:
            data = self.data_source.read_data()
            duration = time.time() - start_time
            
            log_entry = {
                'operation': 'read',
                'timestamp': datetime.now().isoformat(),
                'duration_ms': round(duration * 1000, 2),
                'data_length': len(data),
                'success': True
            }
            self.operation_log.append(log_entry)
            
            print(f"LoggingDecorator: READ operation completed in {duration*1000:.2f}ms")
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            log_entry = {
                'operation': 'read',
                'timestamp': datetime.now().isoformat(),
                'duration_ms': round(duration * 1000, 2),
                'error': str(e),
                'success': False
            }
            self.operation_log.append(log_entry)
            print(f"LoggingDecorator: READ operation failed: {e}")
            raise
    
    def write_data(self, data: str) -> bool:
        """Write data with logging."""
        start_time = time.time()
        self.operation_count += 1
        
        try:
            result = self.data_source.write_data(data)
            duration = time.time() - start_time
            
            log_entry = {
                'operation': 'write',
                'timestamp': datetime.now().isoformat(),
                'duration_ms': round(duration * 1000, 2),
                'data_length': len(data),
                'success': result
            }
            self.operation_log.append(log_entry)
            
            print(f"LoggingDecorator: WRITE operation completed in {duration*1000:.2f}ms")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            log_entry = {
                'operation': 'write',
                'timestamp': datetime.now().isoformat(),
                'duration_ms': round(duration * 1000, 2),
                'error': str(e),
                'success': False
            }
            self.operation_log.append(log_entry)
            print(f"LoggingDecorator: WRITE operation failed: {e}")
            raise
    
    def get_source_info(self) -> Dict[str, Any]:
        info = self.data_source.get_source_info()
        info['logging'] = {
            'enabled': True,
            'level': self.log_level,
            'total_operations': self.operation_count,
            'recent_operations': len(self.operation_log)
        }
        return info
    
    def get_operation_log(self) -> List[Dict[str, Any]]:
        """Get operation log."""
        return self.operation_log.copy()


# ============================================================================
# CONCRETE DECORATORS - TEXT PROCESSING SYSTEM
# ============================================================================

class UpperCaseDecorator(TextProcessorDecorator):
    """Decorator that converts text to uppercase."""
    
    def process(self, text: str) -> str:
        processed = self.processor.process(text)
        return processed.upper()
    
    def get_processor_info(self) -> Dict[str, str]:
        info = self.processor.get_processor_info()
        info['uppercase'] = 'enabled'
        return info


class TrimDecorator(TextProcessorDecorator):
    """Decorator that trims whitespace from text."""
    
    def process(self, text: str) -> str:
        processed = self.processor.process(text)
        return processed.strip()
    
    def get_processor_info(self) -> Dict[str, str]:
        info = self.processor.get_processor_info()
        info['trim'] = 'enabled'
        return info


class ReplaceDecorator(TextProcessorDecorator):
    """Decorator that replaces text patterns."""
    
    def __init__(self, processor: TextProcessor, old_text: str, new_text: str):
        super().__init__(processor)
        self.old_text = old_text
        self.new_text = new_text
    
    def process(self, text: str) -> str:
        processed = self.processor.process(text)
        return processed.replace(self.old_text, self.new_text)
    
    def get_processor_info(self) -> Dict[str, str]:
        info = self.processor.get_processor_info()
        info['replace'] = f"'{self.old_text}' -> '{self.new_text}'"
        return info


class WordCountDecorator(TextProcessorDecorator):
    """Decorator that adds word count information."""
    
    def __init__(self, processor: TextProcessor):
        super().__init__(processor)
        self.last_word_count = 0
    
    def process(self, text: str) -> str:
        processed = self.processor.process(text)
        self.last_word_count = len(processed.split())
        return f"{processed}\n[Word count: {self.last_word_count}]"
    
    def get_processor_info(self) -> Dict[str, str]:
        info = self.processor.get_processor_info()
        info['word_count'] = f'enabled (last count: {self.last_word_count})'
        return info


# ============================================================================
# FUNCTION DECORATORS (PYTHON-SPECIFIC)
# ============================================================================

def timing_decorator(func: Callable) -> Callable:
    """Function decorator that measures execution time."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Function {func.__name__} executed in {duration*1000:.2f}ms")
        return result
    
    return wrapper


def retry_decorator(max_attempts: int = 3, delay: float = 1.0):
    """Function decorator that retries failed operations."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"All {max_attempts} attempts failed.")
            
            raise last_exception
        
        return wrapper
    return decorator


def cache_decorator(ttl: int = 300):
    """Function decorator that caches results."""
    
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_timestamps = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            cache_key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check if cached result is still valid
            if (cache_key in cache and 
                cache_key in cache_timestamps and
                current_time - cache_timestamps[cache_key] < ttl):
                print(f"Cache hit for {func.__name__}")
                return cache[cache_key]
            
            # Cache miss - execute function
            result = func(*args, **kwargs)
            cache[cache_key] = result
            cache_timestamps[cache_key] = current_time
            print(f"Result cached for {func.__name__}")
            return result
        
        return wrapper
    return decorator


# ============================================================================
# DECORATOR FACTORY AND MANAGER
# ============================================================================

class DecoratorFactory:
    """Factory for creating and managing decorators."""
    
    def __init__(self):
        self.coffee_decorators = {
            'milk': MilkDecorator,
            'sugar': SugarDecorator,
            'whip': WhipDecorator,
            'vanilla': VanillaDecorator,
            'size': SizeDecorator
        }
        
        self.data_source_decorators = {
            'encryption': EncryptionDecorator,
            'compression': CompressionDecorator,
            'caching': CachingDecorator,
            'logging': LoggingDecorator
        }
        
        self.text_processor_decorators = {
            'uppercase': UpperCaseDecorator,
            'trim': TrimDecorator,
            'replace': ReplaceDecorator,
            'word_count': WordCountDecorator
        }
    
    def create_decorated_coffee(self, base_coffee: Coffee, decorations: List[Dict[str, Any]]) -> Coffee:
        """Create decorated coffee with multiple decorators."""
        decorated_coffee = base_coffee
        
        for decoration in decorations:
            decorator_name = decoration.get('type')
            decorator_args = decoration.get('args', {})
            
            if decorator_name in self.coffee_decorators:
                decorator_class = self.coffee_decorators[decorator_name]
                
                if decorator_name == 'sugar':
                    decorated_coffee = decorator_class(decorated_coffee, 
                                                     decorator_args.get('packets', 1))
                elif decorator_name == 'size':
                    decorated_coffee = decorator_class(decorated_coffee, 
                                                     decorator_args.get('size', 'medium'))
                else:
                    decorated_coffee = decorator_class(decorated_coffee)
        
        return decorated_coffee
    
    def create_decorated_data_source(self, base_source: DataSource, 
                                   decorations: List[Dict[str, Any]]) -> DataSource:
        """Create decorated data source with multiple decorators."""
        decorated_source = base_source
        
        for decoration in decorations:
            decorator_name = decoration.get('type')
            decorator_args = decoration.get('args', {})
            
            if decorator_name in self.data_source_decorators:
                decorator_class = self.data_source_decorators[decorator_name]
                
                if decorator_name == 'encryption':
                    decorated_source = decorator_class(decorated_source,
                                                     decorator_args.get('key', 'default_key'))
                elif decorator_name == 'compression':
                    decorated_source = decorator_class(decorated_source,
                                                     decorator_args.get('level', 5))
                elif decorator_name == 'caching':
                    decorated_source = decorator_class(decorated_source,
                                                     decorator_args.get('ttl', 300))
                elif decorator_name == 'logging':
                    decorated_source = decorator_class(decorated_source,
                                                     decorator_args.get('level', 'INFO'))
                else:
                    decorated_source = decorator_class(decorated_source)
        
        return decorated_source
    
    def get_available_decorators(self) -> Dict[str, List[str]]:
        """Get list of available decorators by category."""
        return {
            'coffee': list(self.coffee_decorators.keys()),
            'data_source': list(self.data_source_decorators.keys()),
            'text_processor': list(self.text_processor_decorators.keys())
        }


def demonstrate_decorator_pattern():
    """
    Demonstrate Decorator pattern implementations.
    """
    print("=== DECORATOR PATTERN DEMONSTRATION ===\n")
    
    # 1. Basic Coffee Decorators
    print("1. BASIC COFFEE DECORATORS:")
    
    # Start with simple coffee
    coffee = SimpleCoffee()
    print(f"   Base coffee: {coffee.get_description()}")
    print(f"   Cost: ${coffee.get_cost():.2f}")
    print(f"   Ingredients: {coffee.get_ingredients()}")
    
    # Add milk
    coffee_with_milk = MilkDecorator(coffee)
    print(f"\n   With milk: {coffee_with_milk.get_description()}")
    print(f"   Cost: ${coffee_with_milk.get_cost():.2f}")
    print(f"   Ingredients: {coffee_with_milk.get_ingredients()}")
    
    # Add sugar
    coffee_with_milk_and_sugar = SugarDecorator(coffee_with_milk, 2)
    print(f"\n   With milk and sugar: {coffee_with_milk_and_sugar.get_description()}")
    print(f"   Cost: ${coffee_with_milk_and_sugar.get_cost():.2f}")
    print(f"   Ingredients: {coffee_with_milk_and_sugar.get_ingredients()}")
    
    # Add whip
    deluxe_coffee = WhipDecorator(coffee_with_milk_and_sugar)
    print(f"\n   Deluxe coffee: {deluxe_coffee.get_description()}")
    print(f"   Cost: ${deluxe_coffee.get_cost():.2f}")
    print(f"   Ingredients: {deluxe_coffee.get_ingredients()}")
    print()
    
    # 2. Different Base Coffees with Same Decorators
    print("2. DIFFERENT BASE COFFEES WITH SAME DECORATORS:")
    
    base_coffees = [SimpleCoffee(), Espresso(), DarkRoast()]
    
    for base_coffee in base_coffees:
        # Apply same decorations to different bases
        decorated = VanillaDecorator(MilkDecorator(base_coffee))
        decorated = SizeDecorator(decorated, "large")
        
        print(f"   {decorated.get_description()}")
        print(f"     Cost: ${decorated.get_cost():.2f}")
        print(f"     Ingredients: {len(decorated.get_ingredients())} items")
    
    print()
    
    # 3. Data Source Decorators
    print("3. DATA SOURCE DECORATORS:")
    
    # Create base data source
    file_source = FileDataSource("data.txt")
    print("   Base file source:")
    data = file_source.read_data()
    print(f"     Data: {data[:50]}...")
    print(f"     Info: {file_source.get_source_info()}")
    
    # Add encryption
    encrypted_source = EncryptionDecorator(file_source, "secret_key_123")
    print("\n   With encryption:")
    encrypted_source.write_data("Sensitive information")
    encrypted_data = encrypted_source.read_data()
    print(f"     Encrypted data: {encrypted_data[:50]}...")
    print(f"     Info: {encrypted_source.get_source_info()}")
    
    # Add compression
    compressed_encrypted_source = CompressionDecorator(encrypted_source, 9)
    print("\n   With compression and encryption:")
    compressed_encrypted_source.write_data("Large amount of data to compress")
    compressed_data = compressed_encrypted_source.read_data()
    print(f"     Processed data: {compressed_data[:50]}...")
    print(f"     Info: {compressed_encrypted_source.get_source_info()}")
    
    print()
    
    # 4. Complex Decorator Chain
    print("4. COMPLEX DECORATOR CHAIN:")
    
    # Create database source
    db_source = DatabaseDataSource("postgresql://localhost:5432/mydb")
    
    # Apply multiple decorators in chain
    fully_decorated_source = LoggingDecorator(
        CachingDecorator(
            CompressionDecorator(
                EncryptionDecorator(db_source, "ultra_secure_key"),
                compression_level=7
            ),
            cache_ttl=600
        ),
        log_level="DEBUG"
    )
    
    print("   Fully decorated data source chain:")
    print("     Database -> Encryption -> Compression -> Caching -> Logging")
    
    # Test the decorated source
    print("\n   Testing decorated source:")
    test_data = "Important business data that needs security and performance"
    
    print("     Writing data...")
    fully_decorated_source.write_data(test_data)
    
    print("     Reading data (first time)...")
    read_data1 = fully_decorated_source.read_data()
    
    print("     Reading data (second time - should hit cache)...")
    read_data2 = fully_decorated_source.read_data()
    
    print(f"     Final info: {fully_decorated_source.get_source_info()}")
    print()
    
    # 5. Text Processing Decorators
    print("5. TEXT PROCESSING DECORATORS:")
    
    # Create base processor
    processor = BasicTextProcessor()
    
    # Create decorated processor chain
    decorated_processor = WordCountDecorator(
        ReplaceDecorator(
            TrimDecorator(
                UpperCaseDecorator(processor)
            ),
            "HELLO", "GREETINGS"
        )
    )
    
    test_text = "  hello world, this is a test message  "
    print(f"   Original text: '{test_text}'")
    
    processed_text = decorated_processor.process(test_text)
    print(f"   Processed text: '{processed_text}'")
    print(f"   Processor info: {decorated_processor.get_processor_info()}")
    print()
    
    # 6. Decorator Factory
    print("6. DECORATOR FACTORY:")
    
    factory = DecoratorFactory()
    
    # Show available decorators
    available = factory.get_available_decorators()
    print("   Available decorators:")
    for category, decorators in available.items():
        print(f"     {category}: {decorators}")
    
    # Create complex coffee using factory
    coffee_config = [
        {'type': 'size', 'args': {'size': 'large'}},
        {'type': 'milk'},
        {'type': 'vanilla'},
        {'type': 'sugar', 'args': {'packets': 2}},
        {'type': 'whip'}
    ]
    
    factory_coffee = factory.create_decorated_coffee(Espresso(), coffee_config)
    print(f"\n   Factory-created coffee: {factory_coffee.get_description()}")
    print(f"   Cost: ${factory_coffee.get_cost():.2f}")
    
    # Create complex data source using factory
    data_source_config = [
        {'type': 'encryption', 'args': {'key': 'factory_key'}},
        {'type': 'compression', 'args': {'level': 8}},
        {'type': 'caching', 'args': {'ttl': 900}},
        {'type': 'logging', 'args': {'level': 'INFO'}}
    ]
    
    factory_data_source = factory.create_decorated_data_source(
        FileDataSource("factory_data.txt"), 
        data_source_config
    )
    
    print(f"\n   Factory-created data source info:")
    source_info = factory_data_source.get_source_info()
    for key, value in source_info.items():
        print(f"     {key}: {value}")
    
    print()
    
    # 7. Function Decorators
    print("7. FUNCTION DECORATORS:")
    
    @timing_decorator
    @cache_decorator(ttl=60)
    @retry_decorator(max_attempts=3, delay=0.1)
    def expensive_calculation(n: int) -> int:
        """Simulate expensive calculation."""
        if n < 0:
            raise ValueError("Negative numbers not supported")
        
        # Simulate work
        time.sleep(0.01)
        return n * n
    
    print("   Testing decorated function:")
    
    # First call
    result1 = expensive_calculation(5)
    print(f"     First call result: {result1}")
    
    # Second call (should hit cache)
    result2 = expensive_calculation(5)
    print(f"     Second call result: {result2}")
    
    # Different parameter
    result3 = expensive_calculation(7)
    print(f"     Different parameter result: {result3}")
    
    print()
    
    # 8. Decorator Pattern Benefits
    print("8. DECORATOR PATTERN BENEFITS:")
    print("   ✓ Runtime Enhancement: Add behavior to objects at runtime")
    print("   ✓ Flexible Composition: Combine multiple decorators in any order")
    print("   ✓ Single Responsibility: Each decorator has one specific purpose")
    print("   ✓ Open/Closed Principle: Extend functionality without modifying existing code")
    print("   ✓ Alternative to Inheritance: Avoid class explosion from subclassing")
    print("   ✓ Transparent Interface: Decorated objects maintain the same interface")
    print("   ✓ Dynamic Configuration: Decorators can be applied based on runtime conditions")
    print("   ✓ Reusable Components: Decorators can be reused across different objects")
    print()
    
    print("=== DECORATOR PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_decorator_pattern()
