"""
STRATEGY PATTERN - Behavioral Design Pattern
============================================

Problem Statement:
Implement the Strategy pattern to define a family of algorithms, encapsulate
each one, and make them interchangeable at runtime:
- Algorithm family with common interface
- Runtime algorithm selection and switching
- Payment processing with multiple methods
- Sorting algorithms with different strategies
- Compression strategies for different file types

Learning Objectives:
- Understand Strategy vs State pattern differences
- Implement algorithm families with common interfaces
- Design runtime algorithm selection mechanisms
- Handle context-strategy collaboration
- Create pluggable algorithm architectures
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union
import time
import random
import hashlib
from datetime import datetime, timedelta
from enum import Enum
import json


# ============================================================================
# STRATEGY INTERFACE
# ============================================================================

class Strategy(ABC):
    """Abstract strategy interface."""
    
    @abstractmethod
    def execute(self, context: 'Context', *args, **kwargs) -> Any:
        """Execute the strategy algorithm."""
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about this strategy."""
        pass


class Context:
    """Context class that uses strategies."""
    
    def __init__(self, strategy: Strategy = None):
        self._strategy = strategy
        self.execution_history: List[Dict[str, Any]] = []
    
    def set_strategy(self, strategy: Strategy) -> None:
        """Set the current strategy."""
        self._strategy = strategy
        print(f"Strategy changed to: {strategy.__class__.__name__}")
    
    def execute_strategy(self, *args, **kwargs) -> Any:
        """Execute the current strategy."""
        if not self._strategy:
            raise ValueError("No strategy set")
        
        start_time = time.time()
        result = self._strategy.execute(self, *args, **kwargs)
        execution_time = time.time() - start_time
        
        # Record execution
        execution_record = {
            'strategy': self._strategy.__class__.__name__,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat(),
            'args_count': len(args),
            'kwargs_count': len(kwargs)
        }
        self.execution_history.append(execution_record)
        
        return result
    
    def get_current_strategy(self) -> Optional[Strategy]:
        """Get current strategy."""
        return self._strategy
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {'total_executions': 0}
        
        strategy_counts = {}
        total_time = 0
        
        for record in self.execution_history:
            strategy_name = record['strategy']
            strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
            total_time += record['execution_time']
        
        return {
            'total_executions': len(self.execution_history),
            'total_time': total_time,
            'average_time': total_time / len(self.execution_history),
            'strategy_usage': strategy_counts,
            'most_used_strategy': max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else None
        }


# ============================================================================
# PAYMENT PROCESSING STRATEGIES
# ============================================================================

class PaymentStrategy(Strategy):
    """Abstract payment strategy."""
    
    @abstractmethod
    def process_payment(self, amount: float, currency: str, payment_details: Dict[str, Any]) -> Dict[str, Any]:
        """Process payment using this strategy."""
        pass
    
    def execute(self, context: Context, amount: float, currency: str, payment_details: Dict[str, Any]) -> Dict[str, Any]:
        """Execute payment processing."""
        return self.process_payment(amount, currency, payment_details)


class CreditCardStrategy(PaymentStrategy):
    """Credit card payment strategy."""
    
    def __init__(self):
        self.transaction_fee = 0.029  # 2.9% transaction fee
        self.processing_time = 2.0  # 2 seconds processing time
    
    def process_payment(self, amount: float, currency: str, payment_details: Dict[str, Any]) -> Dict[str, Any]:
        """Process credit card payment."""
        print(f"Processing credit card payment: {currency} {amount:.2f}")
        
        # Validate credit card details
        card_number = payment_details.get('card_number', '')
        expiry_date = payment_details.get('expiry_date', '')
        cvv = payment_details.get('cvv', '')
        
        if len(card_number) != 16 or len(cvv) != 3:
            return {
                'success': False,
                'error': 'Invalid card details',
                'transaction_id': None
            }
        
        # Simulate processing time
        time.sleep(0.1)  # Reduced for demo
        
        # Calculate fees
        fee = amount * self.transaction_fee
        net_amount = amount - fee
        
        # Generate transaction ID
        transaction_id = f"CC_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return {
            'success': True,
            'transaction_id': transaction_id,
            'amount': amount,
            'fee': fee,
            'net_amount': net_amount,
            'currency': currency,
            'payment_method': 'Credit Card',
            'card_last_four': card_number[-4:],
            'processed_at': datetime.now().isoformat()
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get credit card strategy information."""
        return {
            'name': 'Credit Card Payment',
            'transaction_fee_percent': self.transaction_fee * 100,
            'processing_time_seconds': self.processing_time,
            'supported_currencies': ['USD', 'EUR', 'GBP'],
            'security_features': ['CVV verification', '3D Secure', 'Fraud detection']
        }


class PayPalStrategy(PaymentStrategy):
    """PayPal payment strategy."""
    
    def __init__(self):
        self.transaction_fee = 0.034  # 3.4% transaction fee
        self.fixed_fee = 0.30  # $0.30 fixed fee
        self.processing_time = 1.5  # 1.5 seconds processing time
    
    def process_payment(self, amount: float, currency: str, payment_details: Dict[str, Any]) -> Dict[str, Any]:
        """Process PayPal payment."""
        print(f"Processing PayPal payment: {currency} {amount:.2f}")
        
        # Validate PayPal details
        email = payment_details.get('email', '')
        password = payment_details.get('password', '')
        
        if '@' not in email or len(password) < 6:
            return {
                'success': False,
                'error': 'Invalid PayPal credentials',
                'transaction_id': None
            }
        
        # Simulate processing time
        time.sleep(0.08)  # Reduced for demo
        
        # Calculate fees
        fee = (amount * self.transaction_fee) + self.fixed_fee
        net_amount = amount - fee
        
        # Generate transaction ID
        transaction_id = f"PP_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return {
            'success': True,
            'transaction_id': transaction_id,
            'amount': amount,
            'fee': fee,
            'net_amount': net_amount,
            'currency': currency,
            'payment_method': 'PayPal',
            'email': email,
            'processed_at': datetime.now().isoformat()
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get PayPal strategy information."""
        return {
            'name': 'PayPal Payment',
            'transaction_fee_percent': self.transaction_fee * 100,
            'fixed_fee': self.fixed_fee,
            'processing_time_seconds': self.processing_time,
            'supported_currencies': ['USD', 'EUR', 'GBP', 'CAD', 'AUD'],
            'security_features': ['Two-factor authentication', 'Buyer protection', 'Encrypted transactions']
        }


class BankTransferStrategy(PaymentStrategy):
    """Bank transfer payment strategy."""
    
    def __init__(self):
        self.transaction_fee = 0.01  # 1% transaction fee
        self.processing_time = 24.0  # 24 hours processing time
        self.minimum_amount = 10.0
    
    def process_payment(self, amount: float, currency: str, payment_details: Dict[str, Any]) -> Dict[str, Any]:
        """Process bank transfer payment."""
        print(f"Processing bank transfer: {currency} {amount:.2f}")
        
        # Validate bank details
        account_number = payment_details.get('account_number', '')
        routing_number = payment_details.get('routing_number', '')
        
        if len(account_number) < 8 or len(routing_number) != 9:
            return {
                'success': False,
                'error': 'Invalid bank account details',
                'transaction_id': None
            }
        
        # Check minimum amount
        if amount < self.minimum_amount:
            return {
                'success': False,
                'error': f'Minimum transfer amount is {currency} {self.minimum_amount}',
                'transaction_id': None
            }
        
        # Simulate processing time (reduced for demo)
        time.sleep(0.05)
        
        # Calculate fees
        fee = amount * self.transaction_fee
        net_amount = amount - fee
        
        # Generate transaction ID
        transaction_id = f"BT_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return {
            'success': True,
            'transaction_id': transaction_id,
            'amount': amount,
            'fee': fee,
            'net_amount': net_amount,
            'currency': currency,
            'payment_method': 'Bank Transfer',
            'account_last_four': account_number[-4:],
            'estimated_completion': (datetime.now() + timedelta(hours=24)).isoformat(),
            'processed_at': datetime.now().isoformat()
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get bank transfer strategy information."""
        return {
            'name': 'Bank Transfer',
            'transaction_fee_percent': self.transaction_fee * 100,
            'processing_time_hours': self.processing_time,
            'minimum_amount': self.minimum_amount,
            'supported_currencies': ['USD', 'EUR'],
            'security_features': ['Bank-grade encryption', 'ACH verification', 'Fraud monitoring']
        }


class CryptocurrencyStrategy(PaymentStrategy):
    """Cryptocurrency payment strategy."""
    
    def __init__(self, crypto_type: str = 'Bitcoin'):
        self.crypto_type = crypto_type
        self.network_fee = 0.0001  # Fixed network fee in crypto
        self.processing_time = 0.5  # 30 minutes average confirmation
        self.exchange_rates = {
            'Bitcoin': 45000.0,  # 1 BTC = $45,000
            'Ethereum': 3000.0,   # 1 ETH = $3,000
            'Litecoin': 150.0     # 1 LTC = $150
        }
    
    def process_payment(self, amount: float, currency: str, payment_details: Dict[str, Any]) -> Dict[str, Any]:
        """Process cryptocurrency payment."""
        print(f"Processing {self.crypto_type} payment: {currency} {amount:.2f}")
        
        # Validate crypto details
        wallet_address = payment_details.get('wallet_address', '')
        private_key_hash = payment_details.get('private_key_hash', '')
        
        if len(wallet_address) < 26 or not private_key_hash:
            return {
                'success': False,
                'error': 'Invalid cryptocurrency wallet details',
                'transaction_id': None
            }
        
        # Convert to cryptocurrency amount
        crypto_rate = self.exchange_rates.get(self.crypto_type, 1.0)
        crypto_amount = amount / crypto_rate
        
        # Simulate processing time
        time.sleep(0.03)  # Reduced for demo
        
        # Calculate network fee in USD
        network_fee_usd = self.network_fee * crypto_rate
        net_amount = amount - network_fee_usd
        
        # Generate transaction ID (simulate blockchain hash)
        transaction_data = f"{wallet_address}{amount}{time.time()}"
        transaction_id = hashlib.sha256(transaction_data.encode()).hexdigest()[:16]
        
        return {
            'success': True,
            'transaction_id': transaction_id,
            'amount': amount,
            'crypto_amount': crypto_amount,
            'crypto_type': self.crypto_type,
            'network_fee': network_fee_usd,
            'net_amount': net_amount,
            'currency': currency,
            'payment_method': f'{self.crypto_type} Payment',
            'wallet_address': wallet_address[:8] + '...' + wallet_address[-8:],
            'exchange_rate': crypto_rate,
            'processed_at': datetime.now().isoformat()
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get cryptocurrency strategy information."""
        return {
            'name': f'{self.crypto_type} Payment',
            'crypto_type': self.crypto_type,
            'network_fee': self.network_fee,
            'processing_time_minutes': self.processing_time * 60,
            'exchange_rate': self.exchange_rates.get(self.crypto_type, 1.0),
            'security_features': ['Blockchain verification', 'Cryptographic signatures', 'Decentralized network']
        }


# ============================================================================
# SORTING STRATEGIES
# ============================================================================

class SortingStrategy(Strategy):
    """Abstract sorting strategy."""
    
    @abstractmethod
    def sort(self, data: List[Any], key_func: Callable = None, reverse: bool = False) -> List[Any]:
        """Sort data using this strategy."""
        pass
    
    def execute(self, context: Context, data: List[Any], key_func: Callable = None, reverse: bool = False) -> List[Any]:
        """Execute sorting strategy."""
        return self.sort(data.copy(), key_func, reverse)  # Work on copy to avoid modifying original


class BubbleSortStrategy(SortingStrategy):
    """Bubble sort strategy - O(n²) time complexity."""
    
    def sort(self, data: List[Any], key_func: Callable = None, reverse: bool = False) -> List[Any]:
        """Implement bubble sort algorithm."""
        print(f"Sorting {len(data)} items using Bubble Sort")
        
        n = len(data)
        comparisons = 0
        swaps = 0
        
        for i in range(n):
            for j in range(0, n - i - 1):
                comparisons += 1
                
                # Get comparison values
                val1 = key_func(data[j]) if key_func else data[j]
                val2 = key_func(data[j + 1]) if key_func else data[j + 1]
                
                # Compare based on reverse flag
                should_swap = (val1 > val2) if not reverse else (val1 < val2)
                
                if should_swap:
                    data[j], data[j + 1] = data[j + 1], data[j]
                    swaps += 1
        
        print(f"Bubble Sort completed: {comparisons} comparisons, {swaps} swaps")
        return data
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get bubble sort strategy information."""
        return {
            'name': 'Bubble Sort',
            'time_complexity': 'O(n²)',
            'space_complexity': 'O(1)',
            'stable': True,
            'in_place': True,
            'best_case': 'O(n)',
            'worst_case': 'O(n²)',
            'description': 'Simple comparison-based sorting algorithm'
        }


class QuickSortStrategy(SortingStrategy):
    """Quick sort strategy - O(n log n) average time complexity."""
    
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
    
    def sort(self, data: List[Any], key_func: Callable = None, reverse: bool = False) -> List[Any]:
        """Implement quick sort algorithm."""
        print(f"Sorting {len(data)} items using Quick Sort")
        
        self.comparisons = 0
        self.swaps = 0
        
        self._quick_sort(data, 0, len(data) - 1, key_func, reverse)
        
        print(f"Quick Sort completed: {self.comparisons} comparisons, {self.swaps} swaps")
        return data
    
    def _quick_sort(self, data: List[Any], low: int, high: int, key_func: Callable, reverse: bool) -> None:
        """Recursive quick sort implementation."""
        if low < high:
            pivot_index = self._partition(data, low, high, key_func, reverse)
            self._quick_sort(data, low, pivot_index - 1, key_func, reverse)
            self._quick_sort(data, pivot_index + 1, high, key_func, reverse)
    
    def _partition(self, data: List[Any], low: int, high: int, key_func: Callable, reverse: bool) -> int:
        """Partition function for quick sort."""
        pivot_val = key_func(data[high]) if key_func else data[high]
        i = low - 1
        
        for j in range(low, high):
            self.comparisons += 1
            
            current_val = key_func(data[j]) if key_func else data[j]
            should_swap = (current_val <= pivot_val) if not reverse else (current_val >= pivot_val)
            
            if should_swap:
                i += 1
                data[i], data[j] = data[j], data[i]
                self.swaps += 1
        
        data[i + 1], data[high] = data[high], data[i + 1]
        self.swaps += 1
        return i + 1
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get quick sort strategy information."""
        return {
            'name': 'Quick Sort',
            'time_complexity': 'O(n log n) average',
            'space_complexity': 'O(log n)',
            'stable': False,
            'in_place': True,
            'best_case': 'O(n log n)',
            'worst_case': 'O(n²)',
            'description': 'Divide-and-conquer sorting algorithm'
        }


class MergeSortStrategy(SortingStrategy):
    """Merge sort strategy - O(n log n) guaranteed time complexity."""
    
    def __init__(self):
        self.comparisons = 0
        self.merges = 0
    
    def sort(self, data: List[Any], key_func: Callable = None, reverse: bool = False) -> List[Any]:
        """Implement merge sort algorithm."""
        print(f"Sorting {len(data)} items using Merge Sort")
        
        self.comparisons = 0
        self.merges = 0
        
        result = self._merge_sort(data, key_func, reverse)
        
        print(f"Merge Sort completed: {self.comparisons} comparisons, {self.merges} merges")
        return result
    
    def _merge_sort(self, data: List[Any], key_func: Callable, reverse: bool) -> List[Any]:
        """Recursive merge sort implementation."""
        if len(data) <= 1:
            return data
        
        mid = len(data) // 2
        left = self._merge_sort(data[:mid], key_func, reverse)
        right = self._merge_sort(data[mid:], key_func, reverse)
        
        return self._merge(left, right, key_func, reverse)
    
    def _merge(self, left: List[Any], right: List[Any], key_func: Callable, reverse: bool) -> List[Any]:
        """Merge two sorted arrays."""
        result = []
        i = j = 0
        self.merges += 1
        
        while i < len(left) and j < len(right):
            self.comparisons += 1
            
            left_val = key_func(left[i]) if key_func else left[i]
            right_val = key_func(right[j]) if key_func else right[j]
            
            should_take_left = (left_val <= right_val) if not reverse else (left_val >= right_val)
            
            if should_take_left:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        # Add remaining elements
        result.extend(left[i:])
        result.extend(right[j:])
        
        return result
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get merge sort strategy information."""
        return {
            'name': 'Merge Sort',
            'time_complexity': 'O(n log n)',
            'space_complexity': 'O(n)',
            'stable': True,
            'in_place': False,
            'best_case': 'O(n log n)',
            'worst_case': 'O(n log n)',
            'description': 'Stable divide-and-conquer sorting algorithm'
        }


# ============================================================================
# COMPRESSION STRATEGIES
# ============================================================================

class CompressionStrategy(Strategy):
    """Abstract compression strategy."""
    
    @abstractmethod
    def compress(self, data: str) -> Dict[str, Any]:
        """Compress data using this strategy."""
        pass
    
    @abstractmethod
    def decompress(self, compressed_data: str) -> str:
        """Decompress data using this strategy."""
        pass
    
    def execute(self, context: Context, data: str, operation: str = 'compress') -> Any:
        """Execute compression or decompression."""
        if operation == 'compress':
            return self.compress(data)
        elif operation == 'decompress':
            return self.decompress(data)
        else:
            raise ValueError(f"Unknown operation: {operation}")


class RunLengthCompressionStrategy(CompressionStrategy):
    """Run-length encoding compression strategy."""
    
    def compress(self, data: str) -> Dict[str, Any]:
        """Compress using run-length encoding."""
        print(f"Compressing {len(data)} characters using Run-Length Encoding")
        
        if not data:
            return {'compressed': '', 'original_size': 0, 'compressed_size': 0, 'ratio': 0}
        
        compressed = []
        current_char = data[0]
        count = 1
        
        for char in data[1:]:
            if char == current_char:
                count += 1
            else:
                compressed.append(f"{count}{current_char}")
                current_char = char
                count = 1
        
        compressed.append(f"{count}{current_char}")
        compressed_str = ''.join(compressed)
        
        original_size = len(data)
        compressed_size = len(compressed_str)
        ratio = (original_size - compressed_size) / original_size * 100 if original_size > 0 else 0
        
        return {
            'compressed': compressed_str,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'ratio': ratio,
            'algorithm': 'Run-Length Encoding'
        }
    
    def decompress(self, compressed_data: str) -> str:
        """Decompress run-length encoded data."""
        print(f"Decompressing {len(compressed_data)} characters using Run-Length Encoding")
        
        result = []
        i = 0
        
        while i < len(compressed_data):
            # Read count
            count_str = ''
            while i < len(compressed_data) and compressed_data[i].isdigit():
                count_str += compressed_data[i]
                i += 1
            
            # Read character
            if i < len(compressed_data):
                char = compressed_data[i]
                count = int(count_str) if count_str else 1
                result.append(char * count)
                i += 1
        
        return ''.join(result)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get run-length compression strategy information."""
        return {
            'name': 'Run-Length Encoding',
            'type': 'Lossless compression',
            'best_for': 'Data with many consecutive repeated characters',
            'time_complexity': 'O(n)',
            'space_complexity': 'O(n)',
            'compression_ratio': 'Variable (depends on data repetition)'
        }


class HuffmanCompressionStrategy(CompressionStrategy):
    """Huffman coding compression strategy (simplified)."""
    
    def compress(self, data: str) -> Dict[str, Any]:
        """Compress using simplified Huffman coding."""
        print(f"Compressing {len(data)} characters using Huffman Coding")
        
        if not data:
            return {'compressed': '', 'original_size': 0, 'compressed_size': 0, 'ratio': 0}
        
        # Count character frequencies
        freq = {}
        for char in data:
            freq[char] = freq.get(char, 0) + 1
        
        # Create simple encoding (more frequent = shorter codes)
        sorted_chars = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        encoding = {}
        
        # Assign binary codes (simplified - not actual Huffman tree)
        for i, (char, _) in enumerate(sorted_chars):
            encoding[char] = format(i, f'0{len(bin(len(sorted_chars)-1))-2}b')
        
        # Encode data
        compressed_bits = ''.join(encoding[char] for char in data)
        
        # Convert to string representation
        compressed_str = f"HUFFMAN:{json.dumps(encoding)}:{compressed_bits}"
        
        original_size = len(data) * 8  # 8 bits per character
        compressed_size = len(compressed_bits) + len(json.dumps(encoding)) * 8
        ratio = (original_size - compressed_size) / original_size * 100 if original_size > 0 else 0
        
        return {
            'compressed': compressed_str,
            'original_size': len(data),
            'compressed_size': len(compressed_str),
            'ratio': ratio,
            'algorithm': 'Huffman Coding',
            'encoding_table': encoding
        }
    
    def decompress(self, compressed_data: str) -> str:
        """Decompress Huffman coded data."""
        print(f"Decompressing Huffman coded data")
        
        if not compressed_data.startswith('HUFFMAN:'):
            raise ValueError("Invalid Huffman compressed data format")
        
        # Parse compressed data
        parts = compressed_data[8:].split(':', 2)  # Remove 'HUFFMAN:' prefix
        encoding = json.loads(parts[0])
        compressed_bits = parts[1]
        
        # Create decoding table
        decoding = {code: char for char, code in encoding.items()}
        
        # Decode bits
        result = []
        i = 0
        while i < len(compressed_bits):
            for code_length in range(1, len(compressed_bits) - i + 1):
                code = compressed_bits[i:i + code_length]
                if code in decoding:
                    result.append(decoding[code])
                    i += code_length
                    break
            else:
                break  # No valid code found
        
        return ''.join(result)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get Huffman compression strategy information."""
        return {
            'name': 'Huffman Coding',
            'type': 'Lossless compression',
            'best_for': 'Text data with varying character frequencies',
            'time_complexity': 'O(n log n)',
            'space_complexity': 'O(n)',
            'compression_ratio': 'Typically 20-90% depending on data'
        }


# ============================================================================
# STRATEGY FACTORY AND MANAGER
# ============================================================================

class StrategyFactory:
    """Factory for creating and managing strategies."""
    
    def __init__(self):
        self._payment_strategies = {
            'credit_card': CreditCardStrategy,
            'paypal': PayPalStrategy,
            'bank_transfer': BankTransferStrategy,
            'bitcoin': lambda: CryptocurrencyStrategy('Bitcoin'),
            'ethereum': lambda: CryptocurrencyStrategy('Ethereum')
        }
        
        self._sorting_strategies = {
            'bubble': BubbleSortStrategy,
            'quick': QuickSortStrategy,
            'merge': MergeSortStrategy
        }
        
        self._compression_strategies = {
            'runlength': RunLengthCompressionStrategy,
            'huffman': HuffmanCompressionStrategy
        }
    
    def create_payment_strategy(self, strategy_name: str) -> PaymentStrategy:
        """Create payment strategy by name."""
        if strategy_name not in self._payment_strategies:
            raise ValueError(f"Unknown payment strategy: {strategy_name}")
        
        strategy_class = self._payment_strategies[strategy_name]
        return strategy_class()
    
    def create_sorting_strategy(self, strategy_name: str) -> SortingStrategy:
        """Create sorting strategy by name."""
        if strategy_name not in self._sorting_strategies:
            raise ValueError(f"Unknown sorting strategy: {strategy_name}")
        
        strategy_class = self._sorting_strategies[strategy_name]
        return strategy_class()
    
    def create_compression_strategy(self, strategy_name: str) -> CompressionStrategy:
        """Create compression strategy by name."""
        if strategy_name not in self._compression_strategies:
            raise ValueError(f"Unknown compression strategy: {strategy_name}")
        
        strategy_class = self._compression_strategies[strategy_name]
        return strategy_class()
    
    def get_available_strategies(self) -> Dict[str, List[str]]:
        """Get list of available strategies by category."""
        return {
            'payment': list(self._payment_strategies.keys()),
            'sorting': list(self._sorting_strategies.keys()),
            'compression': list(self._compression_strategies.keys())
        }
    
    def register_strategy(self, category: str, name: str, strategy_class: type) -> None:
        """Register a new strategy."""
        if category == 'payment':
            self._payment_strategies[name] = strategy_class
        elif category == 'sorting':
            self._sorting_strategies[name] = strategy_class
        elif category == 'compression':
            self._compression_strategies[name] = strategy_class
        else:
            raise ValueError(f"Unknown strategy category: {category}")
        
        print(f"Registered {category} strategy: {name}")


class PaymentProcessor(Context):
    """Payment processor using strategy pattern."""
    
    def __init__(self):
        super().__init__()
        self.processed_payments: List[Dict[str, Any]] = []
    
    def process_payment(self, amount: float, currency: str, payment_details: Dict[str, Any]) -> Dict[str, Any]:
        """Process payment using current strategy."""
        result = self.execute_strategy(amount, currency, payment_details)
        
        if result['success']:
            self.processed_payments.append(result)
        
        return result
    
    def get_payment_stats(self) -> Dict[str, Any]:
        """Get payment processing statistics."""
        if not self.processed_payments:
            return {'total_payments': 0}
        
        total_amount = sum(p['amount'] for p in self.processed_payments)
        total_fees = sum(p.get('fee', 0) for p in self.processed_payments)
        
        payment_methods = {}
        for payment in self.processed_payments:
            method = payment['payment_method']
            payment_methods[method] = payment_methods.get(method, 0) + 1
        
        return {
            'total_payments': len(self.processed_payments),
            'total_amount': total_amount,
            'total_fees': total_fees,
            'payment_methods': payment_methods,
            'average_amount': total_amount / len(self.processed_payments)
        }


def demonstrate_strategy_pattern():
    """
    Demonstrate Strategy pattern implementations.
    """
    print("=== STRATEGY PATTERN DEMONSTRATION ===\n")
    
    # 1. Payment Processing Strategies
    print("1. PAYMENT PROCESSING STRATEGIES:")
    
    # Create payment processor
    payment_processor = PaymentProcessor()
    
    # Create different payment strategies
    credit_card = CreditCardStrategy()
    paypal = PayPalStrategy()
    bank_transfer = BankTransferStrategy()
    bitcoin = CryptocurrencyStrategy('Bitcoin')
    
    # Test different payment methods
    test_payments = [
        {
            'amount': 100.00,
            'currency': 'USD',
            'strategy': credit_card,
            'details': {
                'card_number': '1234567890123456',
                'expiry_date': '12/25',
                'cvv': '123'
            }
        },
        {
            'amount': 250.00,
            'currency': 'USD',
            'strategy': paypal,
            'details': {
                'email': 'user@example.com',
                'password': 'securepassword'
            }
        },
        {
            'amount': 500.00,
            'currency': 'USD',
            'strategy': bank_transfer,
            'details': {
                'account_number': '12345678901',
                'routing_number': '123456789'
            }
        },
        {
            'amount': 75.00,
            'currency': 'USD',
            'strategy': bitcoin,
            'details': {
                'wallet_address': '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
                'private_key_hash': 'abc123def456'
            }
        }
    ]
    
    print("   Processing payments with different strategies:")
    for payment in test_payments:
        payment_processor.set_strategy(payment['strategy'])
        
        result = payment_processor.process_payment(
            payment['amount'],
            payment['currency'],
            payment['details']
        )
        
        if result['success']:
            print(f"   ✓ {result['payment_method']}: ${result['amount']:.2f} "
                  f"(Fee: ${result.get('fee', 0):.2f})")
        else:
            print(f"   ✗ Payment failed: {result['error']}")
    
    # Show payment statistics
    payment_stats = payment_processor.get_payment_stats()
    print(f"\n   Payment Statistics:")
    print(f"     Total payments: {payment_stats['total_payments']}")
    print(f"     Total amount: ${payment_stats['total_amount']:.2f}")
    print(f"     Total fees: ${payment_stats['total_fees']:.2f}")
    print(f"     Payment methods: {payment_stats['payment_methods']}")
    
    print()
    
    # 2. Sorting Strategies
    print("2. SORTING STRATEGIES:")
    
    # Create sorting context
    sorter = Context()
    
    # Create test data
    test_data = [64, 34, 25, 12, 22, 11, 90, 5, 77, 30]
    print(f"   Original data: {test_data}")
    
    # Test different sorting strategies
    sorting_strategies = [
        ('Bubble Sort', BubbleSortStrategy()),
        ('Quick Sort', QuickSortStrategy()),
        ('Merge Sort', MergeSortStrategy())
    ]
    
    print(f"\n   Testing different sorting algorithms:")
    for name, strategy in sorting_strategies:
        sorter.set_strategy(strategy)
        
        start_time = time.time()
        sorted_data = sorter.execute_strategy(test_data)
        end_time = time.time()
        
        print(f"   {name}: {sorted_data}")
        print(f"     Execution time: {(end_time - start_time)*1000:.2f}ms")
        
        # Show strategy info
        info = strategy.get_strategy_info()
        print(f"     Time complexity: {info['time_complexity']}")
        print(f"     Space complexity: {info['space_complexity']}")
        print(f"     Stable: {info['stable']}")
    
    # Test sorting with custom key function
    print(f"\n   Sorting strings by length:")
    string_data = ['apple', 'pie', 'washington', 'book', 'python', 'a']
    
    sorter.set_strategy(MergeSortStrategy())
    sorted_strings = sorter.execute_strategy(string_data, key_func=len)
    print(f"   Sorted by length: {sorted_strings}")
    
    # Show execution statistics
    exec_stats = sorter.get_execution_stats()
    print(f"\n   Execution Statistics:")
    print(f"     Total executions: {exec_stats['total_executions']}")
    print(f"     Most used strategy: {exec_stats['most_used_strategy']}")
    print(f"     Average execution time: {exec_stats['average_time']*1000:.2f}ms")
    
    print()
    
    # 3. Compression Strategies
    print("3. COMPRESSION STRATEGIES:")
    
    # Create compression context
    compressor = Context()
    
    # Test data with different characteristics
    test_texts = [
        ("Repetitive text", "aaaaaabbbbbbccccccdddddd"),
        ("Normal text", "The quick brown fox jumps over the lazy dog"),
        ("Mixed content", "Hello World! 123 Hello World! 456 Hello World!")
    ]
    
    compression_strategies = [
        ('Run-Length Encoding', RunLengthCompressionStrategy()),
        ('Huffman Coding', HuffmanCompressionStrategy())
    ]
    
    print("   Testing compression strategies:")
    for text_name, text_data in test_texts:
        print(f"\n   {text_name}: '{text_data}'")
        
        for strategy_name, strategy in compression_strategies:
            compressor.set_strategy(strategy)
            
            # Compress
            compressed_result = compressor.execute_strategy(text_data, 'compress')
            
            # Decompress
            decompressed = compressor.execute_strategy(compressed_result['compressed'], 'decompress')
            
            print(f"     {strategy_name}:")
            print(f"       Original size: {compressed_result['original_size']} chars")
            print(f"       Compressed size: {compressed_result['compressed_size']} chars")
            print(f"       Compression ratio: {compressed_result['ratio']:.1f}%")
            print(f"       Decompression successful: {decompressed == text_data}")
    
    print()
    
    # 4. Strategy Factory
    print("4. STRATEGY FACTORY:")
    
    factory = StrategyFactory()
    
    # Show available strategies
    available = factory.get_available_strategies()
    print("   Available strategies:")
    for category, strategies in available.items():
        print(f"     {category.title()}: {strategies}")
    
    # Create strategies using factory
    print(f"\n   Creating strategies using factory:")
    
    # Payment strategies
    cc_strategy = factory.create_payment_strategy('credit_card')
    pp_strategy = factory.create_payment_strategy('paypal')
    
    print(f"   Created payment strategies:")
    print(f"     Credit Card: {cc_strategy.get_strategy_info()['name']}")
    print(f"     PayPal: {pp_strategy.get_strategy_info()['name']}")
    
    # Sorting strategies
    quick_sort = factory.create_sorting_strategy('quick')
    merge_sort = factory.create_sorting_strategy('merge')
    
    print(f"   Created sorting strategies:")
    print(f"     Quick Sort: {quick_sort.get_strategy_info()['name']}")
    print(f"     Merge Sort: {merge_sort.get_strategy_info()['name']}")
    
    # Register custom strategy
    class CustomSortStrategy(SortingStrategy):
        def sort(self, data: List[Any], key_func: Callable = None, reverse: bool = False) -> List[Any]:
            return sorted(data, key=key_func, reverse=reverse)
        
        def get_strategy_info(self) -> Dict[str, Any]:
            return {'name': 'Python Built-in Sort', 'time_complexity': 'O(n log n)'}
    
    factory.register_strategy('sorting', 'python_builtin', CustomSortStrategy)
    builtin_sort = factory.create_sorting_strategy('python_builtin')
    
    print(f"   Registered and created custom strategy:")
    print(f"     Python Built-in: {builtin_sort.get_strategy_info()['name']}")
    
    print()
    
    # 5. Dynamic Strategy Selection
    print("5. DYNAMIC STRATEGY SELECTION:")
    
    class SmartPaymentProcessor(PaymentProcessor):
        """Payment processor that selects strategy based on amount and preferences."""
        
        def __init__(self):
            super().__init__()
            self.strategies = {
                'small': CreditCardStrategy(),      # < $100
                'medium': PayPalStrategy(),         # $100 - $1000
                'large': BankTransferStrategy(),    # > $1000
                'crypto': CryptocurrencyStrategy('Bitcoin')  # For crypto preference
            }
        
        def auto_select_strategy(self, amount: float, preferences: Dict[str, Any] = None) -> str:
            """Automatically select best strategy based on amount and preferences."""
            preferences = preferences or {}
            
            # Check for crypto preference
            if preferences.get('prefer_crypto', False):
                self.set_strategy(self.strategies['crypto'])
                return 'crypto'
            
            # Select based on amount
            if amount < 100:
                self.set_strategy(self.strategies['small'])
                return 'small'
            elif amount <= 1000:
                self.set_strategy(self.strategies['medium'])
                return 'medium'
            else:
                self.set_strategy(self.strategies['large'])
                return 'large'
    
    smart_processor = SmartPaymentProcessor()
    
    # Test automatic strategy selection
    test_amounts = [50, 500, 2000, 100]
    test_preferences = [{}, {}, {}, {'prefer_crypto': True}]
    
    print("   Automatic strategy selection based on amount:")
    for amount, prefs in zip(test_amounts, test_preferences):
        selected = smart_processor.auto_select_strategy(amount, prefs)
        current_strategy = smart_processor.get_current_strategy()
        
        print(f"     ${amount}: Selected {selected} strategy ({current_strategy.__class__.__name__})")
    
    print()
    
    # 6. Strategy Pattern Benefits
    print("6. STRATEGY PATTERN BENEFITS:")
    print("   ✓ Algorithm Flexibility: Easy to switch between different algorithms")
    print("   ✓ Runtime Selection: Algorithms can be chosen at runtime")
    print("   ✓ Open/Closed Principle: Easy to add new strategies without modifying existing code")
    print("   ✓ Testability: Each strategy can be tested independently")
    print("   ✓ Code Reuse: Strategies can be reused across different contexts")
    print("   ✓ Separation of Concerns: Algorithm implementation separated from usage")
    print("   ✓ Maintainability: Changes to one algorithm don't affect others")
    print("   ✓ Performance Optimization: Choose optimal algorithm for specific scenarios")
    print()
    
    print("=== STRATEGY PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_strategy_pattern()
