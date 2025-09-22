# Troubleshooting Common Issues: Agent Development Debug Guide

## Issue Categories Quick Reference

| Issue Type | Symptoms | Common Causes | Diagnostic Priority |
|------------|----------|---------------|-------------------|
| **Performance** | Slow responses, timeouts | Model size, inefficient code | High |
| **Memory** | OOM errors, crashes | Memory leaks, large models | Critical |
| **Integration** | API failures, connection errors | Config issues, network problems | High |
| **Logic** | Wrong outputs, unexpected behavior | Prompt issues, logic bugs | High |
| **Deployment** | Won't start, connection refused | Config, dependencies, ports | Critical |
| **Scaling** | Performance degrades with load | Resource limits, bottlenecks | Medium |

---

## Performance Issues

### **1. Slow Response Times**

**Symptoms:**
- Agent takes >5 seconds to respond
- Timeouts in production
- High CPU usage

**Diagnostic Steps:**
```python
import time
import cProfile
import pstats
from functools import wraps

def performance_profiler(func):
    """Decorator to profile function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Time measurement
        start_time = time.time()
        
        # Memory profiling
        import tracemalloc
        tracemalloc.start()
        
        # CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Stop profiling
            profiler.disable()
            
            # Print timing
            end_time = time.time()
            print(f"Execution time: {end_time - start_time:.2f} seconds")
            
            # Print memory usage
            current, peak = tracemalloc.get_traced_memory()
            print(f"Memory usage: {current / 1024 / 1024:.1f} MB (peak: {peak / 1024 / 1024:.1f} MB)")
            tracemalloc.stop()
            
            # Print CPU profile
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(10)  # Top 10 functions
    
    return wrapper

# Usage
class DiagnosticAgent:
    @performance_profiler
    def process_request(self, request):
        return self.agent.process(request)
```

**Common Solutions:**
```python
# 1. Model optimization
def optimize_model_inference():
    # Use smaller model variants
    model = load_model("distilbert-base-uncased")  # Instead of bert-large
    
    # Quantization
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "model-name",
        torch_dtype=torch.float16,  # Half precision
        device_map="auto"
    )
    
    # Caching frequent computations
    from functools import lru_cache
    
    @lru_cache(maxsize=1000)
    def cached_embedding(text):
        return model.encode(text)

# 2. Batch processing
async def optimize_batch_processing(requests):
    # Process multiple requests together
    if len(requests) > 1:
        batch_result = await process_batch(requests)
        return batch_result
    else:
        return await process_single(requests[0])

# 3. Async optimization
async def optimize_io_operations():
    # Use async for I/O operations
    import aiohttp
    import asyncio
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            session.get(url) for url in urls
        ]
        responses = await asyncio.gather(*tasks)
    
    return responses
```

### **2. Memory Issues**

**Symptoms:**
- Out of Memory (OOM) errors
- Gradual memory increase
- System becomes unresponsive

**Memory Leak Detection:**
```python
import gc
import tracemalloc
import psutil
import weakref

class MemoryDiagnostic:
    def __init__(self):
        self.baseline_memory = None
        self.memory_snapshots = []
        tracemalloc.start()
    
    def start_monitoring(self):
        """Start memory monitoring"""
        self.baseline_memory = self.get_memory_usage()
        self.memory_snapshots = []
    
    def get_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process()
        return {
            'rss': process.memory_info().rss / 1024 / 1024,  # MB
            'vms': process.memory_info().vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }
    
    def take_snapshot(self, label=""):
        """Take memory snapshot"""
        snapshot = tracemalloc.take_snapshot()
        memory_usage = self.get_memory_usage()
        
        self.memory_snapshots.append({
            'label': label,
            'snapshot': snapshot,
            'memory_usage': memory_usage,
            'timestamp': time.time()
        })
    
    def analyze_memory_growth(self):
        """Analyze memory growth between snapshots"""
        if len(self.memory_snapshots) < 2:
            return "Need at least 2 snapshots"
        
        current = self.memory_snapshots[-1]['snapshot']
        previous = self.memory_snapshots[-2]['snapshot']
        
        # Compare snapshots
        top_stats = current.compare_to(previous, 'lineno')
        
        print("Top 10 memory growth:")
        for stat in top_stats[:10]:
            print(stat)
        
        # Memory usage growth
        current_mem = self.memory_snapshots[-1]['memory_usage']
        previous_mem = self.memory_snapshots[-2]['memory_usage']
        
        growth = current_mem['rss'] - previous_mem['rss']
        print(f"Memory growth: {growth:.1f} MB")
        
        return top_stats
    
    def find_memory_leaks(self):
        """Find potential memory leaks"""
        # Check for uncollectable objects
        gc.collect()
        uncollectable = len(gc.garbage)
        
        # Check object counts
        object_counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        # Sort by count
        sorted_objects = sorted(
            object_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            'uncollectable_objects': uncollectable,
            'top_object_types': sorted_objects[:20]
        }

# Memory-efficient practices
class MemoryEfficientAgent:
    def __init__(self):
        self.cache = weakref.WeakValueDictionary()  # Automatic cleanup
        self.memory_diagnostic = MemoryDiagnostic()
    
    def process_large_dataset(self, data_iterator):
        """Process large dataset without loading all into memory"""
        for batch in self.batch_iterator(data_iterator, batch_size=1000):
            # Process batch
            results = self.process_batch(batch)
            
            # Yield results immediately (don't accumulate)
            yield from results
            
            # Force garbage collection periodically
            if random.random() < 0.1:  # 10% chance
                gc.collect()
    
    def batch_iterator(self, iterator, batch_size):
        """Create batches from iterator"""
        batch = []
        for item in iterator:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []  # Important: clear batch
        
        if batch:
            yield batch
```

---

## Integration Issues

### **3. API Connection Problems**

**Symptoms:**
- Connection refused errors
- Timeout exceptions
- Authentication failures

**Diagnostic Tools:**
```python
import requests
import aiohttp
import asyncio
import time
from typing import Dict, Any

class APIConnectionDiagnostic:
    def __init__(self, base_url: str, headers: Dict[str, str] = None):
        self.base_url = base_url
        self.headers = headers or {}
        self.test_results = {}
    
    async def comprehensive_api_test(self):
        """Run comprehensive API connectivity tests"""
        tests = [
            ("connectivity", self.test_basic_connectivity),
            ("authentication", self.test_authentication),
            ("rate_limits", self.test_rate_limits),
            ("timeout_handling", self.test_timeout_handling),
            ("error_responses", self.test_error_responses)
        ]
        
        for test_name, test_func in tests:
            try:
                print(f"Running {test_name} test...")
                result = await test_func()
                self.test_results[test_name] = {"status": "passed", "result": result}
            except Exception as e:
                self.test_results[test_name] = {"status": "failed", "error": str(e)}
        
        return self.test_results
    
    async def test_basic_connectivity(self):
        """Test basic API connectivity"""
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(
                    f"{self.base_url}/health",
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    end_time = time.time()
                    
                    return {
                        "status_code": response.status,
                        "response_time": end_time - start_time,
                        "reachable": True
                    }
        except Exception as e:
            return {"reachable": False, "error": str(e)}
    
    async def test_authentication(self):
        """Test API authentication"""
        test_endpoints = [
            "/api/protected",
            "/api/user/profile"
        ]
        
        results = {}
        async with aiohttp.ClientSession() as session:
            for endpoint in test_endpoints:
                try:
                    async with session.get(
                        f"{self.base_url}{endpoint}",
                        headers=self.headers
                    ) as response:
                        results[endpoint] = {
                            "status_code": response.status,
                            "authenticated": response.status != 401
                        }
                except Exception as e:
                    results[endpoint] = {"error": str(e)}
        
        return results
    
    async def test_rate_limits(self):
        """Test API rate limiting behavior"""
        async with aiohttp.ClientSession() as session:
            responses = []
            
            # Make rapid requests to test rate limiting
            tasks = []
            for i in range(20):
                task = self.make_request(session, f"{self.base_url}/api/test")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            rate_limited = sum(1 for r in results if isinstance(r, dict) and r.get('status_code') == 429)
            
            return {
                "total_requests": len(results),
                "rate_limited": rate_limited,
                "rate_limit_triggered": rate_limited > 0
            }
    
    async def make_request(self, session, url):
        """Make single request with error handling"""
        try:
            async with session.get(url, headers=self.headers) as response:
                return {"status_code": response.status}
        except Exception as e:
            return {"error": str(e)}

# Connection retry logic
class RobustAPIClient:
    def __init__(self, base_url: str, max_retries: int = 3):
        self.base_url = base_url
        self.max_retries = max_retries
        self.session = None
    
    async def make_robust_request(self, endpoint: str, method: str = "GET", **kwargs):
        """Make API request with retry logic"""
        if not self.session:
            await self.initialize_session()
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Exponential backoff
                if attempt > 0:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                
                async with self.session.request(
                    method,
                    f"{self.base_url}{endpoint}",
                    **kwargs
                ) as response:
                    if response.status < 500:  # Don't retry client errors
                        return await response.json()
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                print(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries:
                    break
        
        raise Exception(f"All {self.max_retries + 1} attempts failed. Last error: {last_exception}")
    
    async def initialize_session(self):
        """Initialize HTTP session with proper configuration"""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30
        )
        
        timeout = aiohttp.ClientTimeout(
            total=30,
            connect=10,
            sock_read=10
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
```

### **4. Configuration Issues**

**Common Configuration Problems:**
```python
import os
import json
from typing import Any, Dict
import yaml

class ConfigurationValidator:
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = {}
        self.validation_errors = []
    
    def load_and_validate_config(self):
        """Load and validate configuration"""
        try:
            # Load configuration
            self.config = self.load_config()
            
            # Validate required fields
            self.validate_required_fields()
            
            # Validate field types and values
            self.validate_field_values()
            
            # Validate dependencies
            self.validate_dependencies()
            
            return len(self.validation_errors) == 0
            
        except Exception as e:
            self.validation_errors.append(f"Configuration loading failed: {e}")
            return False
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from various sources"""
        config = {}
        
        # 1. Load from file
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.json'):
                    config.update(json.load(f))
                elif self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config.update(yaml.safe_load(f))
        
        # 2. Override with environment variables
        env_config = self.load_from_environment()
        config.update(env_config)
        
        return config
    
    def load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # Define environment variable mappings
        env_mappings = {
            'OPENAI_API_KEY': 'openai.api_key',
            'DATABASE_URL': 'database.url',
            'REDIS_URL': 'redis.url',
            'LOG_LEVEL': 'logging.level',
            'MAX_WORKERS': 'server.max_workers'
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                # Set nested configuration
                self.set_nested_config(env_config, config_path, value)
        
        return env_config
    
    def set_nested_config(self, config: dict, path: str, value: Any):
        """Set nested configuration value"""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def validate_required_fields(self):
        """Validate required configuration fields"""
        required_fields = [
            'openai.api_key',
            'database.url',
            'server.port'
        ]
        
        for field in required_fields:
            if not self.get_nested_config(self.config, field):
                self.validation_errors.append(f"Required field missing: {field}")
    
    def get_nested_config(self, config: dict, path: str):
        """Get nested configuration value"""
        keys = path.split('.')
        current = config
        
        for key in keys:
            if key not in current:
                return None
            current = current[key]
        
        return current
    
    def validate_field_values(self):
        """Validate configuration field values"""
        validations = [
            ('server.port', self.validate_port),
            ('database.url', self.validate_database_url),
            ('openai.api_key', self.validate_api_key),
            ('logging.level', self.validate_log_level)
        ]
        
        for field_path, validator in validations:
            value = self.get_nested_config(self.config, field_path)
            if value and not validator(value):
                self.validation_errors.append(f"Invalid value for {field_path}: {value}")
    
    def validate_port(self, port) -> bool:
        """Validate port number"""
        try:
            port_num = int(port)
            return 1 <= port_num <= 65535
        except ValueError:
            return False
    
    def validate_database_url(self, url) -> bool:
        """Validate database URL format"""
        return url.startswith(('postgresql://', 'mysql://', 'sqlite:///'))
    
    def validate_api_key(self, api_key) -> bool:
        """Validate API key format"""
        return len(api_key) > 10 and api_key.startswith('sk-')
    
    def validate_log_level(self, level) -> bool:
        """Validate log level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        return level.upper() in valid_levels
    
    def print_validation_report(self):
        """Print configuration validation report"""
        if not self.validation_errors:
            print("✅ Configuration validation passed")
        else:
            print("❌ Configuration validation failed:")
            for error in self.validation_errors:
                print(f"  - {error}")

# Usage
config_validator = ConfigurationValidator("config.yaml")
if not config_validator.load_and_validate_config():
    config_validator.print_validation_report()
    exit(1)
```

---

## Logic and Behavior Issues

### **5. Incorrect Agent Responses**

**Debugging Agent Logic:**
```python
import logging
from typing import List, Dict, Any

class AgentDebugger:
    def __init__(self, agent):
        self.agent = agent
        self.debug_log = []
        self.logger = logging.getLogger(__name__)
        
        # Setup detailed logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def debug_response_generation(self, input_text: str):
        """Debug the complete response generation process"""
        debug_info = {
            'input': input_text,
            'steps': [],
            'final_output': None,
            'errors': []
        }
        
        try:
            # Step 1: Input processing
            processed_input = self.debug_input_processing(input_text)
            debug_info['steps'].append({
                'step': 'input_processing',
                'input': input_text,
                'output': processed_input
            })
            
            # Step 2: Intent recognition
            intent = self.debug_intent_recognition(processed_input)
            debug_info['steps'].append({
                'step': 'intent_recognition',
                'input': processed_input,
                'output': intent
            })
            
            # Step 3: Context retrieval
            context = self.debug_context_retrieval(intent)
            debug_info['steps'].append({
                'step': 'context_retrieval',
                'input': intent,
                'output': context
            })
            
            # Step 4: Response generation
            response = self.debug_response_generation_step(intent, context)
            debug_info['steps'].append({
                'step': 'response_generation',
                'input': {'intent': intent, 'context': context},
                'output': response
            })
            
            debug_info['final_output'] = response
            
        except Exception as e:
            debug_info['errors'].append(str(e))
            self.logger.error(f"Debug error: {e}")
        
        self.debug_log.append(debug_info)
        return debug_info
    
    def debug_prompt_effectiveness(self, test_cases: List[Dict[str, str]]):
        """Test prompt effectiveness with multiple test cases"""
        results = []
        
        for i, test_case in enumerate(test_cases):
            input_text = test_case['input']
            expected_output = test_case['expected']
            
            # Generate response
            debug_info = self.debug_response_generation(input_text)
            actual_output = debug_info['final_output']
            
            # Compare with expected
            similarity_score = self.calculate_similarity(actual_output, expected_output)
            
            result = {
                'test_id': i,
                'input': input_text,
                'expected': expected_output,
                'actual': actual_output,
                'similarity_score': similarity_score,
                'passed': similarity_score > 0.8,
                'debug_info': debug_info
            }
            
            results.append(result)
        
        # Generate summary
        passed_tests = sum(1 for r in results if r['passed'])
        summary = {
            'total_tests': len(test_cases),
            'passed_tests': passed_tests,
            'pass_rate': passed_tests / len(test_cases),
            'failed_tests': [r for r in results if not r['passed']]
        }
        
        return {'results': results, 'summary': summary}
    
    def analyze_failure_patterns(self):
        """Analyze patterns in agent failures"""
        failures = []
        
        for debug_entry in self.debug_log:
            if debug_entry['errors'] or self.has_quality_issues(debug_entry):
                failures.append(debug_entry)
        
        # Analyze common failure patterns
        failure_patterns = {
            'input_processing_errors': 0,
            'intent_recognition_errors': 0,
            'context_retrieval_errors': 0,
            'response_generation_errors': 0,
            'quality_issues': 0
        }
        
        for failure in failures:
            for step in failure['steps']:
                if 'error' in step:
                    failure_patterns[f"{step['step']}_errors"] += 1
            
            if self.has_quality_issues(failure):
                failure_patterns['quality_issues'] += 1
        
        return {
            'total_failures': len(failures),
            'failure_patterns': failure_patterns,
            'failure_rate': len(failures) / len(self.debug_log) if self.debug_log else 0
        }

# Prompt optimization
class PromptOptimizer:
    def __init__(self):
        self.prompt_variants = []
        self.test_results = {}
    
    def generate_prompt_variants(self, base_prompt: str) -> List[str]:
        """Generate different prompt variants to test"""
        variants = []
        
        # 1. More specific instructions
        specific_prompt = f"""
        {base_prompt}
        
        Please provide a detailed, step-by-step response.
        Be specific and avoid vague statements.
        """
        variants.append(specific_prompt)
        
        # 2. Add examples
        example_prompt = f"""
        {base_prompt}
        
        Example:
        Input: "How do I reset my password?"
        Output: "To reset your password: 1) Go to login page, 2) Click 'Forgot Password', 3) Enter your email, 4) Check email for reset link"
        
        Now respond to the user's request:
        """
        variants.append(example_prompt)
        
        # 3. Add constraints
        constrained_prompt = f"""
        {base_prompt}
        
        Requirements:
        - Keep response under 200 words
        - Use bullet points for steps
        - Include relevant links if applicable
        """
        variants.append(constrained_prompt)
        
        return variants
    
    def test_prompt_variants(self, variants: List[str], test_cases: List[Dict]):
        """Test different prompt variants"""
        results = {}
        
        for i, prompt in enumerate(variants):
            prompt_results = []
            
            for test_case in test_cases:
                # Test prompt with this case
                response = self.test_prompt_with_case(prompt, test_case)
                prompt_results.append(response)
            
            # Calculate average performance
            avg_score = sum(r['score'] for r in prompt_results) / len(prompt_results)
            
            results[f"variant_{i}"] = {
                'prompt': prompt,
                'average_score': avg_score,
                'individual_results': prompt_results
            }
        
        # Find best performing prompt
        best_variant = max(results.items(), key=lambda x: x[1]['average_score'])
        
        return {
            'all_results': results,
            'best_variant': best_variant,
            'improvement': best_variant[1]['average_score'] - results['variant_0']['average_score']
        }
```

---

## Quick Debugging Checklist

### **🚨 Critical Issues (Fix Immediately)**
- [ ] Check API keys and authentication
- [ ] Verify network connectivity
- [ ] Check memory usage (<80%)
- [ ] Validate configuration files
- [ ] Test with simple inputs first

### **⚡ Performance Issues**
- [ ] Profile slow functions
- [ ] Check for memory leaks
- [ ] Optimize batch processing
- [ ] Review caching strategies
- [ ] Monitor resource usage

### **🔧 Logic Issues**
- [ ] Test with known good inputs
- [ ] Validate prompt templates
- [ ] Check input/output processing
- [ ] Review error handling
- [ ] Add debug logging

### **📊 Monitoring Setup**
- [ ] Add performance metrics
- [ ] Setup error tracking
- [ ] Monitor resource usage
- [ ] Track user satisfaction
- [ ] Log important events

### **🔍 Diagnostic Tools**
```bash
# System diagnostics
htop                    # Monitor CPU/Memory
iostat -x 1            # Monitor disk I/O
netstat -an | grep 8080 # Check port usage
curl -v http://localhost:8080/health  # Test API

# Application diagnostics
python -m cProfile agent.py           # Profile Python code
python -m memory_profiler agent.py    # Profile memory usage
python -m tracemalloc agent.py        # Track memory allocations
```

### **📝 Common Solutions Quick Reference**

| Problem | Quick Fix | Prevention |
|---------|-----------|------------|
| **OOM Error** | Restart service, reduce batch size | Monitor memory, optimize models |
| **Slow Response** | Add caching, optimize queries | Profile regularly, benchmark |
| **API Timeout** | Increase timeout, retry logic | Connection pooling, monitoring |
| **Config Error** | Validate config file | Use config validation |
| **Wrong Output** | Check prompt, add examples | Test with diverse inputs |
| **High CPU** | Optimize algorithms, scale horizontally | Profile and optimize hot paths |

This troubleshooting guide provides systematic approaches to identify, diagnose, and resolve common issues in AI agent development and deployment.
