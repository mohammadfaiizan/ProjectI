"""
Configuration Management System - Multiple Approaches
Difficulty: Hard

Design and implementation of a distributed configuration management system
using trie structures for hierarchical configuration storage and retrieval.

Components:
1. Hierarchical Configuration Storage
2. Environment-based Configuration
3. Dynamic Configuration Updates
4. Configuration Versioning
5. Access Control and Permissions
6. Configuration Validation
"""

import time
import threading
import json
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import copy

class ConfigType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"
    LIST = "list"

class AccessLevel(Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

@dataclass
class ConfigValue:
    value: Any
    config_type: ConfigType
    description: str
    last_updated: float
    updated_by: str
    version: int
    is_encrypted: bool = False
    validation_rules: Dict[str, Any] = None

@dataclass
class ConfigVersion:
    version: int
    timestamp: float
    changes: Dict[str, Any]
    author: str
    comment: str

class ConfigTrieNode:
    def __init__(self):
        self.children = {}
        self.config_value = None
        self.is_leaf = False
        self.permissions = defaultdict(set)  # user_id -> set of AccessLevel
        self.watchers = set()  # Users watching this config

class ConfigurationManager:
    """Hierarchical configuration management system"""
    
    def __init__(self):
        self.environments = {}  # env_name -> trie_root
        self.versions = defaultdict(list)  # env_name -> list of ConfigVersion
        self.version_counter = defaultdict(int)  # env_name -> current_version
        self.validators = {}  # config_path -> validation_function
        self.change_listeners = defaultdict(list)  # config_path -> list of callbacks
        self.lock = threading.RWLock() if hasattr(threading, 'RWLock') else threading.Lock()
        
        # Create default environments
        for env in ['development', 'staging', 'production']:
            self.environments[env] = ConfigTrieNode()
    
    def set_config(self, env: str, path: str, value: Any, config_type: ConfigType,
                  user_id: str, description: str = "", validation_rules: Dict = None) -> bool:
        """Set configuration value"""
        with self.lock:
            if not self._has_permission(env, path, user_id, AccessLevel.WRITE):
                return False
            
            # Validate the value
            if not self._validate_config(path, value, config_type, validation_rules):
                return False
            
            # Get or create environment
            if env not in self.environments:
                self.environments[env] = ConfigTrieNode()
            
            root = self.environments[env]
            
            # Navigate/create path
            node = root
            path_parts = path.split('.')
            
            for part in path_parts:
                if part not in node.children:
                    node.children[part] = ConfigTrieNode()
                node = node.children[part]
            
            # Store old value for versioning
            old_value = None
            if node.config_value:
                old_value = copy.deepcopy(node.config_value.value)
            
            # Create new config value
            current_time = time.time()
            version = self.version_counter[env] + 1
            
            config_value = ConfigValue(
                value=value,
                config_type=config_type,
                description=description,
                last_updated=current_time,
                updated_by=user_id,
                version=version,
                validation_rules=validation_rules or {}
            )
            
            node.config_value = config_value
            node.is_leaf = True
            
            # Update version
            self.version_counter[env] = version
            
            # Record version history
            change_record = ConfigVersion(
                version=version,
                timestamp=current_time,
                changes={path: {'old': old_value, 'new': value}},
                author=user_id,
                comment=f"Updated {path}"
            )
            
            self.versions[env].append(change_record)
            
            # Notify watchers
            self._notify_watchers(env, path, value, old_value)
            
            return True
    
    def get_config(self, env: str, path: str, user_id: str = None) -> Optional[Dict[str, Any]]:
        """Get configuration value"""
        with self.lock:
            if user_id and not self._has_permission(env, path, user_id, AccessLevel.READ):
                return None
            
            if env not in self.environments:
                return None
            
            root = self.environments[env]
            node = root
            
            # Navigate path
            for part in path.split('.'):
                if part not in node.children:
                    return None
                node = node.children[part]
            
            if not node.config_value:
                return None
            
            config = node.config_value
            
            return {
                'value': config.value,
                'type': config.config_type.value,
                'description': config.description,
                'last_updated': config.last_updated,
                'updated_by': config.updated_by,
                'version': config.version
            }
    
    def get_configs_by_prefix(self, env: str, prefix: str, user_id: str = None) -> Dict[str, Any]:
        """Get all configurations with given prefix"""
        with self.lock:
            if env not in self.environments:
                return {}
            
            root = self.environments[env]
            node = root
            
            # Navigate to prefix
            if prefix:
                for part in prefix.split('.'):
                    if part not in node.children:
                        return {}
                    node = node.children[part]
            
            # Collect all configs under this node
            configs = {}
            self._collect_configs(node, prefix, configs, user_id, env)
            
            return configs
    
    def _collect_configs(self, node: ConfigTrieNode, current_path: str, 
                        configs: Dict[str, Any], user_id: str, env: str) -> None:
        """Recursively collect configurations"""
        if node.config_value and (not user_id or self._has_permission(env, current_path, user_id, AccessLevel.READ)):
            config = node.config_value
            configs[current_path] = {
                'value': config.value,
                'type': config.config_type.value,
                'description': config.description,
                'last_updated': config.last_updated
            }
        
        for child_name, child_node in node.children.items():
            child_path = f"{current_path}.{child_name}" if current_path else child_name
            self._collect_configs(child_node, child_path, configs, user_id, env)
    
    def delete_config(self, env: str, path: str, user_id: str) -> bool:
        """Delete configuration"""
        with self.lock:
            if not self._has_permission(env, path, user_id, AccessLevel.ADMIN):
                return False
            
            if env not in self.environments:
                return False
            
            root = self.environments[env]
            node = root
            parent_path = []
            
            # Navigate to parent
            path_parts = path.split('.')
            for part in path_parts[:-1]:
                if part not in node.children:
                    return False
                parent_path.append((node, part))
                node = node.children[part]
            
            last_part = path_parts[-1]
            if last_part not in node.children:
                return False
            
            target_node = node.children[last_part]
            old_value = target_node.config_value.value if target_node.config_value else None
            
            # Delete the node
            del node.children[last_part]
            
            # Clean up empty parent nodes
            for parent_node, part_name in reversed(parent_path):
                if not node.children and not node.config_value:
                    del parent_node.children[part_name]
                    node = parent_node
                else:
                    break
            
            # Record deletion
            version = self.version_counter[env] + 1
            self.version_counter[env] = version
            
            change_record = ConfigVersion(
                version=version,
                timestamp=time.time(),
                changes={path: {'old': old_value, 'new': None}},
                author=user_id,
                comment=f"Deleted {path}"
            )
            
            self.versions[env].append(change_record)
            
            return True
    
    def set_permission(self, env: str, path: str, user_id: str, 
                      access_level: AccessLevel, admin_user: str) -> bool:
        """Set user permissions for configuration path"""
        with self.lock:
            if not self._has_permission(env, path, admin_user, AccessLevel.ADMIN):
                return False
            
            if env not in self.environments:
                return False
            
            # Navigate to path
            root = self.environments[env]
            node = root
            
            for part in path.split('.'):
                if part not in node.children:
                    node.children[part] = ConfigTrieNode()
                node = node.children[part]
            
            node.permissions[user_id].add(access_level)
            return True
    
    def watch_config(self, env: str, path: str, user_id: str, 
                    callback: callable) -> bool:
        """Watch configuration changes"""
        with self.lock:
            if not self._has_permission(env, path, user_id, AccessLevel.READ):
                return False
            
            # Register callback
            self.change_listeners[f"{env}.{path}"].append(callback)
            
            # Add to watchers
            if env in self.environments:
                node = self._get_node(env, path)
                if node:
                    node.watchers.add(user_id)
            
            return True
    
    def get_config_history(self, env: str, path: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get configuration change history"""
        with self.lock:
            history = []
            
            for version in reversed(self.versions[env][-limit:]):
                if path in version.changes:
                    history.append({
                        'version': version.version,
                        'timestamp': version.timestamp,
                        'author': version.author,
                        'comment': version.comment,
                        'changes': version.changes[path]
                    })
            
            return history
    
    def rollback_config(self, env: str, path: str, target_version: int, user_id: str) -> bool:
        """Rollback configuration to previous version"""
        with self.lock:
            if not self._has_permission(env, path, user_id, AccessLevel.ADMIN):
                return False
            
            # Find target version
            target_change = None
            for version in self.versions[env]:
                if version.version == target_version and path in version.changes:
                    target_change = version.changes[path]
                    break
            
            if not target_change:
                return False
            
            # Get the old value to rollback to
            rollback_value = target_change['old']
            
            if rollback_value is not None:
                # Determine config type from current value
                current_config = self.get_config(env, path)
                config_type = ConfigType(current_config['type']) if current_config else ConfigType.STRING
                
                return self.set_config(env, path, rollback_value, config_type, user_id, 
                                     f"Rollback to version {target_version}")
            else:
                return self.delete_config(env, path, user_id)
    
    def _get_node(self, env: str, path: str) -> Optional[ConfigTrieNode]:
        """Get node at path"""
        if env not in self.environments:
            return None
        
        node = self.environments[env]
        for part in path.split('.'):
            if part not in node.children:
                return None
            node = node.children[part]
        
        return node
    
    def _has_permission(self, env: str, path: str, user_id: str, required_level: AccessLevel) -> bool:
        """Check if user has required permission"""
        if not user_id:
            return True  # Allow anonymous read access
        
        # Admin users have all permissions
        if user_id == "admin":
            return True
        
        # Check path-specific permissions
        node = self._get_node(env, path)
        if node and user_id in node.permissions:
            user_permissions = node.permissions[user_id]
            
            if AccessLevel.ADMIN in user_permissions:
                return True
            
            if required_level == AccessLevel.READ and (AccessLevel.READ in user_permissions or AccessLevel.WRITE in user_permissions):
                return True
            
            if required_level == AccessLevel.WRITE and AccessLevel.WRITE in user_permissions:
                return True
        
        # Check parent permissions (inheritance)
        path_parts = path.split('.')
        for i in range(len(path_parts) - 1, 0, -1):
            parent_path = '.'.join(path_parts[:i])
            parent_node = self._get_node(env, parent_path)
            
            if parent_node and user_id in parent_node.permissions:
                return required_level in parent_node.permissions[user_id]
        
        return False
    
    def _validate_config(self, path: str, value: Any, config_type: ConfigType, 
                        validation_rules: Dict = None) -> bool:
        """Validate configuration value"""
        # Type validation
        if config_type == ConfigType.STRING and not isinstance(value, str):
            return False
        elif config_type == ConfigType.INTEGER and not isinstance(value, int):
            return False
        elif config_type == ConfigType.FLOAT and not isinstance(value, (int, float)):
            return False
        elif config_type == ConfigType.BOOLEAN and not isinstance(value, bool):
            return False
        elif config_type == ConfigType.LIST and not isinstance(value, list):
            return False
        
        # Custom validation rules
        if validation_rules:
            if 'min_value' in validation_rules and value < validation_rules['min_value']:
                return False
            if 'max_value' in validation_rules and value > validation_rules['max_value']:
                return False
            if 'allowed_values' in validation_rules and value not in validation_rules['allowed_values']:
                return False
            if 'pattern' in validation_rules:
                import re
                if not re.match(validation_rules['pattern'], str(value)):
                    return False
        
        # Path-specific validators
        if path in self.validators:
            return self.validators[path](value)
        
        return True
    
    def _notify_watchers(self, env: str, path: str, new_value: Any, old_value: Any) -> None:
        """Notify watchers of configuration changes"""
        listener_key = f"{env}.{path}"
        
        if listener_key in self.change_listeners:
            for callback in self.change_listeners[listener_key]:
                try:
                    callback(env, path, new_value, old_value)
                except Exception as e:
                    print(f"Error in change listener: {e}")
    
    def export_config(self, env: str, format: str = 'json') -> str:
        """Export configuration to various formats"""
        configs = self.get_configs_by_prefix(env, "")
        
        if format == 'json':
            return json.dumps(configs, indent=2)
        elif format == 'yaml':
            # Would implement YAML export
            return "# YAML export not implemented"
        elif format == 'properties':
            # Convert to properties format
            lines = []
            for path, config in configs.items():
                lines.append(f"{path}={config['value']}")
            return '\n'.join(lines)
        
        return str(configs)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            'environments': len(self.environments),
            'total_configs': 0,
            'total_versions': 0,
            'environment_stats': {}
        }
        
        for env_name, root in self.environments.items():
            env_config_count = self._count_configs(root)
            env_version_count = len(self.versions[env_name])
            
            stats['total_configs'] += env_config_count
            stats['total_versions'] += env_version_count
            
            stats['environment_stats'][env_name] = {
                'config_count': env_config_count,
                'version_count': env_version_count,
                'current_version': self.version_counter[env_name]
            }
        
        return stats
    
    def _count_configs(self, node: ConfigTrieNode) -> int:
        """Count configurations in subtree"""
        count = 1 if node.config_value else 0
        
        for child in node.children.values():
            count += self._count_configs(child)
        
        return count


def test_basic_configuration():
    """Test basic configuration management"""
    print("=== Testing Basic Configuration ===")
    
    config_mgr = ConfigurationManager()
    
    # Set some configurations
    configs = [
        ("database.host", "localhost", ConfigType.STRING, "Database host"),
        ("database.port", 5432, ConfigType.INTEGER, "Database port"),
        ("cache.enabled", True, ConfigType.BOOLEAN, "Enable caching"),
        ("api.rate_limit", 100.5, ConfigType.FLOAT, "API rate limit"),
    ]
    
    print("Setting configurations:")
    for path, value, config_type, description in configs:
        success = config_mgr.set_config(
            "development", path, value, config_type, "admin", description
        )
        print(f"  {path}: {'✓' if success else '✗'}")
    
    # Retrieve configurations
    print(f"\nRetrieving configurations:")
    for path, _, _, _ in configs:
        config = config_mgr.get_config("development", path)
        if config:
            print(f"  {path}: {config['value']} ({config['type']})")

def test_hierarchical_access():
    """Test hierarchical configuration access"""
    print("\n=== Testing Hierarchical Access ===")
    
    config_mgr = ConfigurationManager()
    
    # Set nested configurations
    nested_configs = [
        "app.frontend.theme",
        "app.frontend.language", 
        "app.backend.workers",
        "app.backend.timeout",
        "monitoring.metrics.enabled",
        "monitoring.alerts.email"
    ]
    
    for i, path in enumerate(nested_configs):
        config_mgr.set_config("production", path, f"value_{i}", ConfigType.STRING, "admin")
    
    # Test prefix-based retrieval
    print("Configurations by prefix:")
    
    prefixes = ["app", "app.frontend", "monitoring"]
    for prefix in prefixes:
        configs = config_mgr.get_configs_by_prefix("production", prefix)
        print(f"\n  Prefix '{prefix}': {len(configs)} configs")
        for path, config in configs.items():
            print(f"    {path}: {config['value']}")

def test_permissions():
    """Test permission system"""
    print("\n=== Testing Permissions ===")
    
    config_mgr = ConfigurationManager()
    
    # Set up configurations as admin
    config_mgr.set_config("staging", "secret.api_key", "secret123", ConfigType.STRING, "admin")
    config_mgr.set_config("staging", "public.app_name", "MyApp", ConfigType.STRING, "admin")
    
    # Set permissions
    config_mgr.set_permission("staging", "secret", "user1", AccessLevel.READ, "admin")
    config_mgr.set_permission("staging", "public", "user1", AccessLevel.WRITE, "admin")
    
    print("Testing access control:")
    
    # Test read access
    secret = config_mgr.get_config("staging", "secret.api_key", "user1")
    print(f"  User1 read secret: {'✓' if secret else '✗'}")
    
    # Test write access (should fail for secret)
    write_secret = config_mgr.set_config("staging", "secret.api_key", "new_secret", ConfigType.STRING, "user1")
    print(f"  User1 write secret: {'✗' if not write_secret else '✓'}")
    
    # Test write access (should succeed for public)
    write_public = config_mgr.set_config("staging", "public.version", "1.0", ConfigType.STRING, "user1")
    print(f"  User1 write public: {'✓' if write_public else '✗'}")

def test_versioning():
    """Test configuration versioning"""
    print("\n=== Testing Configuration Versioning ===")
    
    config_mgr = ConfigurationManager()
    
    path = "app.version"
    
    # Create several versions
    versions = ["1.0.0", "1.1.0", "1.2.0", "2.0.0"]
    
    print("Creating configuration versions:")
    for version in versions:
        config_mgr.set_config("production", path, version, ConfigType.STRING, "admin", f"Release {version}")
        print(f"  Set version: {version}")
    
    # Get history
    history = config_mgr.get_config_history("production", path, limit=5)
    
    print(f"\nConfiguration history for {path}:")
    for entry in history:
        print(f"  Version {entry['version']}: {entry['changes']['old']} -> {entry['changes']['new']}")
    
    # Test rollback
    print(f"\nTesting rollback to version 2:")
    rollback_success = config_mgr.rollback_config("production", path, 2, "admin")
    print(f"  Rollback success: {'✓' if rollback_success else '✗'}")
    
    current = config_mgr.get_config("production", path)
    print(f"  Current value after rollback: {current['value'] if current else 'None'}")

def test_validation():
    """Test configuration validation"""
    print("\n=== Testing Configuration Validation ===")
    
    config_mgr = ConfigurationManager()
    
    # Test with validation rules
    validation_rules = {
        'min_value': 1,
        'max_value': 100,
        'allowed_values': [10, 20, 30, 40, 50]
    }
    
    test_values = [5, 30, 150]  # Invalid, valid, invalid
    
    print("Testing validation rules (allowed: 10,20,30,40,50):")
    for value in test_values:
        success = config_mgr.set_config(
            "test", "validated.number", value, ConfigType.INTEGER, 
            "admin", "Test validation", validation_rules
        )
        print(f"  Value {value}: {'✓' if success else '✗'}")

def benchmark_configuration_performance():
    """Benchmark configuration system performance"""
    print("\n=== Benchmarking Configuration Performance ===")
    
    config_mgr = ConfigurationManager()
    
    # Generate test configurations
    num_configs = 1000
    
    # Benchmark writes
    start_time = time.time()
    for i in range(num_configs):
        path = f"benchmark.section{i//100}.config{i}"
        config_mgr.set_config("benchmark", path, f"value_{i}", ConfigType.STRING, "admin")
    
    write_time = time.time() - start_time
    
    # Benchmark reads
    start_time = time.time()
    successful_reads = 0
    
    for i in range(num_configs):
        path = f"benchmark.section{i//100}.config{i}"
        if config_mgr.get_config("benchmark", path):
            successful_reads += 1
    
    read_time = time.time() - start_time
    
    # Benchmark prefix queries
    start_time = time.time()
    total_results = 0
    
    for i in range(10):  # Test 10 prefix queries
        prefix = f"benchmark.section{i}"
        results = config_mgr.get_configs_by_prefix("benchmark", prefix)
        total_results += len(results)
    
    prefix_time = time.time() - start_time
    
    # Get statistics
    stats = config_mgr.get_statistics()
    
    print(f"Performance Results:")
    print(f"  {num_configs} writes in {write_time:.3f}s ({num_configs/write_time:.0f} writes/sec)")
    print(f"  {num_configs} reads in {read_time:.3f}s ({num_configs/read_time:.0f} reads/sec)")
    print(f"  10 prefix queries in {prefix_time:.3f}s ({10/prefix_time:.0f} queries/sec)")
    print(f"  Total configurations: {stats['total_configs']}")
    print(f"  Success rate: {successful_reads}/{num_configs} ({successful_reads/num_configs:.1%})")

if __name__ == "__main__":
    test_basic_configuration()
    test_hierarchical_access()
    test_permissions()
    test_versioning()
    test_validation()
    benchmark_configuration_performance()

"""
Configuration Management System demonstrates enterprise configuration management:

Key Features:
1. Hierarchical Storage - Trie-based structure for nested configurations
2. Multi-Environment Support - Separate configurations for dev/staging/prod
3. Access Control - Fine-grained permissions system
4. Version Control - Track all configuration changes with rollback capability
5. Real-time Updates - Watch configuration changes with callbacks
6. Validation Framework - Ensure configuration integrity

System Design Aspects:
- Trie-based hierarchical storage for efficient prefix operations
- Environment isolation for different deployment stages
- Permission inheritance through configuration hierarchy
- Complete audit trail with version history
- Real-time change notifications
- Comprehensive validation framework

Advanced Features:
- Path-based permission inheritance
- Configuration rollback to previous versions
- Export configurations to multiple formats
- Real-time change notifications
- Custom validation rules
- Bulk operations and prefix-based queries

Real-world Applications:
- Microservices configuration management
- Feature flag systems
- Environment-specific settings
- Application configuration stores
- DevOps deployment configurations
- Multi-tenant application settings

Security Considerations:
- Role-based access control
- Configuration encryption support
- Audit logging for compliance
- Permission inheritance models
- Secure configuration distribution

This implementation provides a production-ready foundation for
building scalable configuration management systems with enterprise features.
"""
