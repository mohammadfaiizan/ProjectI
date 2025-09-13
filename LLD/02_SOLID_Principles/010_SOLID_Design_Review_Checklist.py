"""
SOLID DESIGN REVIEW CHECKLIST - Code Quality Assessment
========================================================

Problem Statement:
Provide comprehensive checklist and tools for reviewing SOLID compliance:
- Automated SOLID principle violation detection
- Code quality metrics and analysis
- Design review guidelines and best practices
- Refactoring recommendations
- SOLID compliance scoring system

Learning Objectives:
- Create systematic SOLID review processes
- Identify violations through automated analysis
- Provide actionable refactoring recommendations
- Establish SOLID compliance metrics
- Build quality gates for SOLID principles
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from datetime import datetime
from enum import Enum
import inspect
import ast
import re


# ============================================================================
# VIOLATION SEVERITY AND TYPES
# ============================================================================

class ViolationSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SOLIDPrinciple(Enum):
    SRP = "Single Responsibility Principle"
    OCP = "Open/Closed Principle"
    LSP = "Liskov Substitution Principle"
    ISP = "Interface Segregation Principle"
    DIP = "Dependency Inversion Principle"


class ViolationType(Enum):
    # SRP Violations
    GOD_CLASS = "god_class"
    MULTIPLE_RESPONSIBILITIES = "multiple_responsibilities"
    HIGH_COUPLING = "high_coupling"
    
    # OCP Violations
    MODIFICATION_FOR_EXTENSION = "modification_for_extension"
    SWITCH_STATEMENTS = "switch_statements"
    INSTANCEOF_CHECKS = "instanceof_checks"
    
    # LSP Violations
    BEHAVIORAL_INCOMPATIBILITY = "behavioral_incompatibility"
    STRENGTHENED_PRECONDITIONS = "strengthened_preconditions"
    WEAKENED_POSTCONDITIONS = "weakened_postconditions"
    
    # ISP Violations
    FAT_INTERFACE = "fat_interface"
    UNUSED_METHODS = "unused_methods"
    INTERFACE_POLLUTION = "interface_pollution"
    
    # DIP Violations
    CONCRETE_DEPENDENCIES = "concrete_dependencies"
    HIGH_LEVEL_DEPENDS_LOW_LEVEL = "high_level_depends_low_level"
    NO_ABSTRACTION = "no_abstraction"


# ============================================================================
# VIOLATION DETECTION CLASSES
# ============================================================================

class SOLIDViolation:
    """Represents a SOLID principle violation."""
    
    def __init__(self, principle: SOLIDPrinciple, violation_type: ViolationType,
                 severity: ViolationSeverity, description: str, location: str,
                 recommendation: str, code_example: Optional[str] = None):
        self.principle = principle
        self.violation_type = violation_type
        self.severity = severity
        self.description = description
        self.location = location
        self.recommendation = recommendation
        self.code_example = code_example
        self.detected_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary."""
        return {
            'principle': self.principle.value,
            'violation_type': self.violation_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'location': self.location,
            'recommendation': self.recommendation,
            'code_example': self.code_example,
            'detected_at': self.detected_at.isoformat()
        }


class ClassAnalyzer:
    """Analyzes classes for SOLID violations."""
    
    def __init__(self, cls: type):
        self.cls = cls
        self.methods = [method for method in dir(cls) if callable(getattr(cls, method))]
        self.public_methods = [m for m in self.methods if not m.startswith('_')]
        self.private_methods = [m for m in self.methods if m.startswith('_') and not m.startswith('__')]
        self.magic_methods = [m for m in self.methods if m.startswith('__') and m.endswith('__')]
    
    def analyze_srp_violations(self) -> List[SOLIDViolation]:
        """Analyze Single Responsibility Principle violations."""
        violations = []
        
        # Check for God Class (too many methods)
        if len(self.public_methods) > 15:
            violations.append(SOLIDViolation(
                SOLIDPrinciple.SRP,
                ViolationType.GOD_CLASS,
                ViolationSeverity.HIGH,
                f"Class {self.cls.__name__} has {len(self.public_methods)} public methods, indicating multiple responsibilities",
                f"Class: {self.cls.__name__}",
                "Consider splitting this class into smaller, focused classes with single responsibilities"
            ))
        
        # Check for multiple responsibilities based on method naming patterns
        responsibility_patterns = {
            'data_access': ['save', 'load', 'find', 'delete', 'create', 'update'],
            'validation': ['validate', 'check', 'verify', 'is_valid'],
            'formatting': ['format', 'parse', 'convert', 'transform'],
            'notification': ['send', 'notify', 'alert', 'email'],
            'logging': ['log', 'debug', 'info', 'error', 'warn'],
            'calculation': ['calculate', 'compute', 'sum', 'total']
        }
        
        found_responsibilities = []
        for responsibility, patterns in responsibility_patterns.items():
            if any(any(pattern in method.lower() for pattern in patterns) for method in self.public_methods):
                found_responsibilities.append(responsibility)
        
        if len(found_responsibilities) > 2:
            violations.append(SOLIDViolation(
                SOLIDPrinciple.SRP,
                ViolationType.MULTIPLE_RESPONSIBILITIES,
                ViolationSeverity.MEDIUM,
                f"Class {self.cls.__name__} appears to have multiple responsibilities: {', '.join(found_responsibilities)}",
                f"Class: {self.cls.__name__}",
                "Extract separate classes for each responsibility (e.g., Repository, Validator, Formatter)"
            ))
        
        return violations
    
    def analyze_isp_violations(self) -> List[SOLIDViolation]:
        """Analyze Interface Segregation Principle violations."""
        violations = []
        
        # Check for fat interfaces (too many abstract methods)
        if inspect.isabstract(self.cls):
            abstract_methods = [method for method in self.public_methods 
                              if hasattr(getattr(self.cls, method, None), '__isabstractmethod__')]
            
            if len(abstract_methods) > 10:
                violations.append(SOLIDViolation(
                    SOLIDPrinciple.ISP,
                    ViolationType.FAT_INTERFACE,
                    ViolationSeverity.HIGH,
                    f"Interface {self.cls.__name__} has {len(abstract_methods)} abstract methods, which may be too many",
                    f"Interface: {self.cls.__name__}",
                    "Consider splitting this interface into smaller, more focused interfaces"
                ))
        
        return violations
    
    def analyze_dip_violations(self) -> List[SOLIDViolation]:
        """Analyze Dependency Inversion Principle violations."""
        violations = []
        
        # Check constructor for concrete dependencies
        if hasattr(self.cls, '__init__'):
            init_method = getattr(self.cls, '__init__')
            if hasattr(init_method, '__annotations__'):
                annotations = init_method.__annotations__
                concrete_deps = []
                
                for param_name, param_type in annotations.items():
                    if param_name != 'return' and hasattr(param_type, '__name__'):
                        # Check if it's a concrete class (not ABC or Protocol)
                        if (not inspect.isabstract(param_type) and 
                            not hasattr(param_type, '_is_protocol')):
                            concrete_deps.append(param_name)
                
                if concrete_deps:
                    violations.append(SOLIDViolation(
                        SOLIDPrinciple.DIP,
                        ViolationType.CONCRETE_DEPENDENCIES,
                        ViolationSeverity.MEDIUM,
                        f"Class {self.cls.__name__} depends on concrete classes: {', '.join(concrete_deps)}",
                        f"Class: {self.cls.__name__}.__init__",
                        "Depend on abstractions (interfaces/protocols) instead of concrete implementations"
                    ))
        
        return violations


class CodeAnalyzer:
    """Analyzes code for SOLID principle violations."""
    
    def __init__(self):
        self.violations: List[SOLIDViolation] = []
        self.analyzed_classes: List[type] = []
    
    def analyze_class(self, cls: type) -> List[SOLIDViolation]:
        """Analyze a single class for SOLID violations."""
        analyzer = ClassAnalyzer(cls)
        violations = []
        
        violations.extend(analyzer.analyze_srp_violations())
        violations.extend(analyzer.analyze_isp_violations())
        violations.extend(analyzer.analyze_dip_violations())
        
        self.analyzed_classes.append(cls)
        self.violations.extend(violations)
        
        return violations
    
    def analyze_module(self, module) -> List[SOLIDViolation]:
        """Analyze all classes in a module."""
        violations = []
        
        for name in dir(module):
            obj = getattr(module, name)
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                violations.extend(self.analyze_class(obj))
        
        return violations
    
    def get_violations_by_principle(self, principle: SOLIDPrinciple) -> List[SOLIDViolation]:
        """Get violations for a specific principle."""
        return [v for v in self.violations if v.principle == principle]
    
    def get_violations_by_severity(self, severity: ViolationSeverity) -> List[SOLIDViolation]:
        """Get violations by severity level."""
        return [v for v in self.violations if v.severity == severity]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        total_violations = len(self.violations)
        
        violations_by_principle = {}
        for principle in SOLIDPrinciple:
            violations_by_principle[principle.value] = len(self.get_violations_by_principle(principle))
        
        violations_by_severity = {}
        for severity in ViolationSeverity:
            violations_by_severity[severity.value] = len(self.get_violations_by_severity(severity))
        
        # Calculate SOLID compliance score (0-100)
        max_possible_violations = len(self.analyzed_classes) * 10  # Assume max 10 violations per class
        compliance_score = max(0, 100 - (total_violations / max(max_possible_violations, 1)) * 100)
        
        return {
            'analysis_date': datetime.now().isoformat(),
            'classes_analyzed': len(self.analyzed_classes),
            'total_violations': total_violations,
            'compliance_score': round(compliance_score, 2),
            'violations_by_principle': violations_by_principle,
            'violations_by_severity': violations_by_severity,
            'violations': [v.to_dict() for v in self.violations]
        }


# ============================================================================
# DESIGN REVIEW CHECKLIST
# ============================================================================

class DesignReviewChecklist:
    """Comprehensive SOLID design review checklist."""
    
    def __init__(self):
        self.checklist_items = self._build_checklist()
    
    def _build_checklist(self) -> Dict[SOLIDPrinciple, List[Dict[str, Any]]]:
        """Build comprehensive SOLID review checklist."""
        return {
            SOLIDPrinciple.SRP: [
                {
                    'id': 'srp_001',
                    'question': 'Does each class have only one reason to change?',
                    'description': 'Classes should have a single responsibility and only one reason to be modified',
                    'check_points': [
                        'Class name clearly indicates its single purpose',
                        'All methods are related to the same responsibility',
                        'Class has fewer than 15 public methods',
                        'No mixed concerns (e.g., business logic + data access)'
                    ]
                },
                {
                    'id': 'srp_002',
                    'question': 'Are responsibilities properly separated?',
                    'description': 'Different concerns should be in different classes',
                    'check_points': [
                        'Data access logic is separated from business logic',
                        'Validation logic is in dedicated classes',
                        'Formatting/presentation logic is separate',
                        'Logging and auditing are handled by dedicated services'
                    ]
                },
                {
                    'id': 'srp_003',
                    'question': 'Is the class cohesive?',
                    'description': 'All parts of the class should work together toward the same goal',
                    'check_points': [
                        'Methods use the same instance variables',
                        'Methods collaborate to fulfill the class purpose',
                        'No unrelated utility methods',
                        'Clear relationship between all class members'
                    ]
                }
            ],
            
            SOLIDPrinciple.OCP: [
                {
                    'id': 'ocp_001',
                    'question': 'Can new functionality be added without modifying existing code?',
                    'description': 'System should be open for extension but closed for modification',
                    'check_points': [
                        'Strategy pattern used for varying algorithms',
                        'Plugin architecture for extensibility',
                        'Abstract base classes for extension points',
                        'No switch statements on type codes'
                    ]
                },
                {
                    'id': 'ocp_002',
                    'question': 'Are abstractions used for extension points?',
                    'description': 'Extension points should be defined through abstractions',
                    'check_points': [
                        'Interfaces define extension contracts',
                        'Abstract classes provide extension points',
                        'Factory patterns for object creation',
                        'Configuration-driven behavior'
                    ]
                },
                {
                    'id': 'ocp_003',
                    'question': 'Is the design stable against changes?',
                    'description': 'Core abstractions should remain stable',
                    'check_points': [
                        'Well-defined interfaces that rarely change',
                        'Stable abstractions with varying implementations',
                        'Dependency direction follows stability',
                        'Changes are additive, not modificative'
                    ]
                }
            ],
            
            SOLIDPrinciple.LSP: [
                {
                    'id': 'lsp_001',
                    'question': 'Are subclasses substitutable for their base classes?',
                    'description': 'Derived classes must be substitutable for their base classes',
                    'check_points': [
                        'Subclasses honor the contract of base classes',
                        'No strengthened preconditions in subclasses',
                        'No weakened postconditions in subclasses',
                        'Behavioral compatibility maintained'
                    ]
                },
                {
                    'id': 'lsp_002',
                    'question': 'Do inheritance hierarchies make logical sense?',
                    'description': 'Inheritance should represent true "is-a" relationships',
                    'check_points': [
                        'Inheritance represents logical relationships',
                        'No inappropriate inheritance (Square from Rectangle)',
                        'Polymorphism works correctly',
                        'No empty or exception-throwing overrides'
                    ]
                },
                {
                    'id': 'lsp_003',
                    'question': 'Are invariants preserved in subclasses?',
                    'description': 'Class invariants must be maintained by all subclasses',
                    'check_points': [
                        'Object state remains consistent',
                        'Class invariants never violated',
                        'History constraints respected',
                        'No unexpected side effects in overrides'
                    ]
                }
            ],
            
            SOLIDPrinciple.ISP: [
                {
                    'id': 'isp_001',
                    'question': 'Are interfaces focused and cohesive?',
                    'description': 'Interfaces should be small and focused on specific client needs',
                    'check_points': [
                        'Interfaces have fewer than 10 methods',
                        'All methods in interface are related',
                        'Clients use most/all methods of interfaces they depend on',
                        'No fat interfaces with unrelated methods'
                    ]
                },
                {
                    'id': 'isp_002',
                    'question': 'Do clients depend only on methods they use?',
                    'description': 'Clients should not be forced to depend on unused methods',
                    'check_points': [
                        'No empty method implementations',
                        'No NotImplementedError exceptions',
                        'Interfaces segregated by client needs',
                        'Role-based interface design'
                    ]
                },
                {
                    'id': 'isp_003',
                    'question': 'Are interfaces designed from client perspective?',
                    'description': 'Interfaces should be designed based on how clients use them',
                    'check_points': [
                        'Interface methods match client usage patterns',
                        'Granularity appropriate for client needs',
                        'No god interfaces',
                        'Clear separation of concerns in interfaces'
                    ]
                }
            ],
            
            SOLIDPrinciple.DIP: [
                {
                    'id': 'dip_001',
                    'question': 'Do high-level modules depend on abstractions?',
                    'description': 'High-level modules should not depend on low-level modules',
                    'check_points': [
                        'Dependencies are injected through constructors',
                        'High-level classes depend on interfaces',
                        'No direct instantiation of concrete dependencies',
                        'Dependency direction follows abstraction level'
                    ]
                },
                {
                    'id': 'dip_002',
                    'question': 'Are abstractions independent of details?',
                    'description': 'Abstractions should not depend on implementation details',
                    'check_points': [
                        'Interfaces are implementation-agnostic',
                        'Abstract classes dont expose implementation details',
                        'Stable abstractions with varying implementations',
                        'Details depend on abstractions, not vice versa'
                    ]
                },
                {
                    'id': 'dip_003',
                    'question': 'Is dependency injection used effectively?',
                    'description': 'Dependencies should be injected rather than created',
                    'check_points': [
                        'Constructor injection for required dependencies',
                        'Setter injection for optional dependencies',
                        'IoC container for complex dependency graphs',
                        'No service locator anti-pattern'
                    ]
                }
            ]
        }
    
    def generate_review_template(self) -> str:
        """Generate a review template for SOLID compliance."""
        template = "# SOLID Design Review Checklist\n\n"
        template += f"**Review Date:** {datetime.now().strftime('%Y-%m-%d')}\n"
        template += "**Reviewer:** [Name]\n"
        template += "**Component:** [Component Name]\n\n"
        
        for principle, items in self.checklist_items.items():
            template += f"## {principle.value}\n\n"
            
            for item in items:
                template += f"### {item['id'].upper()}: {item['question']}\n"
                template += f"**Description:** {item['description']}\n\n"
                template += "**Check Points:**\n"
                
                for check_point in item['check_points']:
                    template += f"- [ ] {check_point}\n"
                
                template += "\n**Notes:**\n[Add review notes here]\n\n"
                template += "**Status:** [ ] Pass [ ] Fail [ ] Needs Improvement\n\n"
                template += "---\n\n"
        
        template += "## Overall Assessment\n\n"
        template += "**SOLID Compliance Score:** [0-100]\n\n"
        template += "**Major Issues:**\n- [List major violations]\n\n"
        template += "**Recommendations:**\n- [List improvement recommendations]\n\n"
        template += "**Action Items:**\n- [ ] [Action item 1]\n- [ ] [Action item 2]\n\n"
        
        return template


# ============================================================================
# EXAMPLE CLASSES FOR TESTING
# ============================================================================

# Good example - SOLID compliant
class GoodUserService:
    """Well-designed service following SOLID principles."""
    
    def __init__(self, user_repository: 'UserRepository', 
                 notification_service: 'NotificationService',
                 logger: 'Logger'):
        self.user_repository = user_repository
        self.notification_service = notification_service
        self.logger = logger
    
    def create_user(self, username: str, email: str) -> bool:
        """Create new user."""
        self.logger.log("INFO", f"Creating user: {username}")
        # Implementation here
        return True
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        return self.user_repository.find_by_id(user_id)


# Bad example - SOLID violations
class BadUserManager:
    """Poorly designed class with multiple SOLID violations."""
    
    def __init__(self):
        # DIP violation - concrete dependencies
        self.database_connection = "sqlite:///users.db"
        self.email_client = "smtp.gmail.com"
        self.users = {}
    
    # SRP violation - too many responsibilities
    def create_user(self, username: str, email: str, password: str) -> bool:
        """Creates user - doing too many things."""
        # Validation
        if not self.validate_email(email):
            return False
        
        # Password hashing
        hashed_password = self.hash_password(password)
        
        # Database operations
        self.save_to_database({'username': username, 'email': email, 'password': hashed_password})
        
        # Email sending
        self.send_welcome_email(email, username)
        
        # Logging
        self.log_user_creation(username)
        
        return True
    
    def validate_email(self, email: str) -> bool:
        return "@" in email
    
    def hash_password(self, password: str) -> str:
        return f"hashed_{password}"
    
    def save_to_database(self, user_data: dict) -> None:
        print(f"Saving to {self.database_connection}")
    
    def send_welcome_email(self, email: str, username: str) -> None:
        print(f"Sending email via {self.email_client}")
    
    def log_user_creation(self, username: str) -> None:
        print(f"User created: {username}")
    
    # More methods making this a god class...
    def authenticate_user(self, username: str, password: str) -> bool:
        return True
    
    def update_user_profile(self, user_id: str, profile_data: dict) -> bool:
        return True
    
    def delete_user(self, user_id: str) -> bool:
        return True
    
    def send_password_reset_email(self, email: str) -> bool:
        return True
    
    def validate_password_strength(self, password: str) -> bool:
        return True
    
    def generate_user_report(self, user_id: str) -> str:
        return "Report"
    
    def backup_user_data(self) -> bool:
        return True
    
    def import_users_from_csv(self, csv_file: str) -> bool:
        return True
    
    def export_users_to_json(self) -> str:
        return "{}"
    
    def calculate_user_statistics(self) -> dict:
        return {}


# Fat interface example - ISP violation
class BadWorkerInterface(ABC):
    """Fat interface violating ISP."""
    
    @abstractmethod
    def work(self) -> str:
        pass
    
    @abstractmethod
    def eat(self) -> str:
        pass
    
    @abstractmethod
    def sleep(self) -> str:
        pass
    
    @abstractmethod
    def program(self) -> str:
        pass
    
    @abstractmethod
    def design(self) -> str:
        pass
    
    @abstractmethod
    def manage_team(self) -> str:
        pass
    
    @abstractmethod
    def write_documentation(self) -> str:
        pass
    
    @abstractmethod
    def test_software(self) -> str:
        pass
    
    @abstractmethod
    def deploy_application(self) -> str:
        pass
    
    @abstractmethod
    def monitor_systems(self) -> str:
        pass


def demonstrate_solid_design_review():
    """
    Demonstrate SOLID design review checklist and automated analysis.
    """
    print("=== SOLID DESIGN REVIEW CHECKLIST DEMONSTRATION ===\n")
    
    # 1. Automated Code Analysis
    print("1. AUTOMATED SOLID VIOLATION DETECTION:")
    
    analyzer = CodeAnalyzer()
    
    # Analyze good and bad classes
    print("   Analyzing well-designed class...")
    good_violations = analyzer.analyze_class(GoodUserService)
    print(f"   ✓ GoodUserService violations: {len(good_violations)}")
    
    print("   Analyzing poorly-designed class...")
    bad_violations = analyzer.analyze_class(BadUserManager)
    print(f"   ✗ BadUserManager violations: {len(bad_violations)}")
    
    print("   Analyzing fat interface...")
    interface_violations = analyzer.analyze_class(BadWorkerInterface)
    print(f"   ✗ BadWorkerInterface violations: {len(interface_violations)}")
    
    print()
    
    # 2. Generate Analysis Report
    print("2. COMPREHENSIVE ANALYSIS REPORT:")
    
    report = analyzer.generate_report()
    
    print(f"   Classes Analyzed: {report['classes_analyzed']}")
    print(f"   Total Violations: {report['total_violations']}")
    print(f"   SOLID Compliance Score: {report['compliance_score']}/100")
    
    print("\n   Violations by Principle:")
    for principle, count in report['violations_by_principle'].items():
        print(f"     {principle}: {count} violations")
    
    print("\n   Violations by Severity:")
    for severity, count in report['violations_by_severity'].items():
        print(f"     {severity.title()}: {count} violations")
    
    print()
    
    # 3. Detailed Violation Analysis
    print("3. DETAILED VIOLATION ANALYSIS:")
    
    for violation in analyzer.violations[:5]:  # Show first 5 violations
        print(f"\n   {violation.principle.value} Violation:")
        print(f"     Type: {violation.violation_type.value}")
        print(f"     Severity: {violation.severity.value}")
        print(f"     Location: {violation.location}")
        print(f"     Description: {violation.description}")
        print(f"     Recommendation: {violation.recommendation}")
    
    if len(analyzer.violations) > 5:
        print(f"\n   ... and {len(analyzer.violations) - 5} more violations")
    
    print()
    
    # 4. Design Review Checklist
    print("4. DESIGN REVIEW CHECKLIST:")
    
    checklist = DesignReviewChecklist()
    
    print("   Available checklist categories:")
    for principle in SOLIDPrinciple:
        items_count = len(checklist.checklist_items[principle])
        print(f"     {principle.value}: {items_count} check items")
    
    print("\n   Sample checklist items for SRP:")
    srp_items = checklist.checklist_items[SOLIDPrinciple.SRP]
    for item in srp_items[:2]:  # Show first 2 items
        print(f"     {item['id']}: {item['question']}")
        print(f"       {item['description']}")
        print(f"       Check points: {len(item['check_points'])}")
    
    print()
    
    # 5. Review Template Generation
    print("5. REVIEW TEMPLATE GENERATION:")
    
    template = checklist.generate_review_template()
    template_lines = template.split('\n')
    
    print("   Generated review template preview:")
    print("   " + "\n   ".join(template_lines[:20]))  # Show first 20 lines
    print(f"   ... (total {len(template_lines)} lines)")
    
    print()
    
    # 6. Quality Gates and Metrics
    print("6. SOLID QUALITY GATES AND METRICS:")
    
    def evaluate_quality_gates(compliance_score: float, violations: List[SOLIDViolation]) -> Dict[str, Any]:
        """Evaluate quality gates based on SOLID compliance."""
        
        critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        high_violations = [v for v in violations if v.severity == ViolationSeverity.HIGH]
        
        gates = {
            'minimum_compliance_score': {
                'threshold': 70.0,
                'current': compliance_score,
                'passed': compliance_score >= 70.0
            },
            'no_critical_violations': {
                'threshold': 0,
                'current': len(critical_violations),
                'passed': len(critical_violations) == 0
            },
            'max_high_violations': {
                'threshold': 3,
                'current': len(high_violations),
                'passed': len(high_violations) <= 3
            },
            'max_total_violations': {
                'threshold': 10,
                'current': len(violations),
                'passed': len(violations) <= 10
            }
        }
        
        overall_passed = all(gate['passed'] for gate in gates.values())
        
        return {
            'overall_passed': overall_passed,
            'gates': gates
        }
    
    quality_gates = evaluate_quality_gates(report['compliance_score'], analyzer.violations)
    
    print(f"   Overall Quality Gate: {'PASSED' if quality_gates['overall_passed'] else 'FAILED'}")
    print("\n   Individual Gates:")
    
    for gate_name, gate_info in quality_gates['gates'].items():
        status = "PASS" if gate_info['passed'] else "FAIL"
        print(f"     {gate_name}: {status}")
        print(f"       Threshold: {gate_info['threshold']}")
        print(f"       Current: {gate_info['current']}")
    
    print()
    
    # 7. Refactoring Recommendations
    print("7. REFACTORING RECOMMENDATIONS:")
    
    def generate_refactoring_plan(violations: List[SOLIDViolation]) -> Dict[str, List[str]]:
        """Generate refactoring recommendations based on violations."""
        
        recommendations = {
            'immediate_actions': [],
            'short_term_improvements': [],
            'long_term_refactoring': []
        }
        
        # Critical and high severity violations need immediate attention
        critical_high = [v for v in violations if v.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH]]
        for violation in critical_high:
            recommendations['immediate_actions'].append(
                f"{violation.location}: {violation.recommendation}"
            )
        
        # Medium severity violations for short-term
        medium = [v for v in violations if v.severity == ViolationSeverity.MEDIUM]
        for violation in medium[:3]:  # Top 3
            recommendations['short_term_improvements'].append(
                f"{violation.location}: {violation.recommendation}"
            )
        
        # Strategic recommendations
        if len([v for v in violations if v.principle == SOLIDPrinciple.SRP]) > 3:
            recommendations['long_term_refactoring'].append(
                "Consider implementing Domain-Driven Design to better separate responsibilities"
            )
        
        if len([v for v in violations if v.principle == SOLIDPrinciple.DIP]) > 2:
            recommendations['long_term_refactoring'].append(
                "Implement IoC container for better dependency management"
            )
        
        return recommendations
    
    refactoring_plan = generate_refactoring_plan(analyzer.violations)
    
    for category, actions in refactoring_plan.items():
        if actions:
            print(f"   {category.replace('_', ' ').title()}:")
            for action in actions:
                print(f"     • {action}")
            print()
    
    # 8. Best Practices Summary
    print("8. SOLID REVIEW BEST PRACTICES:")
    print("   Process Guidelines:")
    print("   ✓ Run automated analysis before manual review")
    print("   ✓ Use checklist systematically for each principle")
    print("   ✓ Focus on high-severity violations first")
    print("   ✓ Consider business impact of refactoring")
    print("   ✓ Plan refactoring in manageable increments")
    print("   ✓ Establish quality gates for continuous compliance")
    print("   ✓ Review SOLID compliance in code reviews")
    print("   ✓ Track compliance metrics over time")
    print()
    
    print("   Review Frequency:")
    print("   • Daily: Automated violation detection in CI/CD")
    print("   • Weekly: Team review of new violations")
    print("   • Monthly: Comprehensive SOLID compliance review")
    print("   • Quarterly: Architecture review and refactoring planning")
    print()
    
    print("=== SOLID DESIGN REVIEW CHECKLIST DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_solid_design_review()
