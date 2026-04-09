"""
RBAC AND ABAC POLICY ENGINES
================================

Problem Statement:
Complex authorization systems need flexible, auditable, and maintainable
access control. RBAC handles most SaaS use cases; ABAC handles complex
context-aware policies (healthcare, finance, government).

RBAC (Role-Based Access Control) — RFC 2903:
  Core RBAC:      User → Roles → Permissions.
  Hierarchical:   Roles inherit from parent roles (admin → editor → viewer).
  Constrained:    Separation of duty: user can't hold conflicting roles.
  Best for: SaaS, enterprise apps with well-defined job functions.

ABAC (Attribute-Based Access Control) — NIST SP 800-162:
  Policy: IF [conditions on attributes] THEN [effect].
  Subject attributes: user.department, user.clearance.
  Resource attributes: resource.classification, resource.owner.
  Environment attributes: time.hour, request.ip_region.
  Action: read, write, delete, execute.
  Best for: fine-grained, context-aware policies.

XACML (eXtensible Access Control Markup Language):
  Standard for ABAC policies. Verbose XML.
  PEP (Policy Enforcement Point): intercepts requests.
  PDP (Policy Decision Point): evaluates policies.
  PAP (Policy Administration Point): manages policies.
  PIP (Policy Information Point): retrieves attributes.

ReBAC (Relationship-Based Access Control) — Google Zanzibar:
  Access based on relationship graph.
  "User can read doc IF user is owner OR user is member of group that is editor."
  Scales to billions of users/objects.
  Powers: Google Drive, YouTube, Google Cloud IAM.

Google Zanzibar Concepts:
  Namespace: type of object (docs, folders, channels).
  Object: specific instance (doc:readme).
  Relation: edge type (owner, editor, viewer, member).
  Userset: set of users satisfying a relation.
  Tuple: (object, relation, user) → "doc:readme#editor@user:alice"
  Check: Can user:alice perform action:write on doc:readme?
         → check (doc:readme, writer, user:alice) or inherited.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from enum import Enum
import time


# ─────────────────────────────────────────────
# RBAC ENGINE (with role hierarchy)
# ─────────────────────────────────────────────

class Permission(Enum):
    READ    = "read"
    WRITE   = "write"
    DELETE  = "delete"
    ADMIN   = "admin"
    PUBLISH = "publish"
    REVIEW  = "review"


@dataclass
class Role:
    name       : str
    permissions: Set[Permission]
    parent     : Optional[str] = None   # for hierarchical RBAC


class HierarchicalRBAC:
    """
    Hierarchical RBAC: child roles inherit parent permissions.
    Supports separation of duty constraints.
    """

    def __init__(self):
        self._roles        : Dict[str, Role]      = {}
        self._user_roles   : Dict[str, Set[str]]  = {}
        self._sod_rules    : List[Tuple[str, str]] = []  # mutually exclusive role pairs

    def add_role(self, name: str, permissions: Set[Permission],
                  parent: str = None):
        self._roles[name] = Role(name=name, permissions=permissions, parent=parent)

    def assign_role(self, user_id: str, role_name: str):
        if user_id not in self._user_roles:
            self._user_roles[user_id] = set()

        # Check SoD constraints
        for (r1, r2) in self._sod_rules:
            if role_name == r1 and r2 in self._user_roles[user_id]:
                raise ValueError(f"SoD violation: cannot have both {r1} and {r2}")
            if role_name == r2 and r1 in self._user_roles[user_id]:
                raise ValueError(f"SoD violation: cannot have both {r1} and {r2}")

        self._user_roles[user_id].add(role_name)

    def revoke_role(self, user_id: str, role_name: str):
        self._user_roles.get(user_id, set()).discard(role_name)

    def add_sod_rule(self, role1: str, role2: str):
        """Separation of Duty: user cannot hold both roles."""
        self._sod_rules.append((role1, role2))

    def _inherited_permissions(self, role_name: str,
                                visited: Set[str] = None) -> Set[Permission]:
        if visited is None:
            visited = set()
        if role_name in visited or role_name not in self._roles:
            return set()
        visited.add(role_name)
        role = self._roles[role_name]
        perms = set(role.permissions)
        if role.parent:
            perms.update(self._inherited_permissions(role.parent, visited))
        return perms

    def effective_permissions(self, user_id: str) -> Set[Permission]:
        perms: Set[Permission] = set()
        for role_name in self._user_roles.get(user_id, set()):
            perms.update(self._inherited_permissions(role_name))
        return perms

    def has_permission(self, user_id: str, permission: Permission) -> bool:
        return permission in self.effective_permissions(user_id)

    def user_roles(self, user_id: str) -> Set[str]:
        return set(self._user_roles.get(user_id, set()))

    def role_hierarchy(self, role_name: str) -> List[str]:
        """Return ancestor chain of a role."""
        chain = [role_name]
        r = self._roles.get(role_name)
        while r and r.parent:
            chain.append(r.parent)
            r = self._roles.get(r.parent)
        return chain


# ─────────────────────────────────────────────
# ABAC POLICY ENGINE (XACML-inspired)
# ─────────────────────────────────────────────

class Effect(Enum):
    PERMIT = "PERMIT"
    DENY   = "DENY"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class Condition:
    attribute: str         # "subject.department", "resource.classification"
    operator : str         # "eq", "gte", "in", "not_in", "contains"
    value    : Any

    def evaluate(self, attrs: Dict[str, Any]) -> bool:
        actual = attrs.get(self.attribute)
        if actual is None:
            return False
        if self.operator == "eq":
            return actual == self.value
        elif self.operator == "neq":
            return actual != self.value
        elif self.operator == "gte":
            return actual >= self.value
        elif self.operator == "lte":
            return actual <= self.value
        elif self.operator == "in":
            return actual in self.value
        elif self.operator == "not_in":
            return actual not in self.value
        elif self.operator == "contains":
            return self.value in actual
        elif self.operator == "startswith":
            return str(actual).startswith(str(self.value))
        return False


@dataclass
class ABACPolicy:
    policy_id  : str
    effect     : Effect
    conditions : List[Condition]    # AND of all conditions
    priority   : int = 100
    description: str = ""

    def matches(self, context: Dict[str, Any]) -> bool:
        return all(c.evaluate(context) for c in self.conditions)

    def evaluate(self, context: Dict[str, Any]) -> Effect:
        if self.matches(context):
            return self.effect
        return Effect.NOT_APPLICABLE


class PolicyCombiningAlgorithm(Enum):
    DENY_OVERRIDES    = "deny_overrides"     # any DENY → DENY
    PERMIT_OVERRIDES  = "permit_overrides"   # any PERMIT → PERMIT
    FIRST_APPLICABLE  = "first_applicable"   # first matching policy wins


class ABACPolicySet:
    """XACML-style policy evaluation."""

    def __init__(self, algorithm: PolicyCombiningAlgorithm =
                 PolicyCombiningAlgorithm.DENY_OVERRIDES):
        self._policies   : List[ABACPolicy] = []
        self._algorithm  = algorithm
        self._audit_log  : List[Dict] = []

    def add_policy(self, policy: ABACPolicy):
        self._policies.append(policy)
        self._policies.sort(key=lambda p: p.priority)

    def evaluate(self, context: Dict[str, Any]) -> Tuple[Effect, List[str]]:
        """Returns (decision, list of matching policies)."""
        decisions: List[Tuple[ABACPolicy, Effect]] = []
        for policy in self._policies:
            effect = policy.evaluate(context)
            if effect != Effect.NOT_APPLICABLE:
                decisions.append((policy, effect))

        self._audit_log.append({
            "ts": time.time(), "context": context,
            "decisions": [(p.policy_id, e.value) for p, e in decisions],
        })

        if not decisions:
            return Effect.DENY, ["default_deny: no applicable policy"]

        if self._algorithm == PolicyCombiningAlgorithm.DENY_OVERRIDES:
            if any(e == Effect.DENY for _, e in decisions):
                deny_reasons = [p.policy_id for p, e in decisions if e == Effect.DENY]
                return Effect.DENY, deny_reasons
            return Effect.PERMIT, [p.policy_id for p, _ in decisions]

        elif self._algorithm == PolicyCombiningAlgorithm.PERMIT_OVERRIDES:
            if any(e == Effect.PERMIT for _, e in decisions):
                return Effect.PERMIT, [p.policy_id for p, e in decisions
                                        if e == Effect.PERMIT]
            return Effect.DENY, [p.policy_id for p, _ in decisions]

        elif self._algorithm == PolicyCombiningAlgorithm.FIRST_APPLICABLE:
            p, e = decisions[0]
            return e, [p.policy_id]

        return Effect.DENY, ["unknown_algorithm"]


# ─────────────────────────────────────────────
# REBAC (Google Zanzibar simplified)
# ─────────────────────────────────────────────

@dataclass
class ZanzibarTuple:
    """(object, relation, user_or_userset)"""
    namespace: str     # "doc", "folder", "group"
    object_id: str
    relation : str
    user_id  : str     # or "group:eng#member"


class ZanzibarEngine:
    """
    Simplified Google Zanzibar ReBAC.
    Resolves: can user U perform action A on object O?
    Via relationship traversal.
    """

    def __init__(self):
        self._tuples: List[ZanzibarTuple] = []
        self._definitions: Dict[str, Dict[str, List[str]]] = {}  # namespace → relation → implied_by

    def write_tuple(self, namespace: str, object_id: str,
                     relation: str, user_id: str):
        self._tuples.append(ZanzibarTuple(namespace, object_id, relation, user_id))

    def define_relation(self, namespace: str, relation: str,
                         implied_by: List[str] = None):
        """Define that `relation` implies/inherits from other relations."""
        if namespace not in self._definitions:
            self._definitions[namespace] = {}
        self._definitions[namespace][relation] = implied_by or []

    def check(self, namespace: str, object_id: str,
               relation: str, user_id: str,
               depth: int = 0, max_depth: int = 10) -> bool:
        if depth > max_depth:
            return False

        # Direct match
        for t in self._tuples:
            if (t.namespace == namespace and t.object_id == object_id and
                    t.relation == relation):
                if t.user_id == user_id:
                    return True
                # Userset expansion: "group:eng#member" → check group membership
                if "#" in t.user_id:
                    parts = t.user_id.split("#")
                    grp_ns, grp_id = parts[0].split(":")
                    grp_rel = parts[1]
                    if self.check(grp_ns, grp_id, grp_rel, user_id, depth + 1):
                        return True

        # Implied relations
        implied_by = self._definitions.get(namespace, {}).get(relation, [])
        for implied_relation in implied_by:
            if self.check(namespace, object_id, implied_relation, user_id, depth + 1):
                return True

        return False

    def expand(self, namespace: str, object_id: str,
                relation: str) -> Set[str]:
        """Return all users who have the given relation on the object."""
        users = set()
        for t in self._tuples:
            if t.namespace == namespace and t.object_id == object_id and \
               t.relation == relation:
                if "#" not in t.user_id:
                    users.add(t.user_id)
                else:
                    parts = t.user_id.split("#")
                    grp_ns, grp_id = parts[0].split(":")
                    grp_rel = parts[1]
                    users.update(self.expand(grp_ns, grp_id, grp_rel))
        return users


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_policy_engines():
    print("=" * 65)
    print("RBAC AND ABAC POLICY ENGINES")
    print("=" * 65)

    # ── Hierarchical RBAC ─────────────────────────
    print("\n[1] HIERARCHICAL RBAC WITH INHERITANCE")
    print("─" * 55)

    rbac = HierarchicalRBAC()
    rbac.add_role("viewer",  {Permission.READ})
    rbac.add_role("editor",  {Permission.WRITE}, parent="viewer")
    rbac.add_role("reviewer",{Permission.REVIEW}, parent="viewer")
    rbac.add_role("publisher",{Permission.PUBLISH}, parent="editor")
    rbac.add_role("admin",   {Permission.ADMIN, Permission.DELETE}, parent="publisher")

    rbac.assign_role("alice", "publisher")
    rbac.assign_role("bob",   "reviewer")
    rbac.assign_role("carol", "admin")

    print(f"  Role hierarchy:")
    for role in ["viewer","editor","publisher","admin"]:
        chain = rbac.role_hierarchy(role)
        print(f"    {role}: inherits from {' → '.join(chain[1:]) if len(chain)>1 else 'none'}")

    print(f"\n  Effective permissions:")
    for user in ["alice", "bob", "carol"]:
        perms = sorted(p.value for p in rbac.effective_permissions(user))
        print(f"    {user:<8}: roles={rbac.user_roles(user)}  perms={perms}")

    # SoD constraint
    rbac.add_sod_rule("editor", "reviewer")
    try:
        rbac.assign_role("dave", "editor")
        rbac.assign_role("dave", "reviewer")   # should fail SoD
    except ValueError as e:
        print(f"\n  SoD constraint: {e}")

    # ── ABAC Policy Evaluation ────────────────────
    print("\n\n[2] ABAC POLICY EVALUATION")
    print("─" * 55)

    policy_set = ABACPolicySet(PolicyCombiningAlgorithm.DENY_OVERRIDES)

    # Allow engineers to read internal docs during business hours
    policy_set.add_policy(ABACPolicy(
        "P1", Effect.PERMIT, priority=10, description="Engineer read internal",
        conditions=[
            Condition("subject.department", "in", ["engineering", "product"]),
            Condition("resource.classification", "eq", "internal"),
            Condition("action", "eq", "read"),
        ]
    ))
    # Deny access outside business hours for confidential
    policy_set.add_policy(ABACPolicy(
        "P2", Effect.DENY, priority=5, description="No confidential after hours",
        conditions=[
            Condition("resource.classification", "eq", "confidential"),
            Condition("environment.hour", "not_in", range(9, 18)),
        ]
    ))
    # Allow finance to read confidential during hours
    policy_set.add_policy(ABACPolicy(
        "P3", Effect.PERMIT, priority=10, description="Finance read confidential",
        conditions=[
            Condition("subject.department", "eq", "finance"),
            Condition("resource.classification", "eq", "confidential"),
            Condition("action", "eq", "read"),
            Condition("environment.hour", "in", range(9, 18)),
        ]
    ))

    test_cases = [
        ("Eng reads internal 10am", {"subject.department":"engineering",
                                      "resource.classification":"internal",
                                      "action":"read", "environment.hour":10}),
        ("Finance reads conf 10am",  {"subject.department":"finance",
                                      "resource.classification":"confidential",
                                      "action":"read", "environment.hour":10}),
        ("Finance reads conf 9pm",   {"subject.department":"finance",
                                      "resource.classification":"confidential",
                                      "action":"read", "environment.hour":21}),
        ("Sales reads internal",     {"subject.department":"sales",
                                      "resource.classification":"internal",
                                      "action":"read", "environment.hour":11}),
    ]
    for label, ctx in test_cases:
        effect, policies = policy_set.evaluate(ctx)
        print(f"  {label:<32}: {effect.value:<8} (policies: {policies})")

    # ── Google Zanzibar ReBAC ─────────────────────
    print("\n\n[3] GOOGLE ZANZIBAR REBAC")
    print("─" * 55)

    z = ZanzibarEngine()
    # Define: writer implies reader
    z.define_relation("doc", "reader", implied_by=["writer"])
    z.define_relation("doc", "writer", implied_by=["owner"])

    # Direct permissions
    z.write_tuple("doc", "readme", "owner",  "user:alice")
    z.write_tuple("doc", "readme", "writer", "user:bob")
    z.write_tuple("doc", "api-spec", "reader", "group:eng#member")

    # Group membership
    z.write_tuple("group", "eng", "member", "user:carol")
    z.write_tuple("group", "eng", "member", "user:dave")

    checks = [
        ("user:alice",  "doc", "readme",   "owner",  True),
        ("user:alice",  "doc", "readme",   "writer", True),   # inherited from owner
        ("user:alice",  "doc", "readme",   "reader", True),   # inherited from writer
        ("user:bob",    "doc", "readme",   "reader", True),   # inherited from writer
        ("user:carol",  "doc", "api-spec", "reader", True),   # via group:eng#member
        ("user:dave",   "doc", "api-spec", "reader", True),
        ("user:alice",  "doc", "api-spec", "reader", False),  # not in group
        ("user:nobody", "doc", "readme",   "reader", False),
    ]
    for user, ns, obj, rel, expected in checks:
        result = z.check(ns, obj, rel, user)
        status = "OK" if result == expected else "FAIL"
        print(f"  [{status}] {user:<16} → {ns}:{obj}#{rel}: {result}")

    # Expand
    readers = z.expand("doc", "api-spec", "reader")
    print(f"\n  All readers of doc:api-spec: {sorted(readers)}")


if __name__ == "__main__":
    demonstrate_policy_engines()
