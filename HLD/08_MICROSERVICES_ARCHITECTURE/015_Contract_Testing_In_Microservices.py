"""
CONTRACT TESTING IN MICROSERVICES
=====================================

Problem Statement:
In a distributed system, Service A (consumer) calls Service B (provider).
How do you know that B's API is compatible with A's expectations?
Integration tests require both services running simultaneously — expensive,
slow, and flaky. End-to-end tests catch issues too late.

Contract Testing:
  The consumer defines exactly what it NEEDS from the provider:
    - Which endpoint it calls.
    - What request body/params it sends.
    - What response structure and fields it expects.
  The provider verifies it can satisfy every consumer's contract.
  No need to run both services simultaneously — contracts are files.

Consumer-Driven Contract Testing (CDC):
  1. Consumer writes a contract: "I expect GET /orders/{id} to return
     {order_id, status, total}. I don't care about other fields."
  2. Contract is published to a Contract Broker (Pact Broker).
  3. Provider's CI pulls contracts and verifies it satisfies each one.
  4. If provider breaks a contract → pipeline fails BEFORE deployment.
  5. Consumer and provider can deploy independently, safely.

Key Principle: Consumer defines the contract, not the provider.
  Provider must satisfy consumer's expectations (not vice versa).
  Postel's Law: "Be liberal in what you accept, conservative in what you send."
  In CDC: provider may return MORE fields; consumer tests for the ones it uses.

Pact-style Testing:
  Pact is the most popular CDC testing framework.
  Consumer generates a pact file (JSON contract) during unit tests.
  Provider verifies the pact file against its actual implementation.
  Pact Broker: central store for pact files, with can-i-deploy checks.

Benefits vs Integration Testing:
  Integration test:   Both services must run. Slow. Flaky. Tests everything.
  Contract test:      No running services needed. Fast. Stable. Tests the boundary.
  Contract tests don't replace integration tests — they complement them.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import time
import uuid


# ─────────────────────────────────────────────
# CONTRACT TYPES
# ─────────────────────────────────────────────

@dataclass
class RequestExpectation:
    method  : str
    path    : str
    headers : Dict[str, str] = field(default_factory=dict)
    body    : Optional[Dict] = None
    query   : Dict[str, str] = field(default_factory=dict)


@dataclass
class ResponseExpectation:
    status  : int
    headers : Dict[str, str] = field(default_factory=dict)
    body    : Optional[Dict] = None   # only the fields the consumer needs


@dataclass
class Interaction:
    """A single request-response pair that the consumer expects."""
    description: str
    request    : RequestExpectation
    response   : ResponseExpectation


@dataclass
class ConsumerContract:
    """
    A pact file: the consumer's expectations of a provider.
    Consumer writes this during its unit tests.
    """
    consumer_name : str
    provider_name : str
    version       : str
    interactions  : List[Interaction] = field(default_factory=list)

    def add_interaction(self, interaction: Interaction):
        self.interactions.append(interaction)

    def to_dict(self) -> Dict:
        return {
            "consumer"    : {"name": self.consumer_name},
            "provider"    : {"name": self.provider_name},
            "version"     : self.version,
            "interactions": [
                {
                    "description": i.description,
                    "request"    : {
                        "method" : i.request.method,
                        "path"   : i.request.path,
                        "headers": i.request.headers,
                        "body"   : i.request.body,
                        "query"  : i.request.query,
                    },
                    "response"   : {
                        "status" : i.response.status,
                        "headers": i.response.headers,
                        "body"   : i.response.body,
                    },
                }
                for i in self.interactions
            ],
        }


# ─────────────────────────────────────────────
# CONTRACT BROKER
# ─────────────────────────────────────────────

@dataclass
class PublishedContract:
    contract    : ConsumerContract
    published_at: float = field(default_factory=time.time)
    verified_by : List[str] = field(default_factory=list)   # provider versions


class ContractBroker:
    """
    Central store for consumer contracts.
    Providers pull contracts from here to verify.
    Answers: "can-i-deploy consumer X version Y with provider Z version W?"
    """

    def __init__(self):
        self._contracts : Dict[str, List[PublishedContract]] = {}  # provider → contracts
        self._verification_results : List[Dict] = []

    def publish(self, contract: ConsumerContract):
        key = f"{contract.provider_name}"
        self._contracts.setdefault(key, []).append(PublishedContract(contract))
        print(f"  [Broker] Published: {contract.consumer_name} → {contract.provider_name} "
              f"v{contract.version}")

    def get_contracts_for_provider(self, provider_name: str) -> List[ConsumerContract]:
        return [pc.contract for pc in self._contracts.get(provider_name, [])]

    def record_verification(self, provider_name: str, provider_version: str,
                             consumer_name: str, contract_version: str,
                             passed: bool, failures: List[str]):
        self._verification_results.append({
            "provider"         : provider_name,
            "provider_version" : provider_version,
            "consumer"         : consumer_name,
            "contract_version" : contract_version,
            "passed"           : passed,
            "failures"         : failures,
            "verified_at"      : time.time(),
        })

    def can_deploy(self, service_name: str, version: str) -> Tuple[bool, List[str]]:
        """Check if a service version has passed all required contract verifications."""
        relevant = [
            r for r in self._verification_results
            if (r["provider"] == service_name and r["provider_version"] == version)
            or (r["consumer"] == service_name and r["contract_version"] == version)
        ]
        if not relevant:
            return False, ["No verification results found — run contract tests first"]
        failures = []
        for r in relevant:
            if not r["passed"]:
                failures.extend([f"[{r['consumer']} → {r['provider']}] {f}"
                                  for f in r["failures"]])
        return len(failures) == 0, failures

    def summary(self) -> List[Dict]:
        return self._verification_results


# ─────────────────────────────────────────────
# PROVIDER VERIFIER
# ─────────────────────────────────────────────

class ProviderVerifier:
    """
    Runs provider verification: given a consumer contract, verify the
    provider's actual implementation satisfies every interaction.
    """

    def __init__(self, provider_name: str, provider_handler: Callable):
        self.provider_name    = provider_name
        self._handler         = provider_handler

    def verify(self, contract: ConsumerContract) -> Tuple[bool, List[str]]:
        """Verify one consumer's contract against this provider."""
        failures = []
        for interaction in contract.interactions:
            failure = self._verify_interaction(interaction)
            if failure:
                failures.append(f"[{interaction.description}] {failure}")

        return len(failures) == 0, failures

    def _verify_interaction(self, interaction: Interaction) -> Optional[str]:
        """Return None if interaction is satisfied, else error message."""
        req  = interaction.request
        exp  = interaction.response

        try:
            actual = self._handler(
                req.method, req.path, req.headers, req.body, req.query)
        except Exception as e:
            return f"Handler raised exception: {e}"

        if actual["status"] != exp.status:
            return (f"Status mismatch: expected {exp.status}, "
                    f"got {actual['status']}")

        if exp.body:
            body_error = self._check_body(exp.body, actual.get("body", {}))
            if body_error:
                return f"Body mismatch: {body_error}"

        return None

    def _check_body(self, expected: Dict, actual: Dict,
                    path: str = "") -> Optional[str]:
        """
        Check that all fields in expected are present in actual.
        Extra fields in actual are OK (consumer only checks what it needs).
        """
        if not isinstance(actual, dict):
            return f"expected dict at '{path}', got {type(actual).__name__}"

        for key, exp_val in expected.items():
            full_path = f"{path}.{key}" if path else key
            if key not in actual:
                return f"missing field '{full_path}'"
            if isinstance(exp_val, dict):
                nested = self._check_body(exp_val, actual[key], full_path)
                if nested:
                    return nested
            elif isinstance(exp_val, type):
                if not isinstance(actual[key], exp_val):
                    return (f"type mismatch at '{full_path}': "
                            f"expected {exp_val.__name__}, "
                            f"got {type(actual[key]).__name__}")
        return None


# ─────────────────────────────────────────────
# SIMULATED PROVIDER IMPLEMENTATIONS
# ─────────────────────────────────────────────

def order_service_v1(method: str, path: str, headers: Dict,
                     body: Optional[Dict], query: Dict) -> Dict:
    """Provider v1: returns all required fields."""
    if method == "GET" and path.startswith("/orders/"):
        order_id = path.split("/")[-1]
        return {
            "status": 200,
            "body"  : {
                "order_id"     : order_id,
                "status"       : "confirmed",
                "total"        : 99.99,
                "customer_id"  : "cust-001",
                "created_at"   : "2025-01-01T00:00:00Z",
                "items"        : [{"sku": "A1", "qty": 1, "price": 99.99}],
            }
        }
    if method == "POST" and path == "/orders":
        return {
            "status": 201,
            "body"  : {
                "order_id" : "ord-" + str(uuid.uuid4())[:6],
                "status"   : "pending",
                "total"    : body.get("total", 0) if body else 0,
            }
        }
    return {"status": 404, "body": {"error": "not found"}}


def order_service_v2_broken(method: str, path: str, headers: Dict,
                             body: Optional[Dict], query: Dict) -> Dict:
    """Provider v2: BROKE the contract — renamed 'status' to 'order_status'."""
    if method == "GET" and path.startswith("/orders/"):
        order_id = path.split("/")[-1]
        return {
            "status": 200,
            "body"  : {
                "order_id"     : order_id,
                "order_status" : "confirmed",   # BREAKING: renamed field!
                "total"        : 99.99,
                "customer_id"  : "cust-001",
            }
        }
    return {"status": 404, "body": {"error": "not found"}}


def order_service_v3_fixed(method: str, path: str, headers: Dict,
                            body: Optional[Dict], query: Dict) -> Dict:
    """Provider v3: fixed — has both 'status' and 'order_status' (backwards compat)."""
    if method == "GET" and path.startswith("/orders/"):
        order_id = path.split("/")[-1]
        return {
            "status": 200,
            "body"  : {
                "order_id"     : order_id,
                "status"       : "confirmed",        # restored for consumers
                "order_status" : "confirmed",        # new field too
                "total"        : 99.99,
                "customer_id"  : "cust-001",
                "created_at"   : "2025-01-01T00:00:00Z",
            }
        }
    if method == "POST" and path == "/orders":
        return {
            "status": 201,
            "body"  : {
                "order_id" : "ord-abc123",
                "status"   : "pending",
                "total"    : body.get("total", 0) if body else 0,
            }
        }
    return {"status": 404, "body": {"error": "not found"}}


# ─────────────────────────────────────────────
# CONSUMER CONTRACT DEFINITIONS
# ─────────────────────────────────────────────

def build_mobile_bff_contract() -> ConsumerContract:
    contract = ConsumerContract("mobile-bff", "order-service", "1.0.0")

    contract.add_interaction(Interaction(
        description="get order by ID",
        request  = RequestExpectation("GET", "/orders/ord-001",
                                      headers={"Accept": "application/json"}),
        response = ResponseExpectation(200, body={
            "order_id": str,    # type check: must be a string
            "status"  : str,    # REQUIRED by mobile-bff
            "total"   : float,  # REQUIRED by mobile-bff
            # consumer does NOT require 'items', 'customer_id', etc.
        }),
    ))

    contract.add_interaction(Interaction(
        description="create a new order",
        request  = RequestExpectation("POST", "/orders", body={"total": 99.99}),
        response = ResponseExpectation(201, body={
            "order_id": str,
            "status"  : str,
        }),
    ))
    return contract


def build_analytics_service_contract() -> ConsumerContract:
    contract = ConsumerContract("analytics-service", "order-service", "1.0.0")

    contract.add_interaction(Interaction(
        description="get order totals for reporting",
        request  = RequestExpectation("GET", "/orders/ord-001"),
        response = ResponseExpectation(200, body={
            "order_id"  : str,
            "total"     : float,
            "created_at": str,     # analytics needs this; mobile-bff doesn't
        }),
    ))
    return contract


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_contract_testing():
    print("=" * 65)
    print("CONTRACT TESTING IN MICROSERVICES")
    print("=" * 65)

    broker   = ContractBroker()
    verifier = ProviderVerifier("order-service", order_service_v1)

    # ── 1. Consumers define contracts ─────────────
    print("\n[1] CONSUMERS DEFINE AND PUBLISH CONTRACTS")
    print("─" * 55)
    mobile_contract    = build_mobile_bff_contract()
    analytics_contract = build_analytics_service_contract()

    broker.publish(mobile_contract)
    broker.publish(analytics_contract)
    print(f"\n  Contracts published:")
    for contract in [mobile_contract, analytics_contract]:
        print(f"    {contract.consumer_name} → {contract.provider_name}")
        for i in contract.interactions:
            print(f"      [{i.request.method} {i.request.path}] "
                  f"expects status={i.response.status}")

    # ── 2. Provider v1 verification ───────────────
    print("\n\n[2] PROVIDER v1 VERIFIES ALL CONSUMER CONTRACTS")
    print("─" * 55)
    contracts_to_verify = broker.get_contracts_for_provider("order-service")
    all_passed = True
    for contract in contracts_to_verify:
        passed, failures = verifier.verify(contract)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {contract.consumer_name} → order-service v1")
        for f in failures:
            print(f"         {f}")
        broker.record_verification("order-service", "v1.0.0",
                                    contract.consumer_name, contract.version,
                                    passed, failures)
        all_passed = all_passed and passed

    can_deploy, issues = broker.can_deploy("order-service", "v1.0.0")
    print(f"\n  can-i-deploy order-service v1.0.0? {'YES' if can_deploy else 'NO'}")

    # ── 3. Provider v2 breaks contracts ───────────
    print("\n\n[3] PROVIDER v2 BREAKS CONTRACT — CI CATCHES IT")
    print("─" * 55)
    verifier_v2 = ProviderVerifier("order-service", order_service_v2_broken)
    print("  Provider v2 renamed 'status' → 'order_status' (breaking change)")
    print()
    for contract in contracts_to_verify:
        passed, failures = verifier_v2.verify(contract)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {contract.consumer_name} → order-service v2 (broken)")
        for f in failures:
            print(f"         {f}")
        broker.record_verification("order-service", "v2.0.0-broken",
                                    contract.consumer_name, contract.version,
                                    passed, failures)

    can_deploy, issues = broker.can_deploy("order-service", "v2.0.0-broken")
    print(f"\n  can-i-deploy order-service v2.0.0-broken? {'YES' if can_deploy else 'NO'}")
    for issue in issues:
        print(f"    {issue}")
    print(f"  → Pipeline BLOCKED. Breaking change caught before deployment.")

    # ── 4. Provider v3 fixes it ───────────────────
    print("\n\n[4] PROVIDER v3 ADDS BACKWARDS COMPAT — CONTRACTS PASS")
    print("─" * 55)
    verifier_v3 = ProviderVerifier("order-service", order_service_v3_fixed)
    for contract in contracts_to_verify:
        passed, failures = verifier_v3.verify(contract)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {contract.consumer_name} → order-service v3 (fixed)")
        broker.record_verification("order-service", "v3.0.0",
                                    contract.consumer_name, contract.version,
                                    passed, failures)

    can_deploy, issues = broker.can_deploy("order-service", "v3.0.0")
    print(f"\n  can-i-deploy order-service v3.0.0? {'YES' if can_deploy else 'NO'}")

    # ── 5. Contract JSON structure ─────────────────
    print("\n\n[5] CONTRACT FILE STRUCTURE (Pact-style JSON)")
    print("─" * 55)
    contract_dict = mobile_contract.to_dict()
    print(f"  consumer: {contract_dict['consumer']['name']}")
    print(f"  provider: {contract_dict['provider']['name']}")
    print(f"  version:  {contract_dict['version']}")
    print(f"  interactions ({len(contract_dict['interactions'])}):")
    for ix in contract_dict["interactions"]:
        print(f"    [{ix['request']['method']} {ix['request']['path']}]")
        print(f"      expects status={ix['response']['status']}")
        expected_fields = list(ix['response'].get('body', {}).keys())
        print(f"      expects body fields: {expected_fields}")
        print(f"      (extra fields OK — consumer only checks what it uses)")

    # ── 6. Verification summary ────────────────────
    print("\n\n[6] VERIFICATION HISTORY (all runs)")
    print("─" * 55)
    print(f"  {'Provider Version':<22} {'Consumer':<22} {'Result'}")
    print(f"  {'─'*55}")
    for r in broker.summary():
        result = "PASS" if r["passed"] else f"FAIL ({len(r['failures'])} issues)"
        print(f"  {r['provider']} {r['provider_version']:<14} "
              f"{r['consumer']:<22} {result}")

    # ── 7. Benefits comparison ─────────────────────
    print("\n\n[7] CONTRACT TESTING vs INTEGRATION TESTING")
    print("─" * 55)
    comparisons = [
        ("Requires both services running", "YES (slow, flaky)",   "NO (contracts are files)"),
        ("Speed",                          "Minutes to hours",     "Seconds"),
        ("Stability",                      "Flaky (network, data)","Stable (no network)"),
        ("Catches breaking changes",       "Yes (but late)",       "Yes (at PR time)"),
        ("Tests business logic",           "Yes",                  "No (just the boundary)"),
        ("Enables independent deploy",     "Limited",              "Yes (via can-i-deploy)"),
        ("Replaces integration tests?",    "—",                    "No — complements them"),
    ]
    print(f"  {'Aspect':<35} {'Integration':<25} {'Contract'}")
    print(f"  {'─'*72}")
    for aspect, integ, contract in comparisons:
        print(f"  {aspect:<35} {integ:<25} {contract}")


if __name__ == "__main__":
    demonstrate_contract_testing()
