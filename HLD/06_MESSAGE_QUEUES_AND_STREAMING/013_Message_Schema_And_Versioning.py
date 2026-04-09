"""
MESSAGE SCHEMA AND VERSIONING
================================

Problem Statement:
In a distributed system, producers and consumers evolve independently.
A producer may update its message schema (add fields, rename, remove),
but consumers may be on a different version — deployed at different times.
Schema mismatches cause deserialization errors, data loss, or corrupt processing.

Schema Evolution Rules (compatibility):
  Backward compatible: new consumer can read old messages.
    → Safe to ADD optional fields (with defaults).
    → Safe to REMOVE a field the consumer was ignoring.
  Forward compatible:  old consumer can read new messages.
    → Safe if new fields are optional (consumer ignores unknown).
  Full compatible:     both directions safe simultaneously.
  Breaking change:     rename field, change type, remove required field.

Versioning Strategies:
  1. Field versioning: include "version" field in payload. Consumer switches on version.
     Simple. Risk: consumer must handle all past versions explicitly.

  2. Schema Registry: central registry stores schema definitions.
     Kafka Schema Registry (Avro/Protobuf). Each message carries schema ID, not inline schema.
     Producer registers schema → gets ID. Consumer fetches schema by ID → deserializes.
     Enforces compatibility rules at publish time.

  3. Event type versioning: "order.placed.v1", "order.placed.v2" are separate event types.
     Consumers subscribe to both and handle each version independently.
     Clear but proliferates event type names.

  4. Upcasting: transform old message to latest schema before processing.
     Old messages stored as-is. On read, upcast chain applied: v1 → v2 → v3.
     Used heavily in Event Sourcing.

Avro vs Protobuf vs JSON:
  JSON:      human-readable, no schema enforcement, slow, large.
  Avro:      schema required for encoding/decoding, compact, schema evolution support.
  Protobuf:  field numbers, compact, fast, good evolution support via optional fields.
  MessagePack: binary JSON, faster/smaller but no schema enforcement.

Practical Rules:
  ✓ Always include event_type and schema_version in every message envelope.
  ✓ Never remove or rename fields in a live schema without versioning.
  ✓ Add fields as optional with defaults (backward compatible).
  ✓ Validate schema at producer before publishing (fail fast).
  ✓ Use a schema registry for teams with many producers/consumers.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from collections import defaultdict
import time
import uuid
import json
import copy


# ─────────────────────────────────────────────
# MESSAGE ENVELOPE
# ─────────────────────────────────────────────

@dataclass
class MessageEnvelope:
    event_id      : str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type    : str   = ""
    schema_version: str   = "1.0"
    producer      : str   = ""
    timestamp     : float = field(default_factory=time.time)
    payload       : Dict  = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "event_id"      : self.event_id,
            "event_type"    : self.event_type,
            "schema_version": self.schema_version,
            "producer"      : self.producer,
            "timestamp"     : self.timestamp,
            "payload"       : self.payload,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "MessageEnvelope":
        return cls(
            event_id       = d.get("event_id", ""),
            event_type     = d.get("event_type", ""),
            schema_version = d.get("schema_version", "1.0"),
            producer       = d.get("producer", ""),
            timestamp      = d.get("timestamp", time.time()),
            payload        = d.get("payload", {}),
        )


# ─────────────────────────────────────────────
# SCHEMA DEFINITION
# ─────────────────────────────────────────────

@dataclass
class FieldDef:
    name    : str
    required: bool = True
    default : Any  = None
    field_type: type = str


@dataclass
class Schema:
    event_type   : str
    version      : str
    fields       : List[FieldDef]
    description  : str = ""

    def validate(self, payload: Dict) -> Tuple[bool, List[str]]:
        errors = []
        for f in self.fields:
            if f.required and f.name not in payload:
                errors.append(f"Missing required field: '{f.name}'")
            elif f.name in payload and not isinstance(payload[f.name], f.field_type):
                # basic type check (simplified)
                pass
        return len(errors) == 0, errors

    def fill_defaults(self, payload: Dict) -> Dict:
        result = dict(payload)
        for f in self.fields:
            if f.name not in result and f.default is not None:
                result[f.name] = f.default
        return result


# ─────────────────────────────────────────────
# SCHEMA REGISTRY
# ─────────────────────────────────────────────

class IncompatibleSchemaError(Exception):
    pass


class SchemaRegistry:
    """
    Central schema store. Enforces compatibility rules on registration.
    Simulates Confluent Schema Registry behaviour.
    """

    def __init__(self):
        self._schemas   : Dict[str, List[Schema]] = defaultdict(list)
        self._schema_ids: Dict[int, Schema] = {}
        self._next_id   = 1

    def register(self, schema: Schema,
                 compatibility: str = "BACKWARD") -> int:
        """
        Register schema. Returns schema_id.
        Checks compatibility against the latest version.
        """
        key     = f"{schema.event_type}"
        history = self._schemas[key]

        if history:
            latest = history[-1]
            ok, reason = self._check_compatibility(latest, schema, compatibility)
            if not ok:
                raise IncompatibleSchemaError(
                    f"Schema {schema.event_type} v{schema.version} is not "
                    f"{compatibility} compatible with v{latest.version}: {reason}")

        schema_id = self._next_id
        self._next_id += 1
        self._schemas[key].append(schema)
        self._schema_ids[schema_id] = schema
        return schema_id

    def latest(self, event_type: str) -> Optional[Schema]:
        versions = self._schemas.get(event_type)
        return versions[-1] if versions else None

    def get_by_id(self, schema_id: int) -> Optional[Schema]:
        return self._schema_ids.get(schema_id)

    def _check_compatibility(self, old: Schema, new: Schema,
                              mode: str) -> Tuple[bool, str]:
        """Simplified compatibility check based on field additions/removals."""
        old_fields  = {f.name for f in old.fields}
        new_fields  = {f.name for f in new.fields}
        new_required = {f.name for f in new.fields if f.required}

        if mode == "BACKWARD":
            # New consumer reads old messages:
            # New required fields not in old schema → old messages missing them → FAIL
            new_required_added = new_required - old_fields
            if new_required_added:
                return False, f"Added required fields (no default): {new_required_added}"
        elif mode == "FORWARD":
            # Old consumer reads new messages:
            # Old required fields removed → old consumer breaks → FAIL
            removed = old_fields - new_fields
            old_required = {f.name for f in old.fields if f.required}
            removed_required = removed & old_required
            if removed_required:
                return False, f"Removed required fields: {removed_required}"
        elif mode == "FULL":
            # Both must hold
            ok1, r1 = self._check_compatibility(old, new, "BACKWARD")
            ok2, r2 = self._check_compatibility(old, new, "FORWARD")
            if not ok1:
                return False, r1
            if not ok2:
                return False, r2
        return True, ""

    def versions(self, event_type: str) -> List[str]:
        return [s.version for s in self._schemas.get(event_type, [])]


# ─────────────────────────────────────────────
# UPCASTER CHAIN (for Event Sourcing)
# ─────────────────────────────────────────────

class UpcastChain:
    """
    Transforms old event payloads to the current schema on read.
    Register upcasters per version pair. Chain applied automatically.
    """

    def __init__(self):
        # upcasters[(event_type, from_version)] = fn(payload) -> payload
        self._upcasters: Dict[Tuple[str, str], Callable] = {}
        self._version_order: Dict[str, List[str]] = defaultdict(list)

    def register(self, event_type: str, from_version: str, to_version: str,
                 fn: Callable[[Dict], Dict]):
        self._upcasters[(event_type, from_version)] = fn
        versions = self._version_order[event_type]
        if from_version not in versions:
            versions.append(from_version)
        if to_version not in versions:
            versions.append(to_version)

    def upcast(self, event_type: str, version: str, payload: Dict) -> Tuple[str, Dict]:
        """Apply upcasters until no more upgrades available."""
        current_version = version
        current_payload = copy.deepcopy(payload)
        max_iter = 10
        for _ in range(max_iter):
            key = (event_type, current_version)
            if key not in self._upcasters:
                break
            fn = self._upcasters[key]
            current_payload, current_version = fn(current_payload)
        return current_version, current_payload


# ─────────────────────────────────────────────
# VERSIONED PRODUCER / CONSUMER
# ─────────────────────────────────────────────

class VersionedProducer:
    def __init__(self, producer_id: str, registry: SchemaRegistry):
        self.producer_id = producer_id
        self.registry    = registry

    def publish(self, event_type: str, payload: Dict) -> MessageEnvelope:
        schema = self.registry.latest(event_type)
        if not schema:
            raise ValueError(f"No schema registered for {event_type}")
        payload = schema.fill_defaults(payload)
        ok, errors = schema.validate(payload)
        if not ok:
            raise ValueError(f"Schema validation failed: {errors}")
        return MessageEnvelope(
            event_type     = event_type,
            schema_version = schema.version,
            producer       = self.producer_id,
            payload        = payload,
        )


class VersionedConsumer:
    def __init__(self, consumer_id: str, upcast_chain: UpcastChain):
        self.consumer_id  = consumer_id
        self.upcast_chain = upcast_chain
        self.processed    : List[Dict] = []

    def process(self, envelope: MessageEnvelope):
        version, payload = self.upcast_chain.upcast(
            envelope.event_type, envelope.schema_version, envelope.payload)
        self.processed.append({
            "event_id"      : envelope.event_id,
            "original_ver"  : envelope.schema_version,
            "processed_ver" : version,
            "payload"       : payload,
        })


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_schema_versioning():
    print("=" * 65)
    print("MESSAGE SCHEMA AND VERSIONING")
    print("=" * 65)

    # ── Schema Registry ───────────────────────────
    print("\n[1] SCHEMA REGISTRY — REGISTER & COMPATIBILITY CHECK")
    print("─" * 55)

    registry = SchemaRegistry()

    # V1: original schema
    schema_v1 = Schema(
        event_type = "order.placed",
        version    = "1.0",
        fields     = [
            FieldDef("order_id",    required=True),
            FieldDef("customer_id", required=True),
            FieldDef("amount",      required=True, field_type=float),
        ],
    )
    id1 = registry.register(schema_v1)
    print(f"  Registered order.placed v1.0 (id={id1})")

    # V2: add optional field (backward compatible)
    schema_v2 = Schema(
        event_type = "order.placed",
        version    = "2.0",
        fields     = [
            FieldDef("order_id",    required=True),
            FieldDef("customer_id", required=True),
            FieldDef("amount",      required=True, field_type=float),
            FieldDef("currency",    required=False, default="USD"),   # new optional field
        ],
    )
    id2 = registry.register(schema_v2)
    print(f"  Registered order.placed v2.0 (id={id2}) — added optional 'currency'")

    # V3: try to add required field without default (BACKWARD INCOMPATIBLE)
    schema_v3_bad = Schema(
        event_type = "order.placed",
        version    = "3.0",
        fields     = [
            FieldDef("order_id",     required=True),
            FieldDef("customer_id",  required=True),
            FieldDef("amount",       required=True, field_type=float),
            FieldDef("currency",     required=False, default="USD"),
            FieldDef("region",       required=True),   # new required field — BREAKING
        ],
    )
    try:
        registry.register(schema_v3_bad)
    except IncompatibleSchemaError as e:
        print(f"  Rejected v3.0 (required field 'region' breaks backward compat): ✓")

    # V3 correct: required=False
    schema_v3_ok = Schema(
        event_type = "order.placed",
        version    = "3.0",
        fields     = [
            FieldDef("order_id",    required=True),
            FieldDef("customer_id", required=True),
            FieldDef("amount",      required=True, field_type=float),
            FieldDef("currency",    required=False, default="USD"),
            FieldDef("region",      required=False, default="us-east-1"),   # optional
        ],
    )
    id3 = registry.register(schema_v3_ok)
    print(f"  Registered order.placed v3.0 (id={id3}) — added optional 'region'")
    print(f"  Versions in registry: {registry.versions('order.placed')}")

    # ── Versioned Producer/Consumer ───────────────
    print("\n\n[2] PRODUCER/CONSUMER WITH SCHEMA VALIDATION")
    print("─" * 55)

    upcast = UpcastChain()

    # Upcaster: v1.0 → v2.0 (add currency default)
    def upcast_v1_to_v2(payload: Dict) -> Tuple[Dict, str]:
        p = dict(payload)
        p.setdefault("currency", "USD")
        return p, "2.0"

    # Upcaster: v2.0 → v3.0 (add region default)
    def upcast_v2_to_v3(payload: Dict) -> Tuple[Dict, str]:
        p = dict(payload)
        p.setdefault("region", "us-east-1")
        return p, "3.0"

    upcast.register("order.placed", "1.0", "2.0", upcast_v1_to_v2)
    upcast.register("order.placed", "2.0", "3.0", upcast_v2_to_v3)

    producer = VersionedProducer("order-service", registry)
    consumer = VersionedConsumer("billing-service", upcast)

    # Producer publishes v3 messages
    msg = producer.publish("order.placed", {
        "order_id": "ORD-001", "customer_id": "CUST-A",
        "amount": 99.99,
    })
    print(f"  Producer published: event_type={msg.event_type} "
          f"v={msg.schema_version} payload={msg.payload}")

    # ── Upcasting Old Messages ────────────────────
    print("\n\n[3] UPCASTING — OLD MESSAGES TO CURRENT SCHEMA")
    print("─" * 55)

    # Simulate messages from different producer versions
    old_messages = [
        MessageEnvelope(event_type="order.placed", schema_version="1.0",
                        payload={"order_id": "OLD-001", "customer_id": "C1", "amount": 50.0}),
        MessageEnvelope(event_type="order.placed", schema_version="2.0",
                        payload={"order_id": "OLD-002", "customer_id": "C2",
                                 "amount": 75.0, "currency": "EUR"}),
        MessageEnvelope(event_type="order.placed", schema_version="3.0",
                        payload={"order_id": "NEW-001", "customer_id": "C3",
                                 "amount": 120.0, "currency": "GBP", "region": "eu-west-1"}),
    ]

    for env in old_messages:
        consumer.process(env)

    for record in consumer.processed:
        print(f"  event={record['event_id']} "
              f"v{record['original_ver']} → v{record['processed_ver']} "
              f"payload={record['payload']}")

    # ── Event Type Versioning ─────────────────────
    print("\n\n[4] EVENT TYPE VERSIONING — SEPARATE EVENT TYPES")
    print("─" * 55)
    print("  Strategy: 'order.placed.v1' and 'order.placed.v2' are distinct types")
    print("  Consumer subscribes to BOTH and handles each independently")
    print()
    handlers = {
        "order.placed.v1": lambda p: f"v1 handler: order={p['order_id']}",
        "order.placed.v2": lambda p: f"v2 handler: order={p['order_id']} cur={p.get('currency','?')}",
    }
    test_events = [
        ("order.placed.v1", {"order_id": "A1", "amount": 50}),
        ("order.placed.v2", {"order_id": "A2", "amount": 80, "currency": "EUR"}),
    ]
    for event_type, payload in test_events:
        result = handlers[event_type](payload)
        print(f"  {result}")

    # ── Compatibility Matrix ──────────────────────
    print("\n\n[5] SCHEMA CHANGE COMPATIBILITY MATRIX")
    print("─" * 55)
    rows = [
        ("Add optional field (default)", "✓ BACKWARD", "✓ FORWARD", "✓ FULL"),
        ("Add required field",           "✗ BACKWARD", "✓ FORWARD", "✗ FULL"),
        ("Remove optional field",        "✓ BACKWARD", "✗ FORWARD", "✗ FULL"),
        ("Remove required field",        "✗ BACKWARD", "✗ FORWARD", "✗ FULL"),
        ("Rename field",                 "✗ BREAKING", "✗ BREAKING","✗ BREAKING"),
        ("Change field type",            "✗ BREAKING", "✗ BREAKING","✗ BREAKING"),
    ]
    print(f"  {'Change Type':<32} {'Backward':>10} {'Forward':>10} {'Full':>6}")
    print(f"  {'─'*62}")
    for change, backward, forward, full in rows:
        print(f"  {change:<32} {backward:>10} {forward:>10} {full:>6}")

    print("\n\n[6] SCHEMA BEST PRACTICES")
    print("─" * 55)
    practices = [
        "Include schema_version in every message envelope",
        "Use a schema registry — enforce compatibility at publish time",
        "Never rename or remove fields without a versioning strategy",
        "Add fields as optional with defaults (backward compatible)",
        "Use upcasters for event-sourced systems — don't alter stored events",
        "Prefer Protobuf/Avro over JSON for strong schema enforcement",
        "Version your event types if consumers handle schemas differently",
    ]
    for practice in practices:
        print(f"  • {practice}")


if __name__ == "__main__":
    demonstrate_schema_versioning()
