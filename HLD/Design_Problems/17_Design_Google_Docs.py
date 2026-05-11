"""
Design Google Docs — Python Simulation
========================================
Simulates core collaborative editing mechanics:
  - Operational Transformation (OT) with Insert/Delete/Retain ops
  - Server authority model with sequence numbering
  - Client state machine (synchronized / awaiting / buffering)
  - Cursor broadcasting
  - Document snapshots + delta replay
  - Revision history
  - WebSocket simulation with op broadcasting
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from enum import Enum


# ---------------------------------------------------------------------------
# Operation Types
# ---------------------------------------------------------------------------

class OpType(Enum):
    INSERT = "insert"
    DELETE = "delete"
    RETAIN = "retain"


@dataclass
class Operation:
    op_type: OpType
    position: int
    content: str = ""        # For INSERT
    length: int = 0          # For DELETE / RETAIN
    op_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_id: str = ""
    seq_num: int = 0         # Assigned by server


# ---------------------------------------------------------------------------
# Operational Transformation Engine
# ---------------------------------------------------------------------------

class OperationalTransform:
    """
    Transforms concurrent operations so all clients converge to the same state.

    Core insight: transform(op_a, op_b) returns op_a' such that:
      apply(apply(doc, op_b), op_a') == apply(apply(doc, op_a), op_b')
    """

    @staticmethod
    def transform(op_a: Operation, op_b: Operation) -> Operation:
        """
        Transform op_a given that op_b was applied concurrently.
        Returns a new op_a that should be applied after op_b.
        """
        if op_a.op_type == OpType.INSERT and op_b.op_type == OpType.INSERT:
            return OperationalTransform._transform_insert_insert(op_a, op_b)
        elif op_a.op_type == OpType.INSERT and op_b.op_type == OpType.DELETE:
            return OperationalTransform._transform_insert_delete(op_a, op_b)
        elif op_a.op_type == OpType.DELETE and op_b.op_type == OpType.INSERT:
            return OperationalTransform._transform_delete_insert(op_a, op_b)
        elif op_a.op_type == OpType.DELETE and op_b.op_type == OpType.DELETE:
            return OperationalTransform._transform_delete_delete(op_a, op_b)
        return op_a

    @staticmethod
    def _transform_insert_insert(op_a: Operation, op_b: Operation) -> Operation:
        """
        If op_b inserted text before op_a's position, shift op_a right.
        Tie-break: if same position, use op_id lexicographic order.
        """
        new_pos = op_a.position
        if op_b.position < op_a.position:
            new_pos += len(op_b.content)
        elif op_b.position == op_a.position and op_b.op_id < op_a.op_id:
            new_pos += len(op_b.content)
        return Operation(
            op_type=OpType.INSERT,
            position=new_pos,
            content=op_a.content,
            op_id=op_a.op_id,
            user_id=op_a.user_id
        )

    @staticmethod
    def _transform_insert_delete(op_a: Operation, op_b: Operation) -> Operation:
        """op_a is INSERT, op_b is DELETE. Adjust op_a's position for the deletion."""
        new_pos = op_a.position
        if op_b.position < op_a.position:
            # How much of the delete overlaps before op_a's position?
            shift = min(op_b.length, op_a.position - op_b.position)
            new_pos -= shift
        return Operation(
            op_type=OpType.INSERT,
            position=max(0, new_pos),
            content=op_a.content,
            op_id=op_a.op_id,
            user_id=op_a.user_id
        )

    @staticmethod
    def _transform_delete_insert(op_a: Operation, op_b: Operation) -> Operation:
        """op_a is DELETE, op_b is INSERT. Adjust op_a's position for the insertion."""
        new_pos = op_a.position
        if op_b.position <= op_a.position:
            new_pos += len(op_b.content)
        return Operation(
            op_type=OpType.DELETE,
            position=new_pos,
            length=op_a.length,
            op_id=op_a.op_id,
            user_id=op_a.user_id
        )

    @staticmethod
    def _transform_delete_delete(op_a: Operation, op_b: Operation) -> Operation:
        """Both are deletes. Adjust op_a given op_b's deletion."""
        pa, la = op_a.position, op_a.length
        pb, lb = op_b.position, op_b.length

        if pa >= pb + lb:
            # op_a is entirely after op_b's range -> shift left
            return Operation(OpType.DELETE, pa - lb, length=la,
                             op_id=op_a.op_id, user_id=op_a.user_id)
        elif pa + la <= pb:
            # op_a is entirely before op_b -> unaffected
            return op_a
        else:
            # Overlap: reduce op_a's length by the overlapping portion
            overlap_start = max(pa, pb)
            overlap_end = min(pa + la, pb + lb)
            overlap = overlap_end - overlap_start
            new_pos = min(pa, pb)
            new_len = max(0, la - overlap)
            return Operation(OpType.DELETE, new_pos, length=new_len,
                             op_id=op_a.op_id, user_id=op_a.user_id)


# ---------------------------------------------------------------------------
# Document State
# ---------------------------------------------------------------------------

class DocumentState:
    """Represents the document as a mutable string."""

    def __init__(self, content: str = ""):
        self.content = list(content)     # List of chars for O(1) indexing

    def apply_operation(self, op: Operation) -> bool:
        """Apply a single operation to the document. Returns success."""
        if op.op_type == OpType.INSERT:
            pos = min(op.position, len(self.content))
            self.content[pos:pos] = list(op.content)
            return True
        elif op.op_type == OpType.DELETE:
            pos = op.position
            length = min(op.length, len(self.content) - pos)
            if length > 0 and pos < len(self.content):
                del self.content[pos:pos + length]
            return True
        return False

    def get_content(self) -> str:
        return "".join(self.content)

    def length(self) -> int:
        return len(self.content)


# ---------------------------------------------------------------------------
# Document History (Snapshot + Delta)
# ---------------------------------------------------------------------------

class DocumentHistory:
    """
    Stores snapshots + operation log for efficient document loading.
    Snapshot every SNAPSHOT_INTERVAL operations.
    """
    SNAPSHOT_INTERVAL = 10   # In production: 1000

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self._snapshots: list[tuple[int, str]] = [(0, "")]   # (seq_num, content)
        self._operations: list[Operation] = []
        self._seq_counter = 0

    def append_operation(self, op: Operation) -> int:
        self._seq_counter += 1
        op.seq_num = self._seq_counter
        self._operations.append(op)

        # Auto-snapshot
        if len(self._operations) % self.SNAPSHOT_INTERVAL == 0:
            self._create_snapshot()

        return self._seq_counter

    def _create_snapshot(self):
        """Materialize current document state as a snapshot."""
        doc = self._replay_from_start()
        self._snapshots.append((self._seq_counter, doc.get_content()))

    def load_document(self) -> tuple[str, int]:
        """
        Return (content, seq_num) using most recent snapshot + delta.
        """
        snap_seq, snap_content = self._snapshots[-1]
        doc = DocumentState(snap_content)
        for op in self._operations:
            if op.seq_num > snap_seq:
                doc.apply_operation(op)
        return doc.get_content(), self._seq_counter

    def get_ops_since(self, seq_num: int) -> list[Operation]:
        return [op for op in self._operations if op.seq_num > seq_num]

    def _replay_from_start(self) -> DocumentState:
        doc = DocumentState("")
        for op in self._operations:
            doc.apply_operation(op)
        return doc


# ---------------------------------------------------------------------------
# Collaborator Manager (Cursor & Presence)
# ---------------------------------------------------------------------------

@dataclass
class CursorState:
    user_id: str
    name: str
    color: str
    position: int
    last_seen: float


class CollaboratorManager:
    """Tracks active users and their cursor positions."""

    COLORS = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#FF6D00", "#A142F4"]

    def __init__(self):
        self._collaborators: dict[str, CursorState] = {}
        self._color_idx = 0

    def join(self, user_id: str, name: str):
        color = self.COLORS[self._color_idx % len(self.COLORS)]
        self._color_idx += 1
        self._collaborators[user_id] = CursorState(
            user_id=user_id, name=name, color=color,
            position=0, last_seen=time.time()
        )

    def leave(self, user_id: str):
        self._collaborators.pop(user_id, None)

    def update_cursor(self, user_id: str, position: int):
        if user_id in self._collaborators:
            self._collaborators[user_id].position = position
            self._collaborators[user_id].last_seen = time.time()

    def get_active_cursors(self, exclude_user: str = None) -> list[dict]:
        return [
            {"user": c.name, "position": c.position, "color": c.color}
            for uid, c in self._collaborators.items()
            if uid != exclude_user
        ]

    def active_count(self) -> int:
        return len(self._collaborators)


# ---------------------------------------------------------------------------
# Client (with state machine: sync / awaiting / buffering)
# ---------------------------------------------------------------------------

class ClientState(Enum):
    SYNCHRONIZED = "synchronized"
    AWAITING = "awaiting"
    OFFLINE = "offline"


class CollaborativeClient:
    """Simulates a Google Docs browser client."""

    def __init__(self, user_id: str, name: str, server: "CollaborativeDocument"):
        self.user_id = user_id
        self.name = name
        self.server = server
        self.state = ClientState.SYNCHRONIZED
        self.last_seq = 0
        self._in_flight: Optional[Operation] = None
        self._pending: list[Operation] = []
        self._local_doc = DocumentState("")
        self._offline_buffer: list[Operation] = []

    def type_operation(self, op: Operation) -> dict:
        """User types — apply locally and send to server."""
        op.user_id = self.user_id
        self._local_doc.apply_operation(op)

        if self.state == ClientState.OFFLINE:
            self._offline_buffer.append(op)
            return {"status": "queued_offline", "op_id": op.op_id}

        if self.state == ClientState.SYNCHRONIZED:
            self.state = ClientState.AWAITING
            self._in_flight = op
            result = self.server.receive_operation(op, self.last_seq, self.user_id)
            return result
        else:
            # Buffering mode
            self._pending.append(op)
            return {"status": "buffered", "op_id": op.op_id}

    def receive_server_op(self, server_op: Operation):
        """Server pushes an operation from another client."""
        if server_op.user_id == self.user_id:
            # Acknowledgment of our own op
            self.last_seq = server_op.seq_num
            if self._pending:
                next_op = self._pending.pop(0)
                self._in_flight = next_op
                self.server.receive_operation(next_op, self.last_seq, self.user_id)
            else:
                self._in_flight = None
                self.state = ClientState.SYNCHRONIZED
        else:
            # Another user's op — transform our pending ops against it
            if self._in_flight:
                self._in_flight = OperationalTransform.transform(self._in_flight, server_op)
            self._pending = [
                OperationalTransform.transform(p, server_op) for p in self._pending
            ]
            # Apply to local document
            self._local_doc.apply_operation(server_op)
            self.last_seq = server_op.seq_num

    def go_offline(self):
        self.state = ClientState.OFFLINE

    def reconnect(self):
        """Replay offline buffer on reconnect."""
        self.state = ClientState.SYNCHRONIZED
        buffered = self._offline_buffer[:]
        self._offline_buffer = []
        results = []
        for op in buffered:
            result = self.type_operation(op)
            results.append(result)
        return results

    def get_local_content(self) -> str:
        return self._local_doc.get_content()


# ---------------------------------------------------------------------------
# Collaborative Document Server
# ---------------------------------------------------------------------------

class CollaborativeDocument:
    """
    Server-side document: authoritative OT, total ordering, broadcasting.
    """

    def __init__(self, doc_id: str, initial_content: str = ""):
        self.doc_id = doc_id
        self._doc = DocumentState(initial_content)
        self._history = DocumentHistory(doc_id)
        self._collaborators = CollaboratorManager()
        self._clients: dict[str, CollaborativeClient] = {}
        self._server_ops: list[Operation] = []

    def add_collaborator(self, user_id: str, name: str) -> CollaborativeClient:
        self._collaborators.join(user_id, name)
        client = CollaborativeClient(user_id, name, self)
        self._clients[user_id] = client
        # Sync client to current document state
        content, seq_num = self._history.load_document()
        client._local_doc = DocumentState(content)
        client.last_seq = seq_num
        return client

    def receive_operation(self, op: Operation, client_seq: int, user_id: str) -> dict:
        """
        Core OT: transform op against all server ops the client hasn't seen.
        Then apply, assign seq_num, broadcast.
        """
        # Get ops client hasn't seen
        missed_ops = self._history.get_ops_since(client_seq)

        # Transform incoming op against each missed op
        transformed_op = op
        for server_op in missed_ops:
            if server_op.user_id != user_id:
                transformed_op = OperationalTransform.transform(transformed_op, server_op)

        # Apply to authoritative document
        self._doc.apply_operation(transformed_op)

        # Record in history
        seq_num = self._history.append_operation(transformed_op)
        transformed_op.seq_num = seq_num

        # Update cursor position
        if transformed_op.op_type == OpType.INSERT:
            new_cursor = transformed_op.position + len(transformed_op.content)
        else:
            new_cursor = transformed_op.position
        self._collaborators.update_cursor(user_id, new_cursor)

        # Broadcast to all clients
        self._broadcast(transformed_op)

        return {
            "status": "accepted",
            "seq_num": seq_num,
            "op_id": op.op_id
        }

    def _broadcast(self, op: Operation):
        """Fan out operation to all connected clients."""
        for uid, client in self._clients.items():
            client.receive_server_op(op)

    def get_document_state(self) -> dict:
        content, seq_num = self._history.load_document()
        return {
            "doc_id": self.doc_id,
            "content": content,
            "seq_num": seq_num,
            "length": len(content),
            "active_collaborators": self._collaborators.active_count()
        }

    def get_cursors(self) -> list[dict]:
        return self._collaborators.get_active_cursors()


# ---------------------------------------------------------------------------
# Demo / Simulation
# ---------------------------------------------------------------------------

def run_simulation():
    print("=" * 65)
    print("  Google Docs Collaborative Editing Simulation")
    print("=" * 65)

    # Create document
    doc = CollaborativeDocument("doc_001", initial_content="Hello World")
    print(f"\nInitial document: '{doc._doc.get_content()}'")

    # Add collaborators
    alice = doc.add_collaborator("alice", "Alice")
    bob = doc.add_collaborator("bob", "Bob")
    carol = doc.add_collaborator("carol", "Carol")
    print(f"Collaborators: Alice, Bob, Carol")
    print(f"Active collaborators: {doc._collaborators.active_count()}")

    # --- Scenario 1: Sequential edits ---
    print("\n[1] Sequential edits (no conflict)")
    op1 = Operation(OpType.INSERT, position=5, content=",")
    result1 = alice.type_operation(op1)
    print(f"    Alice inserts ',' at pos 5 -> seq={result1.get('seq_num', '?')}")
    print(f"    Document: '{doc._doc.get_content()}'")

    op2 = Operation(OpType.INSERT, position=12, content="!")
    result2 = bob.type_operation(op2)
    print(f"    Bob inserts '!' at pos 12 -> seq={result2.get('seq_num', '?')}")
    print(f"    Document: '{doc._doc.get_content()}'")

    # --- Scenario 2: Concurrent inserts (OT in action) ---
    print("\n[2] Concurrent inserts — OT Transform")
    doc2 = CollaborativeDocument("doc_002", initial_content="The cat sat")
    user_a = doc2.add_collaborator("user_a", "UserA")
    user_b = doc2.add_collaborator("user_b", "UserB")
    print(f"    Start: '{doc2._doc.get_content()}'")

    # Both users see "The cat sat" and make concurrent ops
    # Simulate: user_a inserts " big" at pos 7, user_b inserts " fat" at pos 7
    op_a = Operation(OpType.INSERT, position=7, content=" big")
    op_b = Operation(OpType.INSERT, position=7, content=" fat")

    # Transform op_b given op_a
    transformed_b = OperationalTransform.transform(op_b, op_a)
    print(f"    UserA: Insert ' big' at pos 7")
    print(f"    UserB: Insert ' fat' at pos 7 (concurrent)")
    print(f"    OT transforms B: new position = {transformed_b.position}")

    result_a = user_a.type_operation(op_a)
    result_b = user_b.type_operation(op_b)
    print(f"    Final document: '{doc2._doc.get_content()}'")
    print(f"    Both users see same content: "
          f"{user_a.get_local_content() == user_b.get_local_content()}")

    # --- Scenario 3: Delete + Insert concurrent ---
    print("\n[3] Concurrent Delete (A) + Insert (B)")
    doc3 = CollaborativeDocument("doc_003", initial_content="Hello Beautiful World")
    ua = doc3.add_collaborator("ua", "UA")
    ub = doc3.add_collaborator("ub", "UB")
    print(f"    Start: '{doc3._doc.get_content()}'")

    del_op = Operation(OpType.DELETE, position=6, length=10)   # Delete "Beautiful "
    ins_op = Operation(OpType.INSERT, position=16, content=" Amazing")  # Insert after "Beautiful"

    transformed_ins = OperationalTransform.transform(ins_op, del_op)
    print(f"    UA: Delete 'Beautiful ' (pos 6, len 10)")
    print(f"    UB: Insert ' Amazing' at pos 16")
    print(f"    OT transforms UB insert: new position = {transformed_ins.position}")

    ua.type_operation(del_op)
    ub.type_operation(ins_op)
    print(f"    Final: '{doc3._doc.get_content()}'")

    # --- Scenario 4: Offline editing + reconnect ---
    print("\n[4] Offline editing and reconnect")
    doc4 = CollaborativeDocument("doc_004", initial_content="Draft document")
    dave = doc4.add_collaborator("dave", "Dave")
    eve = doc4.add_collaborator("eve", "Eve")

    dave.go_offline()
    print("    Dave goes offline")

    # Dave types while offline
    dave.type_operation(Operation(OpType.INSERT, position=5, content=" final"))
    dave.type_operation(Operation(OpType.INSERT, position=0, content="[IMPORTANT] "))
    print(f"    Dave types 2 ops offline (buffered: {len(dave._offline_buffer)})")

    # Eve types while Dave is offline
    eve.type_operation(Operation(OpType.INSERT, position=14, content=" v2"))
    print(f"    Eve (online) inserts ' v2': '{doc4._doc.get_content()}'")

    # Dave reconnects
    print("    Dave reconnects — replaying offline ops...")
    replay_results = dave.reconnect()
    print(f"    Replayed {len(replay_results)} ops")
    print(f"    Final document: '{doc4._doc.get_content()}'")
    print(f"    Dave's local view: '{dave.get_local_content()}'")
    print(f"    Consistent: {doc4._doc.get_content() == dave.get_local_content()}")

    # --- Scenario 5: Document state and history ---
    print("\n[5] Document state and snapshot/delta")
    state = doc.get_document_state()
    print(f"    Doc ID         : {state['doc_id']}")
    print(f"    Content        : '{state['content']}'")
    print(f"    Seq num        : {state['seq_num']}")
    print(f"    Content length : {state['length']}")
    print(f"    Collaborators  : {state['active_collaborators']}")

    # --- Scenario 6: Cursor broadcast ---
    print("\n[6] Cursor positions (broadcasted to all)")
    cursors = doc.get_cursors()
    for c in cursors:
        print(f"    {c['user']:<8} | pos={c['position']} | color={c['color']}")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  OT Transform Summary")
    print("=" * 65)
    print("  Insert vs Insert: shift by len(concurrent insertion)")
    print("  Insert vs Delete: shift by overlap of preceding deletion")
    print("  Delete vs Insert: shift delete position right by insertion")
    print("  Delete vs Delete: reduce length by overlap, adjust position")
    print("  All transforms guarantee: apply(doc, a) then transform(b,a)")
    print("    == apply(doc, b) then transform(a,b)  [convergence]")
    print("=" * 65)


if __name__ == "__main__":
    run_simulation()
