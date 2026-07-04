# Model Context Protocol (MCP) Deep Dive

## The Problem MCP Was Built to Solve

Everything covered in the previous two chapters — declaring tools, letting a model call them, executing them, feeding results back — works, but it has a scaling problem that only becomes visible once you're operating more than one AI application against more than one data source. Suppose your organization has a Postgres database, a Slack workspace, a ticketing system, and a file store, and you want an LLM to be able to work with all four. Under the plain function-calling model, every single AI application that wants to use these four systems — your internal chatbot, your IDE's coding assistant, your customer-support agent, a workflow-automation tool — has to write and maintain its own bespoke integration code for each one: its own Postgres client wrapped in a `query_database` tool schema, its own Slack SDK wrapped in a `send_message` tool schema, and so on. If you have `M` applications and `N` data sources, you end up building and maintaining roughly `M × N` bespoke, non-reusable integrations, each with its own bugs, its own auth handling, its own schema conventions, and no way for improvements made in one to benefit the others.

This is precisely the integration problem that USB solved for physical peripherals, and that LSP (the Language Server Protocol) solved for code editors talking to language tooling: before LSP, every editor had to write its own integration for every language's autocomplete, go-to-definition, and diagnostics; after LSP, a language only needs one server implementation, and any LSP-compliant editor can use it for free. The Model Context Protocol, introduced by Anthropic in November 2024 as an open standard and since adopted broadly across the industry (including by OpenAI and Google DeepMind for their own agent products), applies the same idea to LLM applications and the tools/data they need. Instead of `M × N` bespoke integrations, you get `M + N`: each data source or capability is exposed once, as an MCP server, and any MCP-compliant application (an MCP host) can talk to it using the same protocol, with no per-application custom code on the server side and no per-server custom code on the client side.

This is the single most important thing to say correctly in an interview about MCP: it is not a new way to do function calling. The model still emits exactly the same kind of structured tool-call object it always has, and your code still executes something and feeds a result back — nothing about the fundamentals from the previous two chapters changes. What MCP standardizes is everything *around* that: how a tool gets discovered, how its schema gets fetched, how the connection to whatever's actually executing the tool is established and authenticated, and — going beyond what plain function calling ever addressed — how non-tool context like files, database rows, or documents gets exposed to the model in a standardized way too.

## Architecture: Hosts, Clients, and Servers

MCP defines three architectural roles, and keeping them straight is the most common point of confusion, because "client" and "server" here don't map cleanly onto typical web-development intuition.

The **host** is the actual AI application the end user interacts with — Claude Desktop, an IDE like Cursor or VS Code with an AI extension, a custom internal chatbot, an agent framework. The host is what owns the conversation with the LLM and decides, ultimately, which MCP servers to connect to and when to expose their capabilities to the model.

The **client** lives inside the host, and there is one client instance per server connection — if your host is connected to three MCP servers, it's running three client instances, each maintaining a dedicated, stateful, one-to-one connection to exactly one server. The client's job is protocol mechanics: it performs the initial handshake, requests the server's capability lists (which tools, resources, and prompts it offers), forwards invocation requests from the host, and relays responses back. Critically, a single client-server connection is always 1:1 — an MCP client does not multiplex across multiple servers, which is precisely why a host that wants to talk to N servers instantiates N separate clients.

The **server** is the thing that actually knows how to do something useful: a lightweight, usually independently-deployable process that wraps a specific capability or data source (a database, a filesystem, a SaaS API, a search index) and exposes it via the MCP primitives described below. A server has no idea what host it's talking to, doesn't know or care about the LLM, and doesn't manage the conversation — it just answers protocol requests ("what tools do you have," "run this tool with these arguments," "here's a list of resources you can read") and returns structured results. This separation is deliberate: it means a single server implementation — say, a Postgres MCP server someone writes once — can be dropped into Claude Desktop, into a custom agent, into an IDE, into anything, with zero changes to the server itself.

```
+--------------------------------------------------+
|                     HOST                          |
|   (Claude Desktop / IDE / custom agent app)       |
|                                                    |
|   +-----------+   +-----------+   +-----------+   |
|   | MCP Client|   | MCP Client|   | MCP Client|   |
|   |  (1:1)    |   |  (1:1)    |   |  (1:1)    |   |
|   +-----+-----+   +-----+-----+   +-----+-----+   |
+---------|---------------|---------------|---------+
          |               |               |
     JSON-RPC 2.0     JSON-RPC 2.0    JSON-RPC 2.0
     (stdio / HTTP)   (stdio / HTTP)  (stdio / HTTP)
          |               |               |
   +------v-----+  +------v-----+  +------v------+
   | MCP Server |  | MCP Server |  | MCP Server  |
   | (Postgres) |  | (Slack)    |  | (Filesystem)|
   +------------+  +------------+  +-------------+
```

It's worth explicitly noting where the LLM sits in this picture: nowhere in the diagram above, and that's correct. The LLM is a separate component the host talks to directly (via whatever inference API it uses). MCP governs the host's relationship with the outside world of tools and data; the host is still fully responsible for deciding what to send the model, interpreting the model's tool-call output, and routing that output to the appropriate MCP client. MCP is the standardized plumbing on one side of the host; the model API is the (separately standardized, or not) plumbing on the other side.

## Transport: JSON-RPC 2.0 Underneath Everything

Every message exchanged between an MCP client and server is a JSON-RPC 2.0 message — a well-established, minimal RPC format that predates MCP by two decades and was chosen precisely because it's simple, language-agnostic, and doesn't require inventing new wire semantics. Every request has a `method` name, an `id` for correlating the eventual response, and a `params` object; every response carries either a `result` or an `error` keyed to that same `id`; and JSON-RPC also supports one-way `notifications` (no `id`, no response expected) which MCP uses for things like progress updates during a long-running tool call.

MCP layers its own vocabulary of methods on top of bare JSON-RPC — things like `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`, and an `initialize` handshake method that clients and servers exchange first to negotiate protocol version and advertise which capabilities each side supports (a server that doesn't implement resources, for instance, simply omits that capability during initialization, and a well-behaved client won't attempt `resources/list` against it).

What's transport-agnostic — deliberately — is *how* those JSON-RPC messages physically travel between client and server. MCP currently standardizes two transports. **stdio** is used when the server runs as a local subprocess of the host: the client writes JSON-RPC requests to the child process's stdin and reads responses from its stdout, newline-delimited. This is the common case for local integrations — a filesystem server, a local database, a local git repo — and it has the enormous practical advantage of needing zero network configuration or authentication scaffolding, since the "auth" is just "you're allowed to spawn processes on this machine." **Streamable HTTP** (the successor to an earlier HTTP+Server-Sent-Events transport) is used when the server is a separately hosted remote service — a SaaS product exposing an MCP server to its customers, for instance — and layers standard HTTP semantics (including OAuth-based authorization) on top, with the server able to push messages back to the client over the same connection.

The practical implication of this split is that the same server *logic* can, in principle, be exposed over either transport with only the transport-adapter layer changing — the tool/resource/prompt handlers themselves don't need to know or care whether they were invoked over a stdin pipe or an HTTP request. This is intentional protocol design, not an accident: it's what lets a hobbyist run a filesystem server locally over stdio with no auth story at all, while a company runs the equivalent-in-spirit Salesforce integration as a hosted, OAuth-protected HTTP server for its whole customer base, using the same underlying protocol both times.

## The Three Core Primitives

MCP defines exactly three kinds of things a server can expose, and the distinction between them is one of the most interview-relevant details of the whole protocol, because it's easy to conflate all three with "tools" if you've only ever thought about function calling.

**Tools** are model-invoked actions — the direct analog of what chapters 1 and 2 covered. A tool has a name, a description, and a JSON Schema for its input, exactly like an OpenAI or Anthropic function definition, and the host is expected to surface an MCP server's tools to the LLM as callable functions. The key word is *model-invoked*: the AI decides, based on the conversation, whether and when to call a tool, the same autonomous decision-making covered in Chapter 1.

**Resources** are application-controlled, addressable pieces of context — a file's contents, a database row, a Slack channel's recent messages, a URL's content — each identified by a URI (`file:///home/user/notes.txt`, `postgres://mydb/customers/42`, `slack://channel/C123/history`). Resources are not something the model decides to invoke the way it invokes a tool; they're something the *host application* decides to fetch and attach to the model's context, often based on direct user action (a user in an IDE attaching a specific file to their chat) rather than model autonomy. A server exposing resources typically supports listing them (`resources/list`, possibly with pagination) and reading a specific one (`resources/read`), and can optionally notify subscribed clients when a resource's content changes, which matters for keeping long-running sessions in sync with a live data source.

**Prompts** are reusable, server-defined prompt templates — parameterized message sequences designed to be surfaced to the *user*, typically as something like a slash command in the host's UI (`/summarize-pr`, `/explain-error`), rather than something the model or the application invokes automatically. A server can declare a prompt with named arguments; the client fetches its `prompts/list`, the host exposes those as discoverable actions in its UI, and when the user invokes one, the client calls `prompts/get` with the supplied arguments and receives back a fully-formed sequence of messages to inject into the conversation.

The clean way to hold these three apart is by asking "who decides this happens": tools are invoked by the **model**, resources are attached by the **application** (often via direct user action), and prompts are triggered by the **user** through some explicit UI affordance the host provides. All three are optional — a server can implement just tools, just resources, just prompts, or any combination, and advertises which it supports during the `initialize` handshake.

## A Minimal MCP Server, Conceptually

The official Python SDK (`mcp`, often used via its `FastMCP` convenience layer) makes the shape of a real server look almost identical to writing a plain Python function, with decorators doing the work of generating the JSON Schema and wiring up the JSON-RPC dispatch:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-server")

@mcp.tool()
def get_current_weather(city: str, unit: str = "fahrenheit") -> dict:
    """Get current weather conditions for a city.

    Args:
        city: City name, e.g. 'Austin, TX'.
        unit: Temperature unit, 'celsius' or 'fahrenheit'.
    """
    # Real implementation would call a weather API here.
    return {"city": city, "temperature": 89, "unit": unit, "condition": "sunny"}

@mcp.resource("weather://stations/{station_id}")
def get_station_metadata(station_id: str) -> str:
    """Expose static metadata about a weather station as a readable resource."""
    return f"Station {station_id}: elevation 550m, type=automated"

@mcp.prompt()
def weather_report_prompt(city: str) -> str:
    """A reusable prompt template a user can invoke, e.g. as a slash command."""
    return f"Write a friendly, concise weather report for {city} using the get_current_weather tool."

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Under the hood, `mcp.run(transport="stdio")` starts a loop reading JSON-RPC requests from stdin. When a client sends `initialize`, the server responds with its protocol version and capabilities (it supports tools, resources, and prompts, in this example). When the client sends `tools/list`, the server introspects `get_current_weather`'s signature and docstring and returns a tool definition whose JSON Schema was generated automatically from the Python type hints — `city: str` becomes `{"type": "string"}`, the `unit` default becomes a schema default, and so on. When the client sends `tools/call` with `{"name": "get_current_weather", "arguments": {"city": "Denver"}}`, the server invokes the actual Python function and returns its result wrapped in the protocol's response envelope.

## What the End-to-End Interaction Looks Like

Tying the architecture and the primitives together, here's the sequence of events when a user asks a host application (say, a chat UI backed by an LLM) a question that requires a tool living behind an MCP server:

1. **Startup / handshake.** The host launches (or connects to) the weather MCP server and its client performs the `initialize` handshake, exchanging protocol versions and capability flags.
2. **Discovery.** The client calls `tools/list` (and, if supported, `resources/list` and `prompts/list`). The server responds with its `get_current_weather` tool definition, including the auto-generated JSON Schema.
3. **Exposure to the model.** The host takes that MCP tool definition and translates it into whatever tool-calling format its LLM provider expects — an OpenAI-style `{"type": "function", "function": {...}}` block or an Anthropic-style tool definition — and includes it in the next request to the model, alongside the user's message. This translation step is the host's responsibility; MCP doesn't dictate which LLM API or wire format the host uses downstream.
4. **Model decision.** Exactly as in Chapter 1, the model reads the user's question ("what's the weather in Denver?"), decides `get_current_weather` is relevant, and emits a structured tool call with `{"city": "Denver"}`.
5. **Routing to the client.** The host recognizes this tool call corresponds to a tool advertised by the weather MCP server, and routes it to that server's client instance rather than executing it directly.
6. **Invocation over MCP.** The client sends a `tools/call` JSON-RPC request to the server over stdio (or HTTP), with the model-supplied arguments as `params`.
7. **Server execution.** The server runs the actual `get_current_weather` Python function (which might itself call a real weather API), and returns the result as a JSON-RPC response.
8. **Result injection.** The host takes that result and feeds it back into the conversation as a tool-result message, exactly as described in Chapter 1 — MCP doesn't change this step at all, it only changed how the tool was discovered and invoked.
9. **Final answer.** The model, now with the weather data in context, produces its natural-language reply, which the host displays to the user.

The thing to notice is that steps 4, 8, and 9 are completely unchanged from plain function calling — MCP's entire contribution is standardizing steps 1, 2, 3, 5, and 6: how the tool got discovered and how the call got routed and executed, in a way that's identical regardless of which MCP server is providing the tool or which host application is consuming it.

## MCP vs. Ad Hoc Function Calling: What Actually Changes

The most defensible framing for an interview answer is that MCP doesn't add new *capabilities* the model didn't already have via function calling — a model calling a tool through MCP and a model calling a hand-rolled Python function do fundamentally the same kind of thing from the model's point of view. What MCP changes is the *engineering economics* around building and maintaining those integrations, along a few concrete axes.

**Reusability.** A Postgres MCP server written once can be used, unmodified, by Claude Desktop, a custom LangGraph agent, an IDE assistant, and any future MCP-compliant tool, whereas a hand-rolled `query_database` function tied to one codebase's function-calling setup has to be re-implemented (or at best copy-pasted and adapted) for every new application that wants the same capability.

**Discoverability.** Because `tools/list`, `resources/list`, and `prompts/list` are standardized calls every server must answer the same way, a host application can connect to an MCP server it has never seen before and immediately learn everything it offers, with no prior configuration beyond "here's how to reach this server." Ad hoc function calling has no equivalent notion of a server advertising its own capabilities at runtime — the tool schema is hardcoded into the calling application in advance.

**Separation of concerns.** The people who best understand a given data source (say, the team that owns the internal ticketing system) can write and own the MCP server for it once, independent of which AI applications will eventually consume it and independent of which LLM vendor those applications happen to use. Under ad hoc function calling, that same team would need to either hand-write integration code for every consuming application or publish a library that each application's engineers then have to individually wire into their own function-calling setup.

**Standardized non-tool context.** Plain function calling has no native concept of resources or prompts — everything has to be jammed into the tool-calling paradigm even when "let the model decide to fetch this" isn't actually the right shape for the interaction (e.g., "attach the file the user just opened" is naturally application-driven, not model-driven). MCP's three-primitive model gives you the right mechanism for each kind of interaction instead of overloading tools for everything.

What MCP does *not* give you: it doesn't make tool selection more reliable (that's still entirely governed by the schema-design principles in Chapter 2 — an MCP server with a badly-written tool description is exactly as confusing to a model as a hand-rolled one), it doesn't make execution automatically safe (an MCP tool that shells out to `rm -rf` is just as dangerous as a hand-rolled one — see Chapter 4), and it doesn't eliminate the need for a host to translate between MCP's tool representation and whatever specific LLM API it's actually calling, since MCP standardizes the client-server side, not the host-to-model side.

## Reading the Wire Protocol Directly

It's worth seeing the raw JSON-RPC traffic at least once, because it demystifies what the SDK decorators in the previous section are actually doing on your behalf. The handshake begins with the client sending an `initialize` request declaring the protocol version it speaks and what it supports:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": { "roots": { "listChanged": true }, "sampling": {} },
    "clientInfo": { "name": "my-host-app", "version": "1.0.0" }
  }
}
```

The server answers with its own protocol version and the capabilities it actually implements — note that a server which only offers tools, with no resources or prompts, simply omits those keys, and a well-behaved client is expected to check for their presence before ever calling `resources/list` or `prompts/list` against it:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-06-18",
    "capabilities": { "tools": { "listChanged": true } },
    "serverInfo": { "name": "weather-server", "version": "0.3.0" }
  }
}
```

Discovery and invocation follow the same request/response shape. `tools/list` returns the schema the SDK generated from the Python function signature and docstring in the earlier example:

```json
{
  "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "get_current_weather",
        "description": "Get current weather conditions for a city.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "city": { "type": "string" },
            "unit": { "type": "string", "default": "fahrenheit" }
          },
          "required": ["city"]
        }
      }
    ]
  }
}
```

And `tools/call` is where the host's client actually invokes the tool with the arguments the LLM supplied, receiving back a result envelope that distinguishes a normal return value from a tool-level error via `isError`, separately from a transport-level JSON-RPC error (which would indicate something went wrong with the protocol call itself, not with the tool's own logic):

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": { "name": "get_current_weather", "arguments": { "city": "Denver" } }
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [{ "type": "text", "text": "{\"city\": \"Denver\", \"temperature\": 89, \"unit\": \"fahrenheit\"}" }],
    "isError": false
  }
}
```

Everything the `FastMCP` decorators and the client SDK do is generate exactly this traffic and parse it back into native objects. Understanding the raw shape matters in practice because it's what you're actually debugging when a server misbehaves — a server returning `isError: true` with a human-readable message in `content` is behaving correctly even though the tool "failed," whereas a malformed JSON-RPC envelope (a missing `id`, a `result` and `error` both present) is a protocol-level bug in the server implementation itself, and the two categories call for very different fixes.

## Two Primitives Often Left Out of a First-Pass Explanation: Roots and Sampling

Most introductions to MCP stop at tools, resources, and prompts, but the specification defines two additional, client-side capabilities that flip the usual direction of the relationship and are worth knowing about precisely because they're easy to miss.

**Roots** let the *client* tell a server which directories or URIs it's allowed to operate within — for instance, a host application can declare "you may only operate on files under `/Users/alice/projects/current-repo`," and a well-behaved filesystem server is expected to respect that boundary rather than assuming it has free rein over the whole filesystem. This is a cooperative scoping mechanism, not a hard security guarantee (a malicious server can simply ignore it, which is why it doesn't replace the sandboxing discipline covered in Chapter 4), but for well-behaved servers it lets a host constrain scope declaratively instead of relying on the server author to have hardcoded the right restriction.

**Sampling** is the more architecturally interesting of the two: it lets a *server* request that the client run an LLM completion on its behalf and return the result, effectively borrowing the host's model access rather than needing its own API key and billing relationship with a model provider. Consider a server that summarizes long documents as part of fulfilling a tool call — without sampling, that server would need its own LLM credentials, its own choice of model, and its own cost line; with sampling, it sends a `sampling/createMessage` request back through the very client connection that invoked it, the host mediates that request (typically requiring user approval, since it's real model spend happening outside the main conversation), runs the completion using whatever model the host has configured, and hands the text back to the server to finish its work. This inverts the usual client-calls-server direction and is what lets MCP servers be genuinely portable across hosts using different model providers, without embedding a hardcoded assumption about which LLM vendor is available.

## Capability Negotiation and Protocol Versioning

MCP versions itself with a date-stamped protocol version string (visible in the `initialize` exchange above), and the specification is explicit that clients and servers may implement different versions and must negotiate down to a mutually understood one during the handshake rather than assuming lockstep upgrades across every host and server in an ecosystem. This matters in practice because it's exactly what lets the ecosystem evolve without a flag day — a new client rolled out with support for a newer protocol revision can still talk to an older, unmaintained server by falling back to the version that server declares support for, and a server author is not forced to chase every client's release schedule to remain usable. The same negotiation pattern covers individual capabilities within a version: a server can support tools without resources, or resources without the `listChanged` notification capability, and the client is expected to check the specific capability flags returned during `initialize` before assuming a given piece of protocol surface is available, rather than hardcoding "this server obviously supports X."

## MCP, Plain Function Calling, and a Bespoke REST Integration, Side by Side

A concrete comparison sharpens exactly where MCP sits relative to the two alternatives an engineer is choosing between in practice:

| Aspect | Hand-rolled REST integration | Ad hoc function calling (Ch. 1–2) | MCP |
|---|---|---|---|
| Who defines the interface | Each app, per API, from scratch | Each app, per tool, from scratch | The server author, once |
| Discoverability at runtime | None — read the docs | None — hardcoded in app code | `tools/list`, `resources/list`, `prompts/list` |
| Reusable across AI applications | No | No | Yes, by any MCP-compliant host |
| Handles non-tool context (files, DB rows) | Ad hoc, per app | Not natively — forced into tool shape | Native, via resources |
| Transport | Whatever the API uses (usually HTTP) | N/A (in-process function call) | Standardized: stdio or streamable HTTP |
| Auth model | Per API, bespoke | Handled entirely by your app code | Standardized OAuth flow for remote HTTP servers |
| Where reliability comes from | Your integration code's quality | Your schema design (Ch. 2) | Same schema-design burden — MCP doesn't remove it |

The last row is worth dwelling on because it's the detail people most often get wrong under interview pressure: MCP is not a reliability technology. A `get_customer` tool exposed via an MCP server with a one-word description is exactly as likely to be misused by the model as the same tool hand-rolled directly into an application — every principle from Chapter 2 about naming, description clarity, and avoiding overlapping tools applies identically whether the tool's schema arrived over JSON-RPC from a remote server or was written inline in your own codebase.

## Operational Realities: Running Many Servers From One Host

A host that wants broad capability ends up connecting to a double-digit number of MCP servers — a filesystem server, a Git server, a database server, several SaaS integrations — and this creates operational concerns that don't show up in a single-server tutorial. Every connected server's full tool list is a candidate for injection into the model's context on every turn, which reintroduces the exact "too many tools" problem discussed in Chapter 2, now compounded by the fact that the tool catalog isn't even fully under the host's own control — it depends on whichever servers happen to be connected in a given deployment. Mature hosts address this by applying the same dynamic tool-selection and namespacing discipline from Chapter 2 to the aggregate of all connected servers' tools, not just to a single application's hand-written tool list, and by allowing per-server enable/disable toggles so a user or administrator can prune the effective catalog down to what a given session actually needs. Connection lifecycle is a second concern: stdio-based local servers are subprocesses of the host and need to be spawned, health-checked, and cleaned up on host shutdown or crash, while remote HTTP servers need connection pooling, retry/backoff on transient network failures, and re-authentication handling when an OAuth token expires mid-session — none of which is visible in a minimal single-server example but all of which is ordinary distributed-systems plumbing a host has to own once it's operating multiple long-lived server connections concurrently.

## Security and Trust Considerations Specific to MCP

Because MCP explicitly optimizes for making it easy to plug arbitrary third-party servers into a host, it introduces a trust surface that ad hoc, single-codebase function calling didn't have in the same way: you can now point your AI application at a server you didn't write, don't control, and can't fully audit, and to the host it looks exactly as legitimate as a server you built in-house. A malicious or compromised MCP server can return a tool description crafted to manipulate the model's behavior (a form of prompt injection embedded directly in the "trusted" tool schema itself, sometimes called tool poisoning), can return resource content laced with hidden instructions, or can silently change its tool descriptions between the time a user approved it and a later invocation ("rug-pull" style attacks). This is exactly why the specification includes explicit human-in-the-loop consent points — a compliant host is expected to prompt the user before letting a newly connected server's tools be invoked, and again if a tool's declared behavior changes — and why production deployments generally pin server versions, vet third-party servers before connecting to them, and apply the same execution-sandboxing discipline described in Chapter 4 regardless of whether a tool arrived via MCP or via a hand-rolled integration. MCP standardizes the plumbing; it does not, by itself, make anything on the other end of that plumbing trustworthy.

## Where MCP Sits in an Interview Answer

If asked to summarize MCP concisely, the strongest answer threads together: it's an open protocol (JSON-RPC 2.0 based, over stdio or streamable HTTP) that standardizes how AI applications (hosts, via per-connection clients) discover and use external capabilities exposed by independent servers; it defines three primitives — tools (model-invoked actions), resources (application-attached context), and prompts (user-triggered templates); and its value is architectural — turning an M-times-N integration problem into an M-plus-N one — rather than a change to the fundamental mechanics of how a model requests an action and receives a result, which remain exactly what they were under plain function calling.
