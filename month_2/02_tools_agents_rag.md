# Tools, Agents, and RAG at Scale — Full Course Module

## Module Overview
This module takes learners from single-turn prompts to production-grade systems that use tools, act autonomously, and retrieve grounded knowledge. The emphasis is on architecture patterns, error handling, observability, and measurable quality.

## Target Audience
- Developers and technical professionals
- Comfortable with Python, APIs, and basic LLM concepts (Month 1 foundations)

## Learning Objectives
By the end of this module, learners will be able to:
- Design and register tool schemas for function calling
- Build a multi-step ReAct agent with guardrails and state management
- Implement a RAG pipeline with chunking, embeddings, and vector search
- Add reranking, hybrid search, and citation grounding for quality
- Evaluate a RAG system with offline and online metrics

---

## Prerequisites
- Month 1: LLM architecture, tokenization, decoding strategies, prompting fundamentals
- Python 3.10+
- Access to an LLM API (OpenAI, Anthropic, or local via Ollama)
- Basic familiarity with REST APIs and JSON

---

## Module Structure

| Module | Topic | Lab |
|--------|-------|-----|
| 2.1 | Tool Use and Function Calling | Tool-using assistant with calculator + search |
| 2.2 | Agent Patterns | Multi-step agent with plan → act → verify |
| 2.3 | Retrieval-Augmented Generation | Basic RAG pipeline over a document set |
| 2.4 | Reranking and Grounding | Reranking + citations on RAG outputs |
| Mini-Project | Production-grade RAG with evaluation harness | End-to-end system |

---

# Module 2.1: Tool Use and Function Calling

## Core Concepts

### 1. What is Function Calling?

Function calling (also called tool use) is the ability for an LLM to request the execution of external functions. Instead of generating plain text, the model outputs a structured JSON object describing which function to call and with what arguments. Your application executes the function and returns the result to the model.

**The big idea:** The LLM is great at understanding intent and generating language, but it cannot do math reliably, cannot look up real-time data, cannot send emails, and cannot access your database. Function calling bridges that gap — the LLM *decides* what action to take, and your code *executes* it.

```
User: "What's the weather in Nairobi?"

        |
        v
LLM decides: this requires a function call
        |
        v
LLM outputs: {
  "function": "get_weather",
  "args": {"city": "Nairobi"}
}
        |
        v
Your app calls get_weather("Nairobi") -> returns "22°C, partly cloudy"
        |
        v
LLM receives the result and generates:
"The weather in Nairobi is 22°C and partly cloudy."
```

**Analogy:** Think of the LLM as a smart manager who knows what needs to be done but cannot do the technical work themselves. When they need a calculation, they hand it to an accountant. When they need data, they hand it to a researcher. The manager (LLM) decides *what* to do and delegates *how* to the tools.

**Key insight:** The LLM does not *execute* the function. It *requests* the function call. Your application code is responsible for executing it safely and returning the result. This separation is important for security — you control what runs.

### 2. The Full Function Calling Flow

Here is the complete flow from user message to final response:

```
Step 1: User sends a message
Step 2: System appends user message to conversation history
Step 3: Send entire conversation + tool definitions to LLM
Step 4: LLM responds in one of two ways:
   a) "stop" — generates a text response, you return it to the user
   b) "tool_calls" — requests one or more function calls
Step 5: If tool_calls, execute each tool with the provided arguments
Step 6: Append tool results to conversation history as "tool" role messages
Step 7: Send updated conversation back to LLM (repeat from Step 4)
Step 8: Continue until LLM returns "stop" or max iterations reached
```

```
              ┌─────────────────────────────────────────────────────┐
              │                  Conversation                       │
              │  [user: "What's the weather?"]                     │
              └─────────────────────────────────────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────────────────────────┐
              │  LLM Call (system prompt + tools schema)           │
              └─────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
           finish_reason="stop"    finish_reason="tool_calls"
                    │                       │
                    │                       ▼
                    │           ┌───────────────────────┐
                    │           │ Execute each tool     │
                    │           │ with parsed args      │
                    │           └───────────────────────┘
                    │                       │
                    │                       ▼
                    │           ┌───────────────────────┐
                    │           │ Append tool results    │
                    │           │ to conversation        │
                    │           └───────────────────────┘
                    │                       │
                    │                       └───── back to LLM ────┐
                    ▼                                               │
              ┌─────────────────────────────────────────────────────┘
              │
              ▼
     Return final response to user
```

**Why this loop matters:** The LLM might need multiple tool calls in sequence to answer a question. For example:
- User: "Book me a flight from Nairobi to London under $500"
- Round 1: LLM calls `search_flights(Nairobi, London)`
- Round 2: LLM sees results, calls `filter_by_price(flights, max=500)`
- Round 3: LLM presents options to user

Each round is one LLM call. The conversation grows with each tool result, giving the LLM full context of what happened.

### 3. Tool Schema Design

Every tool must be described to the model with a schema. The schema tells the model:
- What the tool does (description)
- What parameters it accepts (name, type, description, required)
- What the tool returns

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city. Use this when the user asks about weather, temperature, or climate conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. Nairobi, Lagos, Accra"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit. Defaults to celsius.",
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

**Schema design rules — explained:**

| Rule | Why it matters | Example |
|------|---------------|---------|
| Descriptions must be clear | The model reads them to decide when to use the tool | Bad: `"Get weather"` → Good: `"Get the current weather for a city. Use when user asks about weather."` |
| Parameter names self-documenting | Helps the model fill arguments correctly | Bad: `"a"` → Good: `"city"` |
| Required vs optional | Only mark truly required params | `city` is required, `units` is optional with default |
| Use enums for constrained values | Prevents the model from inventing invalid values | `enum: ["celsius", "fahrenheit"]` instead of free-text |
| Parameter descriptions are critical | The model uses these to understand what to pass | `"The city name, e.g. Nairobi, Lagos"` gives the model examples |

**Parameter types supported:**
- `string` — text values
- `number` — float or integer
- `integer` — whole numbers
- `boolean` — true/false
- `array` — lists of items
- `object` — nested JSON

**Full schema example with multiple parameter types:**
```python
{
    "name": "send_email",
    "description": "Send an email to one or more recipients. Use this when the user asks to send a message via email.",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of recipient email addresses"
            },
            "subject": {
                "type": "string",
                "description": "Email subject line"
            },
            "body": {
                "type": "string",
                "description": "Email body content. Can include plain text or markdown."
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "Email priority level"
            }
        },
        "required": ["to", "subject", "body"]
    }
}
```

### 4. Tool Routing and Execution

Tool routing is the logic that dispatches function calls to the correct handler. When the LLM returns a tool call, you need to parse the function name and arguments, then call the right function.

**Basic router:**
```python
def route_tool(name: str, args: dict) -> str:
    if name == "get_weather":
        return get_weather(**args)
    elif name == "calculator":
        return calculator(**args)
    elif name == "search_docs":
        return search_docs(**args)
    elif name == "send_email":
        return send_email(**args)
    else:
        raise ValueError(f"Unknown tool: {name}")
```

**Router with tool registry pattern (better):**
```python
TOOL_REGISTRY = {}

def register_tool(func):
    """Decorator to register a tool handler."""
    TOOL_REGISTRY[func.__name__] = func
    return func

@register_tool
def get_weather(city: str, units: str = "celsius") -> str:
    # implementation
    pass

@register_tool
def calculator(expression: str) -> str:
    # implementation
    pass

def route_tool(name: str, args: dict) -> str:
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")
    return TOOL_REGISTRY[name](**args)
```

**Execution patterns explained:**

| Pattern | How it works | When to use |
|---------|-------------|-------------|
| **Synchronous** | Call function, wait for result, return to model | Simple tools that complete quickly (calculator, search) |
| **Asynchronous** | Queue the function, return a future, continue when ready | Long-running tools (data processing, file generation) |
| **Parallel** | Call multiple independent tools at once, merge results | When the LLM requests multiple tool calls simultaneously |

**Parallel tool calling example:**
```python
# LLM returned TWO tool calls at once
# This is common when the question needs multiple independent data points

import asyncio

async def execute_tool_calls(tool_calls):
    """Execute multiple independent tool calls in parallel."""
    async def call_one(tc):
        name = tc.function.name
        args = json.loads(tc.function.arguments)
        return await route_tool_async(name, args)
    
    # Run all tool calls concurrently
    results = await asyncio.gather(
        *[call_one(tc) for tc in tool_calls]
    )
    return results

# Example: "What's the weather in Nairobi and the time in London?"
# LLM calls: get_weather(city="Nairobi") AND get_time(city="London")
# Both run in parallel, results come back together
```

### 5. State Management Across Turns

Tools need state — conversation history, accumulated results, user preferences. Without state, each turn is isolated and the model cannot remember earlier results.

**Analogy:** A chef working in a kitchen. The chef needs to remember what ingredients have been prepared, what step of the recipe they are on, and what the customer requested. State is the chef's workspace and notes.

**State management strategies ranked by complexity:**

| Strategy | What it stores | Persistence | Best for |
|----------|---------------|-------------|----------|
| In-memory dict | Messages + tool results | Lost on restart | Prototyping, single-user |
| Redis / KV store | Messages + tool results | Persistent | Production, multi-user |
| Database | Messages + tool results + audit logs | Persistent | Production, compliance |

```python
# Simple in-memory state — good for learning
session_state = {
    "session_id": "abc123",
    "messages": [],
    "tool_results": {},
    "user_context": {
        "timezone": "EAT",
        "preferred_units": "metric",
        "name": "Victor"
    }
}

# Adding a message
session_state["messages"].append({
    "role": "user",
    "content": "What's the weather in Nairobi?"
})

# Adding a tool result
session_state["messages"].append({
    "role": "tool",
    "tool_call_id": "call_123",
    "content": "The current weather in Nairobi is 22°C, partly cloudy"
})
```

**The critical role of messages array:**
The `messages` array IS the state for the LLM. Each turn appends:
1. The user's message
2. The assistant's response (text or tool calls)
3. Tool results (one per tool call)

When you call the LLM again, you send the FULL messages array. This is how the model knows what happened in previous steps.

### 6. Error Handling in Tool Calls

Tools fail. Networks time out, APIs return errors, arguments are invalid, databases go down. The agent must handle this gracefully.

**The golden rule: A tool failure should never crash the agent.** The agent should catch the error, inform the LLM, and let the LLM decide what to do next.

```python
def safe_call_tool(name: str, args: dict) -> str:
    """Execute a tool safely. Returns a JSON string the LLM can understand."""
    try:
        result = route_tool(name, args)
        return json.dumps({
            "success": True,
            "result": result
        })
    except KeyError as e:
        # The LLM provided wrong arguments
        return json.dumps({
            "success": False,
            "error": f"Missing or invalid parameter: {e}",
            "hint": "Check the parameter names and types required by this tool"
        })
    except TimeoutError:
        return json.dumps({
            "success": False,
            "error": "Tool timed out after 5 seconds",
            "hint": "You could try a simpler query or check if the service is available"
        })
    except ValueError as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "hint": "The arguments provided were invalid"
        })
    except Exception as e:
        # Catch-all for unexpected errors
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "hint": "Please try again or rephrase your request"
        })
```

**Error recovery strategies in priority order:**

```
                    Tool Error Occurs
                          │
                          ▼
              ┌─────────────────────┐
              │  Retry (up to 3x)   │ ←── Transient errors (timeout, network)
              │  with backoff       │
              └─────────────────────┘
                          │
                     Still failing?
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
    ┌─────────────────────┐  ┌─────────────────────┐
    │  Try fallback tool  │  │  Report to LLM      │ ←── "I tried but failed"
    │  (e.g., bing search │  │  Let LLM decide:    │
    │   if google fails)  │  │  - Apologize        │
    └─────────────────────┘  │  - Ask user         │
              │              │  - Try different    │
              │              │    approach          │
              │              └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Return fallback    │
    │  result or error    │
    └─────────────────────┘
```

### 7. Complete Function Calling Example

Here is a full, runnable example using OpenAI's API pattern:

```python
import json
from openai import OpenAI

client = OpenAI()

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations. Supports +, -, *, /, sqrt, pow.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g. '345 * 789'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current temperature and conditions for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }
]

# Tool implementations
def calculator(expression: str) -> str:
    # SAFETY: eval can be dangerous. This is a simplified example.
    # In production, use a math parser library or a dedicated calculator API.
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: Invalid characters in expression"
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_weather(city: str) -> str:
    # In production, call a real weather API
    weather_data = {
        "nairobi": "22°C, partly cloudy",
        "lagos": "28°C, humid",
        "london": "15°C, light rain",
        "accra": "30°C, sunny"
    }
    result = weather_data.get(city.lower())
    if result:
        return f"The weather in {city} is {result}"
    return f"Sorry, I don't have weather data for {city}"

# The conversation loop
def run_conversation(user_message):
    messages = [{"role": "user", "content": user_message}]
    
    for turn in range(5):  # max 5 turns
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        
        if not msg.tool_calls:
            # LLM is responding in plain text
            return msg.content
        
        # Handle tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            # Execute the tool
            if name == "calculator":
                result = calculator(**args)
            elif name == "get_weather":
                result = get_weather(**args)
            else:
                result = f"Unknown tool: {name}"
            
            # Append to conversation
            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
    
    return "Max turns reached"

# Test it
print(run_conversation("What's 345 * 789 and what's the weather in Nairobi?"))
```

---

## Lab 2.1: Tool-Using Assistant with Calculator + Search

### Goal
Build a CLI assistant that can use a calculator tool and a web-search tool via function calling.

### Steps
1. Define tool schemas for `calculator` and `web_search`
2. Implement the tool handlers with proper error handling
3. Wire up the LLM call → tool routing → response loop
4. Test with prompts like "What's 345 * 789?" and "Search for the latest AI news"
5. Add error handling for invalid inputs and timeouts
6. Log every tool call: what was called, with what args, what result

### Expected Observations
- The model correctly decides when to call a tool vs. respond directly
- Multiple tool calls may be chained (search → compute on result)
- Errors in tool calls should not crash the agent — the LLM recovers gracefully

### Deliverable
A Python script that runs an interactive tool-using session with logging to a file.

---

## Exercises
1. **Schema Design**: Write tool schemas for `send_email`, `lookup_user`, and `translate_text`. Include descriptions, parameter types, enums, and required fields.
2. **Routing**: Implement a registry-based router that dispatches to at least 3 tools with error handling. Use a decorator pattern.
3. **State**: Add conversation history tracking so the model can reference earlier tool results. Test with: "What was the weather in Nairobi? ... What was that temperature in fahrenheit?"

---

# Module 2.2: Agent Patterns

## Core Concepts

### 1. What is an Agent? (The Big Picture)

An agent is an LLM-powered system that can take multi-step actions to achieve a goal. Unlike a single-turn tool call, an agent:
- **Plans**: Decides what steps are needed
- **Executes**: Calls tools one at a time
- **Observes**: Reads tool results
- **Adapts**: Changes its plan based on what it learns
- **Persists**: Keeps going until the goal is reached

**Analogy:** A single tool call is like asking a calculator a math question. An agent is like hiring a research assistant — you give them a goal ("research competitor pricing"), and they figure out the steps (search, compare, analyze, summarize), execute each step, and come back when done.

```
┌──────────────────────────────────────────────────────────────────┐
│                        AGENT LOOP                                │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐              │
│   │  Think   │ →  │   Act    │ →  │  Observe     │ ──┐         │
│   │ (reason  │    │ (call    │    │  (read tool   │   │         │
│   │  about   │    │  tool)   │    │   result)     │   │         │
│   │  next    │    │          │    │              │   │         │
│   │  step)   │    │          │    │              │   │         │
│   └──────────┘    └──────────┘    └──────────────┘   │         │
│       │                                              │         │
│       └──────────────────────────────────────────────┘         │
│                              │                                  │
│                         Goal met?                               │
│                              │                                  │
│                         ┌─────┴──────┐                          │
│                         │            │                          │
│                       Yes           No ─── back to Think        │
│                         │                                        │
│                         ▼                                        │
│                   Return result                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2. The ReAct Pattern (Reason + Act)

ReAct (Reasoning and Acting) is the most widely adopted agent pattern. It was introduced in a 2022 paper by Yao et al. The key insight: **interleaving reasoning traces with actions produces better results than either pure reasoning or pure acting alone.**

**Why it works:**
- The reasoning trace ("Thought: ...") helps the model stay focused
- Each action is grounded in explicit reasoning
- Observations from actions feed back into reasoning
- The model can correct its own mistakes when it sees unexpected results

**Full ReAct trace example:**
```
User: "Find the latest quarterly revenue for Apple and Microsoft, compare them, and write a brief summary."

Thought: I need to find the latest quarterly revenue for both Apple and Microsoft. Let me start by searching for Apple's revenue.

Action: search_web(query="Apple Q1 2025 quarterly revenue")
Observation: Apple reported Q1 2025 revenue of $124.3 billion, up 4% YoY.

Thought: Got Apple's revenue. Now I need Microsoft's revenue for the same period.

Action: search_web(query="Microsoft Q1 2025 quarterly revenue")
Observation: Microsoft reported Q1 2025 revenue of $69.0 billion, up 12% YoY.

Thought: I have both numbers. Apple: $124.3B, Microsoft: $69.0B. Now I need to compose a brief comparative summary.

Action: generate_summary(
    company1="Apple", revenue1="$124.3B",
    company2="Microsoft", revenue2="$69.0B",
    comparison="Apple's revenue is 80% higher, but Microsoft is growing faster at 12% vs 4%"
)
Observation: Summary generated successfully.

Thought: The summary is ready. I'll present it to the user.

Final: Here's the comparison:

Apple reported Q1 2025 revenue of $124.3 billion (+4% YoY), while Microsoft reported $69.0 billion (+12% YoY). Apple's revenue is approximately 80% higher in absolute terms, but Microsoft is growing at three times Apple's rate.
```

**The three ReAct components in detail:**

| Component | Format | Purpose |
|-----------|--------|---------|
| **Thought** | Free-text reasoning | Explains what the agent is thinking, why it's taking this action |
| **Action** | Tool call | The actual function being invoked |
| **Observation** | Tool result | What the tool returned; this is injected by the system, not generated by the LLM |
| **Final** | Free-text answer | The final response to the user when the goal is achieved |

### 3. Implementing the ReAct Loop

**The complete ReAct loop in Python:**

```python
import json

def run_react_agent(
    prompt: str,
    tools: list[dict],
    tool_handlers: dict[str, callable],
    max_steps: int = 10,
) -> str:
    """Run a ReAct agent loop.
    
    Args:
        prompt: The user's initial request
        tools: List of tool schemas to give the LLM
        tool_handlers: Dict mapping tool name -> handler function
        max_steps: Maximum number of thought-action-observation cycles
    
    Returns:
        The agent's final response
    """
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant that can use tools to answer questions. "
            "You should think step by step about what information you need, "
            "use the appropriate tools to gather it, and then provide a final answer. "
            "Format your responses as:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool_name>\n"
            "Action Input: <JSON args>\n"
            "Wait for the observation before continuing."
        )},
        {"role": "user", "content": prompt}
    ]
    
    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        
        response = call_llm(messages, tools=tools)
        content = response.choices[0].message.content
        
        print(f"LLM: {content}")
        
        # Check if the model is done
        if "Final:" in content or response.choices[0].finish_reason == "stop":
            return content.split("Final:")[-1].strip()
        
        # Parse action
        if "Action:" in content:
            # Simple parsing — in production use structured outputs
            action_name = extract_between(content, "Action:", "\n")
            action_input_str = extract_between(content, "Action Input:", "\n")
            
            try:
                action_args = json.loads(action_input_str)
            except json.JSONDecodeError:
                action_args = {"input": action_input_str.strip()}
            
            # Execute tool
            handler = tool_handlers.get(action_name)
            if handler:
                try:
                    result = handler(**action_args)
                except Exception as e:
                    result = f"Error executing tool: {str(e)}"
            else:
                result = f"Error: Unknown tool '{action_name}'"
            
            print(f"  → Tool result: {result[:200]}...")
            
            # Append to conversation
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Observation: {result}"})
        
        else:
            # No action found, ask LLM to continue
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Please continue. What is your next step?"})
    
    return "I was unable to complete the task within the step limit."
```

### 4. Guardrails for Agents

Agents are powerful but can go off-track. Without guardrails, an agent might:
- Loop infinitely on a task it cannot solve
- Execute destructive actions (delete data, send emails to wrong people)
- Spend too much money on API calls
- Generate harmful or unsafe content

**Guardrail layers — defense in depth:**

```
                    User Input
                        │
                        ▼
              ┌─────────────────────┐
              │ INPUT GUARDRAIL     │ ←── Block malicious/off-topic input
              │ • Content filter    │
              │ • Intent check      │
              └─────────────────────┘
                        │
                   Passed check?
                        │
              ┌─────────┴──────────┐
              │                    │
              ▼                   ▼ (blocked)
       ┌─────────────────────┐
       │ AGENT LOOP          │
       │                     │
       │ PROCESS GUARDRAILS: │ ←── Applied at every step
       │ • Max steps (10)    │
       │ • Token budget      │
       │ • Tool allowlist    │
       │ • Rate limiting     │
       └─────────────────────┘
                        │
                   Each tool call
                        │
                        ▼
              ┌─────────────────────┐
              │ ACTION GUARDRAIL    │ ←── Before executing tool
              │ • Arg validation    │
              │ • Destructive?      │ → Require human approval
              │ • Within budget?    │
              └─────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ OUTPUT GUARDRAIL    │ ←── Before returning to user
              │ • PII scan          │
              │ • Content safety    │
              │ • Citation check    │
              └─────────────────────┘
                        │
                        ▼
                  Final Response
```

**Guardrail implementation:**
```python
from dataclasses import dataclass
from enum import Enum

class ToolCategory(Enum):
    READ = "read"
    WRITE = "write" 
    DELETE = "delete"
    DESTRUCTIVE = "destructive"

@dataclass
class GuardrailConfig:
    max_steps: int = 15
    max_tool_calls_per_step: int = 3
    max_tokens_per_session: int = 10000
    require_confirmation: list[str] = None
    blocked_tools: list[str] = None
    allowed_intents: list[str] = None

GUARDRAILS = GuardrailConfig(
    max_steps=15,
    max_tool_calls_per_step=3,
    max_tokens_per_session=10000,
    require_confirmation=["transfer_money", "delete_user", "cancel_order", "send_bulk_email"],
    blocked_tools=["system_shutdown", "drop_database", "exec_shell_command"],
)

def check_action_guardrail(tool_name: str, args: dict) -> bool:
    """Check if an action is allowed. Returns True if approved."""
    if tool_name in GUARDRAILS.blocked_tools:
        return False
    
    if tool_name in GUARDRAILS.require_confirmation:
        print(f"\n⚠️  Tool '{tool_name}' requires confirmation.")
        print(f"   Args: {args}")
        confirm = input("   Proceed? (yes/no): ")
        return confirm.lower() == "yes"
    
    return True
```

**Human-in-the-loop pattern:**
```python
def agent_with_human_approval(prompt):
    """Agent that pauses for human approval before destructive actions."""
    messages = [{"role": "user", "content": prompt}]
    
    for step in range(10):
        response = call_llm(messages, tools=tools)
        
        if response.finish_reason == "stop":
            return response.content
        
        if response.finish_reason == "tool_calls":
            for tc in response.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                
                # Check if this tool needs human approval
                if name in DESTRUCTIVE_TOOLS:
                    print(f"\n🔴 Human approval needed: {name}({args})")
                    approved = input("Approve? (y/n): ")
                    if approved.lower() != "y":
                        result = "Action cancelled by human operator."
                    else:
                        result = execute_tool(name, args)
                else:
                    result = execute_tool(name, args)
                
                messages.append(tc.message)
                messages.append({"role": "tool", "content": result})
    
    return "Max steps reached."
```

### 5. Planning Strategies

**Strategy 1: Simple ReAct (plan as you go)**
```
Thought: I need to find X first
Action: search(X)
Observation: Found Y
Thought: Now I need to use Y to find Z
Action: search(Z)
...
```
Best for: Simple tasks, well-defined steps, when you don't know the full path upfront.

**Strategy 2: Plan-then-execute**
```
// Step 1: Generate full plan
Plan:
  1. Search for quarterly reports
  2. Extract revenue numbers
  3. Compare growth rates
  4. Write summary

// Step 2: Execute each step, verify after each
✓ Step 1 done → got documents
✓ Step 2 done → extracted numbers
✓ Step 3 done → comparison complete
✓ Step 4 done → summary written
```
Best for: Complex tasks where you need a roadmap, verifiable milestones.

**Strategy 3: Hierarchical planning (meta-agent)**
```
Meta-Agent (CEO):
  "Our goal is to analyze competitor pricing."
        │
        ├── Sub-agent 1: "Research competitor A"
        │     └── search, scrape, summarize
        │
        ├── Sub-agent 2: "Research competitor B"
        │     └── search, scrape, summarize
        │
        └── Sub-agent 3: "Compare and produce report"
              └── read both summaries, write comparison
```
Best for: Large, decomposable tasks where sub-tasks can run in parallel.

### 6. Agent Observability

Production agents cannot be black boxes. You must be able to see what the agent is doing, why it made each decision, and where it failed.

**What to log at every step:**
```python
import time
import uuid

class AgentLogger:
    def __init__(self, session_id=None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.steps = []
        self.start_time = time.time()
    
    def log_step(self, step_num, thought, action, action_args, observation, latency_ms, tokens_used):
        entry = {
            "session_id": self.session_id,
            "step": step_num,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "thought": thought,
            "action": {"name": action, "args": action_args},
            "observation_preview": str(observation)[:200],
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
        }
        self.steps.append(entry)
        # In production: write to database, log aggregator, or trace export
    
    def summary(self):
        total_time = time.time() - self.start_time
        total_tokens = sum(s["tokens_used"] for s in self.steps)
        return {
            "session_id": self.session_id,
            "steps": len(self.steps),
            "total_time_s": round(total_time, 2),
            "total_tokens": total_tokens,
            "estimated_cost": (total_tokens / 1000) * 0.01,  # rough estimate
            "tools_used": list(set(s["action"]["name"] for s in self.steps)),
        }
```

**Observability dashboard metrics:**
```
Agent Performance Dashboard
┌─────────────────────────────────────────────────────────────┐
│ Session: sess_abc123  │  Steps: 7  │  Duration: 12.4s     │
├─────────────────────────────────────────────────────────────┤
│  Step │ Tool         │ Latency │ Tokens │ Status          │
│  1    │ search_web   │ 1.2s    │ 450    │ ✓ success       │
│  2    │ search_web   │ 0.9s    │ 420    │ ✓ success       │
│  3    │ calculator   │ 0.1s    │ 300    │ ✓ success       │
│  4    │ unknown_fn   │ 0.0s    │ 200    │ ✗ error         │
│  5    │ search_web   │ 1.1s    │ 480    │ ✓ success       │
│  6    │ gen_summary  │ 2.3s    │ 650    │ ✓ success       │
│  7    │ none (final) │ 3.1s    │ 800    │ ✓ final answer  │
├─────────────────────────────────────────────────────────────┤
│ Total tokens: 3,300  │  Estimated cost: $0.033             │
│ Success rate: 86%    │  Tools used: search_web, calculator │
└─────────────────────────────────────────────────────────────┘
```

---

## Lab 2.2: Multi-Step Agent with Plan → Act → Verify

### Goal
Build a ReAct agent that researches a topic, verifies facts, and produces a structured report.

### Steps
1. Implement the ReAct loop (thought → action → observation → repeat)
2. Add tools: `web_search`, `extract_facts`, `verify_claim`
3. Implement guardrails: step limits, token budgets, blocked tools
4. Add structured logging of every step
5. Build a trace viewer that prints the agent's step-by-step reasoning
6. Test with: "Research the impact of AI on healthcare in Africa and produce a 3-paragraph summary with at least 3 citations"

### Expected Observations
- The agent may take 3-8 steps depending on the complexity
- Without guardrails, the agent can loop or go off-topic
- Logging makes agent behaviour debuggable — you can see exactly where it went wrong
- The verify_claim tool catches hallucinated facts

### Deliverable
A Python script with a `run_agent()` function plus a sample trace log showing all steps.

---

## Exercises
1. **ReAct Trace Debugging**: Given this trace of an agent's thoughts and actions, identify where it went wrong and propose a fix:
   ```
   Thought: I need the user's email.
   Action: search_user(email="...")
   Observation: User not found.
   Thought: Let me try again.
   Action: search_user(email="...")
   (repeats 10 times)
   ```
2. **Guardrail Design**: Write guardrails for an agent that has access to: company database (read/write), email system (send/receive), payment API (process refunds, view transactions). What tools need human approval?
3. **Plan Comparison**: Implement "What are the top 3 programming languages in 2025 by job demand?" with both simple ReAct and plan-then-execute. Compare step counts, success rate, and total tokens used.

---

# Module 2.3: Retrieval-Augmented Generation

## Core Concepts

### 1. What is RAG? (The Full Picture)

Retrieval-Augmented Generation (RAG) is a pattern where the LLM retrieves relevant information from a knowledge base before generating a response. This grounds the model's output in actual data rather than relying solely on its training weights.

**The problem RAG solves:**
- LLMs have a knowledge cutoff — they don't know recent events
- LLMs hallucinate — they make up plausible-sounding facts
- Enterprise data is private — the model wasn't trained on it
- Users need citations — "how do you know this?"

**The RAG mantra:** *Retrieve first, generate second.*

**RAG in one diagram:**
```
                    User Question: "What was company revenue in 2024?"
                             │
                             ▼
                    ┌─────────────────────┐
                    │  1. EMBED QUERY     │
                    │  "company revenue   │
                    │   2024" → vector    │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  2. VECTOR SEARCH   │
                    │  Find nearest       │
                    │  document chunks    │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  Top-3 chunks:      │
                    │  [0] "revenue grew  │
                    │       to $2.4B..."  │
                    │  [1] "Q4 earnings   │
                    │       exceeded..."  │
                    │  [2] "annual report │
                    │       shows..."     │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  3. BUILD PROMPT    │
                    │  Context:           │
                    │  [1] revenue grew   │
                    │  [2] Q4 earnings    │
                    │  [3] annual report  │
                    │                     │
                    │  "Answer based only │
                    │   on the context..."│
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  4. LLM GENERATES   │
                    │  "Revenue grew to   │
                    │   $2.4B in 2024    │
                    │   [1][3]"           │
                    └─────────┬───────────┘
                              │
                              ▼
                    Final answer with citations
```

**Why RAG (not fine-tuning) for most use cases:**
| Concern | RAG | Fine-tuning |
|---------|-----|-------------|
| New information | ✓ Add to vector store | ✗ Need to retrain |
| Accuracy | ✓ Cites sources | ✗ May still hallucinate |
| Cost | Low (storage + API) | High (training compute) |
| Control | Full — you control the data | Partial — data is in weights |
| Updates | Instant — just re-index | Slow — days to retrain |

### 2. Embeddings for Retrieval (Deep Dive)

Embeddings convert text into dense vector representations — lists of floating-point numbers (typically 384 to 1536 dimensions). Documents with similar meaning have vectors close together in the embedding space.

**How embedding works at a high level:**
```
"Kenya's GDP grew by 5.6%"  →  [0.23, -0.45, 0.89, ..., 0.12]  (384 numbers)
"The Central Bank rate"     →  [0.21, -0.42, 0.91, ..., 0.15]  (close to above — similar topic)
"Nairobi is the capital"    →  [-0.67, 0.33, -0.12, ..., 0.54] (far from above — different topic)
```

**Computing similarity:**
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Kenya's GDP grew by 5.6% in 2024 driven by tech and agriculture",
    "The Central Bank of Kenya raised the benchmark rate to 12.5%",
    "Nairobi is the capital and largest city of Kenya",
    "Agriculture accounts for 33% of Kenya's GDP"
]

# Embed all documents at once
doc_embeddings = model.encode(documents)

# Embed the query
query = "What is the economic growth rate?"
query_embedding = model.encode([query])

# Compute similarity between query and each document
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

for doc, score in zip(documents, similarities):
    print(f"  {score:.3f}  {doc}")

# Output:
#   0.542  Kenya's GDP grew by 5.6% in 2024 driven by tech and agriculture
#   0.321  Agriculture accounts for 33% of Kenya's GDP
#   0.289  The Central Bank of Kenya raised the benchmark rate to 12.5%
#   0.187  Nairobi is the capital and largest city of Kenya
```

**Choosing an embedding model — the tradeoffs:**

| Model | Dims | Speed | Quality | Language | Size |
|-------|------|-------|---------|----------|------|
| `all-MiniLM-L6-v2` | 384 | 🔥 Fastest | Good | English | 80MB |
| `all-mpnet-base-v2` | 768 | ⚡ Fast | Better | English | 420MB |
| `text-embedding-3-small` | 1536 | API | Very Good | Multilingual | API |
| `text-embedding-3-large` | 3072 | API | Best | Multilingual | API |
| `BAAI/bge-large-en-v1.5` | 1024 | 🐢 Medium | Excellent | English | 1.3GB |
| `intfloat/multilingual-e5-large` | 1024 | 🐢 Medium | Excellent | 100+ langs | 2.2GB |

**Rule of thumb:** Start with `all-MiniLM-L6-v2` for prototyping. It's fast, small, and good enough. Upgrade to `bge-large-en-v1.5` or an API model when you need better retrieval quality.

### 3. Chunking Strategies (Deep Dive)

Documents are too long to embed as single vectors. A 100-page PDF cannot be reduced to one embedding — too much information is lost. So we split documents into **chunks** before embedding.

**Visualizing chunking:**
```
Document: 100-page PDF about Company Annual Report 2024
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
Chunk 1          Chunk 2          Chunk 3
(p.1-2:          (p.3-4:          (p.5-6:
 Executive       Revenue         Operating
 Summary)        Breakdown)      Expenses)
     │               │               │
     ▼               ▼               ▼
[vec_1]           [vec_2]           [vec_3]
```

**Chunking methods compared:**

**Method 1: Fixed-size (naive)**
```python
def fixed_chunks(text, chunk_size=512, overlap=50):
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append({
            "text": text[start:end],
            "start_char": start,
            "end_char": end
        })
        start += chunk_size - overlap
    return chunks

# Pros: Simple, predictable chunk count
# Cons: May split in the middle of a sentence or thought
```

**Method 2: Recursive character splitting**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # target chunk size in characters
    chunk_overlap=200,    # overlap between chunks
    separators=[          # priority order — tries first separator first
        "\n\n",           # paragraph break (best — preserves meaning)
        "\n",             # line break
        ". ",             # sentence boundary
        " ",              # word boundary
        ""                # character (last resort)
    ]
)

chunks = splitter.split_text(long_document)

# How it works:
# 1. Tries to split on "\n\n" first
# 2. If resulting chunks are > 1000 chars, splits further on "\n"
# 3. If still too large, splits on ". " (sentences)
# 4. Continues down the list until all chunks are under 1000 chars
```

**Method 3: Semantic chunking**
```python
def semantic_chunks(text):
    """Split on natural semantic boundaries."""
    import re
    
    # Split on section headers (Markdown or numbered)
    sections = re.split(r'\n#{1,3}\s+|\n\d+\.\s+', text)
    
    chunks = []
    for section in sections:
        if len(section.strip()) < 50:
            continue  # skip tiny fragments
        
        # Further split long sections by paragraphs
        paragraphs = section.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) < 1000:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return chunks

# Pros: Preserves meaning, sections stay intact
# Cons: Variable chunk sizes, more complex
```

**Chunking rules of thumb (extended):**
- **256-512 tokens**: Good for precise retrieval (Q&A, fact lookup)
- **512-1024 tokens**: Good for general purpose (summarization, analysis)
- **1024+ tokens**: Good for understanding long context (document comparison)
- **Overlap of 10-20%**: Essential. A sentence might start at end of chunk A and finish in chunk B. Overlap ensures both chunks contain the complete sentence.
- **Metadata preservation**: Every chunk must know which document, page, and section it came from.

### 4. Vector Stores

A vector store indexes embeddings and supports fast nearest-neighbour search. Think of it as a database optimized for "find me the most similar vectors" rather than "find me rows where id = X".

**How vector search works (simplified):**

```
Full vector space (384 dimensions — can't visualize)
     │
     ▼
We project to 2D for illustration:
                   
      doc_1 ●                    
                   ● doc_2        
          query ✦                 
                                  
doc_3 ●                          
                                  
              ● doc_4             
                                  
Nearest neighbours (Euclidean distance):
  1. doc_2 (closest to query)
  2. doc_1
  3. doc_4
  4. doc_3 (farthest)
```

**Vector store options compared:**

| Feature | FAISS | Chroma | Qdrant | Pinecone |
|---------|-------|--------|--------|----------|
| Setup | pip install | pip install | Docker | SaaS |
| Persistence | File | File | Service | Cloud |
| Speed | Fastest | Fast | Fast | Fast |
| Scalability | Millions | 100k's | Billions | Billions |
| Filters | Basic | Good | Excellent | Good |
| Self-hosted | ✓ | ✓ | ✓ | ✗ |
| Free tier | ✓ | ✓ | ✓ | ✓ (limited) |

**Full Chroma example:**
```python
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialize
client = chromadb.PersistentClient(path="./my_vectordb")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Create or get collection
collection = client.create_collection(
    name="company_docs",
    metadata={"description": "Company annual reports"}
)

# 3. Add documents with embeddings, metadata, and IDs
documents = [
    "Revenue grew 18% to $2.4 billion in 2024",
    "Operating expenses increased 12% to $1.8 billion",
    "Net income was $450 million, up 22% year over year",
    "The company has 12,000 employees across 15 countries"
]

collection.add(
    embeddings=embedder.encode(documents).tolist(),
    documents=documents,
    metadatas=[
        {"source": "annual_report_2024.pdf", "page": 3, "section": "Revenue"},
        {"source": "annual_report_2024.pdf", "page": 5, "section": "Expenses"},
        {"source": "annual_report_2024.pdf", "page": 7, "section": "Income"},
        {"source": "company_facts_2024.pdf", "page": 2, "section": "Overview"},
    ],
    ids=["doc_001", "doc_002", "doc_003", "doc_004"]
)

# 4. Query
results = collection.query(
    query_embeddings=embedder.encode(["What was the profit?"]).tolist(),
    n_results=3,
    # Optional: filter by metadata
    where={"section": {"$in": ["Revenue", "Income"]}}
)

for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
    print(f"{i+1}. [dist={dist:.3f}] {doc}")
```

### 5. Metadata Strategies (Extended)

Metadata is the unsung hero of RAG. Without metadata, retrieval is a black box — you get chunks but don't know where they came from or when they were written.

**What metadata to store (and why):**
```python
chunk_metadata = {
    # IDENTIFICATION
    "doc_id": "annual_report_2024.pdf",      # Which document
    "chunk_id": "chunk_047",                  # Which chunk within document
    "page": 12,                               # Page number
    "section": "Financial Highlights",        # Section heading
    
    # TEMPORAL
    "date": "2025-03-15",                     # When the document was published
    "ingested_at": "2025-04-01T10:30:00Z",    # When we added it to the vector store
    
    # SOURCE QUALITY
    "author": "Finance Department",           # Who wrote it
    "source_type": "internal_report",         # internal_report, news_article, academic_paper
    "confidence": "high",                      # high/medium/low — trustworthiness
    
    # CATEGORIZATION
    "category": "financial",                   # financial, technical, HR, legal
    "tags": ["revenue", "2024", "annual"],    # Free-form tags
    
    # RETRIEVAL
    "language": "en",                         # Language code
    "token_count": 412,                        # Length of chunk
}
```

**Using metadata for filtering:**
```python
# Only search recent financial documents
results = collection.query(
    query_texts=["revenue growth"],
    n_results=5,
    where={
        "$and": [
            {"category": {"$eq": "financial"}},
            {"date": {"$gte": "2024-01-01"}}
        ]
    }
)

# Only search high-confidence sources
results = collection.query(
    query_texts=["layoff rumors"],
    n_results=10,
    where={"confidence": {"$eq": "high"}}
)
```

**Metadata for citation:**
```python
def format_citation(chunk):
    """Turn a chunk into a citable source."""
    meta = chunk["metadata"]
    parts = []
    if meta.get("source"):
        parts.append(meta["source"])
    if meta.get("page"):
        parts.append(f"p.{meta['page']}")
    if meta.get("section"):
        parts.append(f"§{meta['section']}")
    if meta.get("date"):
        parts.append(f"({meta['date']})")
    
    source_str = ", ".join(parts)
    return f"[{chunk['rank']}] {source_str}"
```

### 6. The Complete Retrieval Pipeline

```
                        User Query: "How did our Q4 sales perform?"
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ QUERY TRANSFORMATION    │
                    │                         │
                    │ Options:                │
                    │ a) Direct use           │
                    │ b) Rewrite for retrieval│
                    │ c) HyDE (hypothetical)  │
                    │ d) Multi-query          │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │ EMBED QUERY             │
                    │ → 384-dim vector        │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │ VECTOR SEARCH (ANN)     │
                    │ Find 20 nearest chunks  │
                    │ Apply metadata filters  │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │ RERANK (Stage 2)        │
                    │ Score each of 20 with   │
                    │ cross-encoder           │
                    │ Keep top-3              │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │ BUILD PROMPT            │
                    │                         │
                    │ System: "Answer using   │
                    │  only the context..."   │
                    │                         │
                    │ Context:                │
                    │ [1] Q4 sales grew 15%   │
                    │ [2] E-commerce drove    │
                    │     22% increase        │
                    │                         │
                    │ Question: {query}       │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │ LLM GENERATE            │
                    │ "Q4 sales grew 15%,     │
                    │  driven by e-commerce   │
                    │  which saw 22% growth   │
                    │  [1][2]"                │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │ VERIFY CITATIONS        │
                    │ [1] → chunk_042 ✓       │
                    │ [2] → chunk_043 ✓       │
                    └──────────┬──────────────┘
                               │
                               ▼
                    Grounded answer with citations
```

**Query transformation techniques explained:**

```python
# Technique 1: Direct use
# Best when query is already well-formed for retrieval
query = "What was Q4 2024 revenue?"

# Technique 2: Rewrite for retrieval
# Best when query is conversational or vague
def rewrite_for_retrieval(conversational_query):
    prompt = f"""Rewrite this user question to be more effective 
    for searching a knowledge base. Extract key terms.
    
    User: {conversational_query}
    
    Search query:"""
    return call_llm(prompt)

# Example:
# User: "How did we do last quarter?"  
# Rewritten: "Q4 2024 financial performance results"

# Technique 3: HyDE (Hypothetical Document Embeddings)
# Generate a hypothetical answer, then embed that for search
def hyde_retrieval(query, num_docs=5):
    """Generate a hypothetical perfect answer, then use its embedding for search."""
    hyde_prompt = f"""Given the question, write a hypothetical paragraph 
    that would be the perfect answer. Be specific with facts and figures.
    
    Question: {query}
    Hypothetical answer:"""
    
    hypothetical_answer = call_llm(hyde_prompt)
    
    # Embed the hypothetical answer (not the query) and search
    hyde_embedding = embedder.encode([hypothetical_answer])
    results = vector_store.query(embeddings=hyde_embedding, n_results=num_docs)
    return results

# Technique 4: Multi-query
# Generate multiple search queries to cover different aspects
def multi_query(query):
    prompt = f"""Generate 3 different search queries to find 
    information about: {query}
    
    Each query should cover a different aspect or use different wording.
    Return as a numbered list."""
    
    queries = call_llm(prompt)
    # Parse into individual queries
    # Search all, merge results, deduplicate
```

---

## Lab 2.3: Basic RAG Pipeline Over a Document Set

### Goal
Build a RAG pipeline that ingests documents, chunks them, embeds them, and answers questions with citations.

### Steps
1. Collect 5-10 PDF or text documents on a topic (company reports, research papers, or documentation)
2. Implement recursive chunking with configurable size and overlap
3. Embed chunks and store in Chroma with rich metadata
4. Implement a retrieval function that finds the top-3 most relevant chunks
5. Build a prompt template that injects retrieved chunks with source labels
6. Implement citation formatting in the output
7. Test with 5 questions and manually verify citations are accurate

### Starter Code Skeleton
```python
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load documents
# 2. Chunk them
# 3. Create vector store
# 4. Implement retrieval
# 5. Build prompt with context
# 6. Call LLM
# 7. Return answer with citations
```

### Expected Observations
- Without retrieval, the LLM hallucinates or gives generic answers
- With retrieval, answers are grounded in the provided documents
- Chunk size affects quality — too large dilutes relevance, too small loses context
- Metadata enables source tracking for citations

### Deliverable
A Python script that ingests documents, builds a vector store, and answers questions with cited sources.

---

## Exercises
1. **Chunking Comparison**: Compare fixed-size (256, 512, 1024) and semantic chunking on the same document. Measure retrieval precision for 5 test questions using Hit Rate and MRR.
2. **Embedding Model Comparison**: Compare `all-MiniLM-L6-v2` vs `text-embedding-3-small` on retrieval recall for the same query set. Measure latency and quality tradeoffs.
3. **Metadata Filtering**: Implement date-range and category filters on your vector store. Compare filtered vs unfiltered results for queries that should only return data from 2024.

---

# Module 2.4: Reranking and Grounding

## Core Concepts

### 1. Why Reranking? (The Two-Stage Problem)

Vector search (ANN) finds approximate nearest neighbours fast. But "vector similar" ≠ "relevant." A document might be vector-close to the query but not actually answer the question.

**Example of the problem:**
```
Query: "What is the capital of Kenya?"

Vector search returns:
1. [dist 0.23] "Kenya's capital Nairobi is also its largest city"  ← Perfect!
2. [dist 0.31] "Nairobi has a population of 5 million"             ← Related but not answering
3. [dist 0.32] "Kenya is an East African country"                  ← Related but not answering
4. [dist 0.35] "The capital city serves as the economic hub"      ← Generic, mentions capital
5. [dist 0.37] "Tourism in Kenya contributes 10% to GDP"          ← Off-topic

Reranker scores:
1. [score 0.95] "Kenya's capital Nairobi is also its largest city"
2. [score 0.45] "Nairobi has a population of 5 million"
3. [score 0.40] "Kenya is an East African country"
```

The reranker dramatically separates the correct answer from merely related content.

**Why two stages? (The engineering tradeoff):**

| Stage | Model | Speed | Accuracy | Applied to |
|-------|-------|-------|----------|------------|
| 1. Vector search (ANN) | Bi-encoder (e.g., MiniLM) | 🔥 Fast: millions in ms | Good: captures semantic similarity | Entire corpus |
| 2. Reranker | Cross-encoder (e.g., MiniLM cross-encoder) | 🐢 Slow: hundreds per second | Excellent: understands query-doc relationship | Top-20 candidates only |

**The key insight:** The bi-encoder embeds query and document independently (fast but loses interaction information). The cross-encoder processes query + document together (slow but captures full interaction).

```
Bi-Encoder (Stage 1):
    Query → [encoder] → query_vec
    Doc   → [encoder] → doc_vec
    Similarity = cos(query_vec, doc_vec)
    ✓ Fast: pre-compute doc vectors
    ✗ No interaction between query and doc

Cross-Encoder (Stage 2):
    [CLS] Query [SEP] Doc [SEP] → [encoder] → relevance score
    ✗ Slow: must process every pair
    ✓ Rich query-doc interaction
```

### 2. Implementing Reranking

```python
from sentence_transformers import CrossEncoder
import numpy as np

# Load reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(query, vector_store, embedder, top_k=20, rerank_top_n=3):
    """Two-stage retrieval: vector search → rerank."""
    
    # Stage 1: Fast vector search
    results = vector_store.query(
        query_embeddings=embedder.encode([query]).tolist(),
        n_results=top_k
    )
    
    candidates = results["documents"][0]
    candidate_ids = results["ids"][0]
    candidate_metadatas = results["metadatas"][0]
    
    # Stage 2: Cross-encoder reranking
    pairs = [(query, doc) for doc in candidates]
    scores = reranker.predict(pairs)
    
    # Sort by reranker score (descending)
    ranked_indices = np.argsort(scores)[::-1]
    
    # Return top reranked results
    results = []
    for i in range(rerank_top_n):
        idx = ranked_indices[i]
        results.append({
            "rank": i + 1,
            "score": float(scores[idx]),
            "document": candidates[idx],
            "id": candidate_ids[idx],
            "metadata": candidate_metadatas[idx]
        })
    
    return results
```

### 3. Hybrid Search (Vector + Keyword)

Vector search is great for semantic similarity but can miss exact keyword matches. Hybrid search combines both:

```python
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, vector_store, bm25_index=None, alpha=0.5):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.alpha = alpha  # 0 = pure keyword, 1 = pure vector
    
    def search(self, query, k=10):
        # Vector search scores
        vec_results = self.vector_store.query(
            query_texts=[query],
            n_results=k
        )
        
        # Normalize vector scores to [0, 1]
        vec_scores = {}
        if vec_results["distances"]:
            max_dist = max(vec_results["distances"][0])
            min_dist = min(vec_results["distances"][0])
            dist_range = max_dist - min_dist if max_dist != min_dist else 1
            
            for doc_id, dist in zip(vec_results["ids"][0], vec_results["distances"][0]):
                # Convert distance to similarity (1 - normalized distance)
                vec_scores[doc_id] = 1 - ((dist - min_dist) / dist_range)
        
        # BM25 keyword scores
        bm25_scores = {}
        if self.bm25_index:
            tokenized_query = query.lower().split()
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Normalize BM25 scores to [0, 1]
            max_bm25 = max(scores) if max(scores) > 0 else 1
            for i, score in enumerate(scores):
                if score > 0:
                    bm25_scores[str(i)] = score / max_bm25
        
        # Combine scores
        all_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
        combined = []
        
        for doc_id in all_ids:
            v_score = vec_scores.get(doc_id, 0)
            b_score = bm25_scores.get(doc_id, 0)
            
            # Weighted harmonic mean (or weighted sum)
            hybrid_score = self.alpha * v_score + (1 - self.alpha) * b_score
            
            combined.append((doc_id, hybrid_score))
        
        # Sort by hybrid score
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:k]
```

**When to use hybrid search — concrete scenarios:**

| Scenario | Pure Vector | Pure Keyword | Hybrid |
|----------|-------------|--------------|--------|
| "revenue growth 2024" | ✓ Good | ✗ Misses synonyms | ✓ Best |
| "Bug #BUG-4421" | ✗ Misses exact code | ✓ Finds it | ✓ Best |
| "I need the form IRS-1040" | ✗ "tax document" | ✓ Exact form | ✓ Best |
| "Tell me about transformers" | ✓ Understands meaning | ✗ All mentions | ✓ Best |

### 4. Citation Grounding (Deep Dive)

Citation grounding means every factual claim in the LLM's answer can be traced back to a specific source chunk. Without citations, the user cannot verify the answer. With citations, the answer is transparent and trustworthy.

**The citation pipeline:**
```
1. We know which chunks were retrieved (chunk_042, chunk_043, chunk_051)
2. Prompt instructs LLM to cite [1], [2], [3] when making claims
3. LLM generates: "Revenue grew 15% in Q4 [1], driven by e-commerce [2]"
4. Verification step checks:
   - [1] maps to chunk_042 → does chunk_042 actually say "revenue grew 15%"?
   - [2] maps to chunk_043 → does chunk_043 actually mention "e-commerce"?
5. If a citation is unverifiable → reject and regenerate
```

**Citation prompt template:**
```python
CITATION_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

RULES:
1. Answer only using information from the context below
2. For every factual claim, add a citation number in brackets like [1], [2]
3. If multiple sources support a claim, cite all of them [1][2]
4. If the context does not contain the answer, say: "I cannot answer this based on the available documents."
5. Do NOT make up citations or use information outside the provided context

CONTEXT:
{context}

QUESTION: {question}

ANSWER (with citations):"""
```

**Citation verification:**
```python
import re

def verify_citations(answer: str, chunks: list[dict]) -> dict:
    """Verify every citation in the answer maps to real, relevant content.
    
    Returns:
        dict with: verified (bool), issues (list), answer (str)
    """
    # Extract all citation numbers from answer
    citation_nums = re.findall(r'\[(\d+)\]', answer)
    citation_nums = [int(n) for n in citation_nums]
    
    issues = []
    
    if not citation_nums:
        return {
            "verified": False,
            "issues": ["No citations found in answer"],
            "answer": answer
        }
    
    for num in citation_nums:
        idx = num - 1  # Convert to 0-indexed
        
        if idx >= len(chunks):
            issues.append(f"Citation [{num}] has no corresponding source chunk")
            continue
        
        chunk = chunks[idx]["document"]
        chunk_meta = chunks[idx].get("metadata", {})
        
        # Verify the claim actually appears in the chunk
        # (Simple check: extract claim sentences from answer and verify)
        # In production: use semantic similarity or NLI model
        
        source = f"{chunk_meta.get('source', 'unknown')}"
        if chunk_meta.get("page"):
            source += f", p.{chunk_meta['page']}"
        
        # Check that the chunk text is actually relevant to the claim
        # (simplified — real implementation uses NLI)
    
    return {
        "verified": len(issues) == 0,
        "issues": issues,
        "citation_count": len(citation_nums),
        "answer": answer
    }
```

**Advanced: Sentence-level citation verification:**
```python
def split_into_claims(answer: str) -> list[dict]:
    """Split answer into individual claims with their citations."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    claims = []
    
    for sentence in sentences:
        citations = re.findall(r'\[(\d+)\]', sentence)
        claims.append({
            "sentence": sentence,
            "citations": [int(c) for c in citations],
            "has_citations": len(citations) > 0
        })
    
    return claims

def verify_claims(claims: list[dict], chunks: list[dict]) -> list[dict]:
    """Verify each claim's citations against source chunks."""
    for claim in claims:
        claim["verified"] = True
        claim["issues"] = []
        
        if not claim["has_citations"]:
            # Claim without citations — might be hallucinated
            claim["verified"] = False
            claim["issues"].append("No citations provided for this claim")
            continue
        
        for c in claim["citations"]:
            idx = c - 1
            if idx >= len(chunks):
                claim["verified"] = False
                claim["issues"].append(f"Citation [{c}] references missing chunk")
    
    return claims
```

### 5. RAG Evaluation Framework (Extended)

You cannot improve what you cannot measure. A rigorous evaluation framework is what separates a demo from a production system.

**Test dataset design:**
```python
test_queries = [
    {
        "id": "q_001",
        "query": "What was the company's revenue in 2024?",
        "expected_answer": "Revenue grew 18% to $2.4 billion",
        "expected_doc_ids": ["annual_2024.pdf"],
        "category": "financial",
        "difficulty": "easy"
    },
    {
        "id": "q_002",
        "query": "Compare Q3 and Q4 performance across all segments",
        "expected_answer": "Q3: $580M, Q4: $620M across enterprise, SMB, and consumer",
        "expected_doc_ids": ["q3_report.pdf", "q4_report.pdf"],
        "category": "financial",
        "difficulty": "hard"
    },
    {
        "id": "q_003",
        "query": "This information is not in the documents",
        "expected_answer": None,  # Should return "cannot answer"
        "expected_doc_ids": [],
        "category": "edge_case",
        "difficulty": "medium"
    }
]
```

**Comprehensive evaluation script:**
```python
def evaluate_rag_system(rag_system, test_set):
    """Run full evaluation of a RAG system."""
    results = []
    
    for test in test_set:
        query = test["query"]
        
        # Time the response
        start = time.time()
        answer, citations, chunks = rag_system.answer(query)
        latency = time.time() - start
        
        # Metrics
        retrieved_ids = [c["doc_id"] for c in citations]
        expected_ids = test["expected_doc_ids"]
        
        # Hit Rate: Did the right document appear in retrieved?
        hit = any(eid in retrieved_ids for eid in expected_ids) if expected_ids else True
        
        # MRR: How early did the first relevant doc appear?
        if expected_ids:
            for rank, rid in enumerate(retrieved_ids, 1):
                if rid in expected_ids:
                    mrr = 1.0 / rank
                    break
            else:
                mrr = 0.0
        else:
            mrr = None  # No relevant docs expected
        
        # Answer correctness (if expected answer provided)
        if test["expected_answer"]:
            correct = test["expected_answer"].lower() in answer.lower()
        else:
            # For edge cases: should have said "cannot answer"
            correct = "cannot answer" in answer.lower()
        
        # Citation precision
        verified = verify_citations(answer, chunks)
        citation_count = len(re.findall(r'\[\d+\]', answer))
        citation_precision = (
            len([i for i in verified["issues"] if not i]) / citation_count
            if citation_count > 0 else 0
        )
        
        results.append({
            "id": test["id"],
            "query": query,
            "latency_ms": round(latency * 1000),
            "hit": hit,
            "mrr": mrr,
            "correct": correct,
            "citation_count": citation_count,
            "citation_precision": citation_precision,
            "answer_length": len(answer.split()),
        })
    
    return results

def print_eval_report(results):
    """Print a formatted evaluation report."""
    print("=" * 60)
    print("RAG EVALUATION REPORT")
    print("=" * 60)
    
    avg_latency = np.mean([r["latency_ms"] for r in results])
    hit_rate = np.mean([r["hit"] for r in results])
    mrr_vals = [r["mrr"] for r in results if r["mrr"] is not None]
    avg_mrr = np.mean(mrr_vals) if mrr_vals else 0
    accuracy = np.mean([r["correct"] for r in results])
    avg_citations = np.mean([r["citation_count"] for r in results])
    avg_cite_precision = np.mean([r["citation_precision"] for r in results])
    
    print(f"  Hit Rate:              {hit_rate:.1%}")
    print(f"  Mean Reciprocal Rank:  {avg_mrr:.3f}")
    print(f"  Answer Accuracy:       {accuracy:.1%}")
    print(f"  Avg Citations:         {avg_citations:.1f}")
    print(f"  Citation Precision:    {avg_cite_precision:.1%}")
    print(f"  Avg Latency:           {avg_latency:.0f}ms")
    print(f"  Total Queries:         {len(results)}")
    print("=" * 60)
    
    # Per-query breakdown
    print("\nPER-QUERY BREAKDOWN:")
    print(f"{'ID':<10} {'Hit':<6} {'MRR':<6} {'Correct':<9} {'Lat(ms)':<8}")
    print("-" * 40)
    for r in results:
        mrr_str = f"{r['mrr']:.3f}" if r['mrr'] else "N/A"
        print(f"{r['id']:<10} {r['hit']!s:<6} {mrr_str:<6} {r['correct']!s:<9} {r['latency_ms']:<8}")
```

**Comparison report format (baseline vs enhanced):**
```markdown
# RAG Comparison Report

## Configuration
| Setting | Baseline | Enhanced |
|---------|----------|----------|
| Retrieval | Vector search only | Hybrid search |
| Reranker | None | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Citation | None | Inline citations + verification |
| Chunk size | 1000 chars | 1000 chars with 200 overlap |

## Results Summary
| Metric | Baseline | Enhanced | Improvement |
|--------|----------|----------|-------------|
| Hit Rate@3 | 72% | 84% | +12% |
| MRR | 0.421 | 0.613 | +46% |
| Answer Accuracy | 65% | 81% | +16% |
| Citation Precision | N/A | 94% | New capability |
| Avg Latency | 1.2s | 1.8s | +0.6s (acceptable tradeoff) |

## Analysis
The enhanced RAG system shows significant improvements across all quality metrics
at the cost of moderate additional latency. The hybrid search component was
particularly effective for queries containing proper nouns and specific terms.
```

---

## Lab 2.4: Add Reranking and Sources to RAG

### Goal
Enhance the basic RAG pipeline from Module 2.3 with reranking, hybrid search, and citation grounding. Measure the improvement.

### Steps
1. Add a cross-encoder reranker to the existing RAG pipeline
2. Implement hybrid search (vector + BM25 keyword)
3. Add citation markers to the prompt template with source metadata
4. Implement a citation verification function
5. Build an evaluation harness that compares baseline vs enhanced RAG
6. Run 10 test queries and produce a comparison report

### Expected Observations
- Reranking improves the relevance of top-3 results (visible in MRR)
- Hybrid search catches exact matches that pure vector search misses
- Citation grounding makes answers verifiable and increases trust
- The quality improvement comes at a latency cost (0.3-0.8s for reranking)

### Deliverable
An enhanced RAG pipeline with reranker, hybrid search, citation output, and an evaluation script that produces a comparison report.

---

## Exercises
1. **Reranker Comparison**: Compare 3 reranker models (`cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-v2-m3`, Cohere Rerank) on MRR and latency for the same query set.
2. **Hybrid Search Tuning**: Vary the alpha parameter from 0.0 to 1.0 in 0.1 increments. Find the optimal blend for your document set and explain why that value works best.
3. **Citation Audit**: Manually audit 20 RAG answers. For each citation, classify as: (a) accurate and relevant, (b) accurate but irrelevant, (c) partially accurate, (d) completely hallucinated. Report percentages.

---

# Month 2 Mini-Project: Production-Grade RAG System with Evaluation Harness

## Goal
Build an end-to-end RAG system over a document collection of your choice that includes hybrid search, reranking, citation grounding, and a rigorous evaluation harness. The system must be demonstrably better than a naive baseline.

## Requirements

### Document Ingestion
- Ingest at least 20 documents (PDF, text, or HTML) on a coherent topic
- Implement recursive chunking with configurable chunk size and overlap
- Store chunks with rich metadata (source, page, section, date, category)
- Use a vector store (Chroma recommended for simplicity)
- Implement a re-indexing function that can add new documents

### Retrieval
- Implement hybrid search (vector embedding + BM25 keyword)
- Alpha parameter must be configurable
- Add a reranker stage using a cross-encoder model
- Support metadata filtering (by date range, category, source type)
- Return the top-3 chunks with their metadata and relevance scores

### Generation
- Use a prompt template that injects retrieved chunks with source labels and metadata
- The LLM must produce inline citations in the format [1], [2], etc.
- Implement a citation verification step that:
  - Checks every citation references a real chunk
  - Warns if a claim lacks a citation
  - Rejects and regenerates if citations cannot be verified
- If the context doesn't contain the answer, the system must say so (not hallucinate)

### Evaluation Harness
- Create a test set of at least 20 queries covering:
  - 10 easy/factual queries (can be answered from a single chunk)
  - 5 hard queries (require synthesis across multiple chunks)
  - 3 edge cases (query has no answer in documents)
  - 2 queries with specific proper nouns or codes
- Implement these metrics: Hit Rate@3, MRR, Answer Correctness, Citation Precision, Average Latency
- Generate a side-by-side comparison report:
  - **Baseline RAG**: vector search only, no reranker, no citations
  - **Enhanced RAG**: hybrid search + reranker + citations + verification
- Report must include a table of metrics and per-query breakdown

### Deliverable Structure
```
month_2/mini_project/
├── rag_system.py          # Full RAG pipeline with all features
├── evaluate.py            # Evaluation harness
├── test_queries.json      # 20+ test queries
├── baseline_report.json   # Baseline results
├── enhanced_report.json   # Enhanced results
├── comparison_report.md   # Final report with analysis
└── README.md              # How to run
```

### Example Test Set Entry
```python
{
    "id": "q_001",
    "query": "What was the company's revenue in 2024?",
    "expected_answer": "The company grew revenue by 18% to $2.4 billion",
    "expected_doc_ids": ["annual_report_2024.pdf"],
    "category": "financial",
    "difficulty": "easy"
}
```

## Rubric (100 points)

| Criterion | Points | Description |
|-----------|--------|-------------|
| Document ingestion and chunking | 15 | Proper chunking with overlap, metadata, vector store setup |
| Hybrid search | 10 | Vector + BM25 combined with configurable alpha |
| Reranker integration | 10 | Cross-encoder reranking correctly applied |
| Citation grounding | 15 | Inline citations accurate, verification implemented, "can't answer" handled |
| Evaluation harness | 20 | Complete test set, all 5 metrics implemented, automated comparison |
| Quality improvement | 15 | Enhanced RAG beats baseline on at least 3 of 5 metrics by measurable margin |
| Code quality | 10 | Clean code, docstrings, type hints, error handling |
| Documentation | 5 | README with setup and usage instructions |

---

## Common Pitfalls and How to Address Them

### Retrieval Returns Irrelevant Chunks
- **Increase top-k**: The reranker needs enough candidates to find good ones
- **Fix chunk boundaries**: A sentence split across two chunks loses meaning
- **Better embedding model**: BGE or OpenAI embeddings outperform MiniLM for domain-specific content
- **Add reranker**: Without it, irrelevant chunks slip through

### LLM Ignores Retrieved Context
- **Strengthen the system prompt**: "ONLY use the provided context. Never use your own knowledge."
- **Put context AFTER instructions**: Some models pay more attention to later content
- **Add a verification step**: Reject outputs that don't cite context
- **Try few-shot examples**: Show examples of good citation usage

### Citations Are Hallucinated
- **Post-process**: Strip any citation that doesn't reference a retrieved chunk ID
- **Add explicit source markers**: `[Source: doc_042, p.12]` instead of just `[1]`
- **Implement claim verification**: Check each claim against its cited chunk using semantic similarity
- **Regenerate on failure**: If verification fails, retry with a stronger "you must cite accurately" prompt

### Latency Is Too High
- **Reduce top-k for reranker**: 10 instead of 20
- **Use a smaller reranker**: MiniLM cross-encoder is 10x faster than BGE reranker
- **Cache embeddings**: Reuse document embeddings instead of recomputing
- **Parallel retrieval**: Search vector and keyword indices simultaneously
- **Async LLM calls**: Stream tokens instead of waiting for full response

### Hybrid Search Alpha Is Wrong
- **Run a sweep**: Test alpha = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] on a validation set
- **Measure MRR for each**: The best alpha depends on your data
- **Rule of thumb**: Start with 0.5, then adjust. More code/product names → lower alpha. More conceptual queries → higher alpha.

### Metadata Filtering Is Slow
- **Use pre-filtering**: Filter metadata BEFORE vector search (not after)
- **Index metadata fields**: Chroma and Qdrant support indexable metadata
- **Limit filter complexity**: Simple equality filters are faster than regex or range queries

---

## Resources

### Libraries
- `sentence-transformers` — embedding models and cross-encoders
- `chromadb` — lightweight vector store (persistent or in-memory)
- `faiss-cpu` — faster vector search for larger collections
- `rank-bm25` — BM25 keyword search for hybrid retrieval
- `pytrec_eval` — standard IR metrics (MRR, NDCG, MAP)
- `langchain` — higher-level RAG abstractions (optional, can obscure learning)

### Papers
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020) — The original RAG paper
- "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022) — The ReAct agent pattern
- "In-Context Retrieval-Augmented Language Models" (Ram et al., 2023) — Advanced RAG techniques
- "Lost in the Middle: How Language Models Use Long Contexts" (Liu et al., 2023) — Why retrieval ordering matters

### Articles
- OpenAI Function Calling documentation — Tool schema design patterns
- Anthropic Tool Use documentation — Alternative function calling approach
- Chroma documentation — Vector store setup and querying
- LangSmith tracing guide — Agent observability in production

### Tools
- Chroma / FAISS / Qdrant — Vector storage options
- Ollama — Run local LLMs for testing without API costs
- LangSmith / Weights & Biases — Tracing and evaluation
- MLflow — Experiment tracking for RAG configurations

---

## Summary

### Key Takeaways

**Tool Use and Function Calling**
- Tools are described to the LLM via JSON schemas with typed parameters, descriptions, and enums
- The LLM requests function calls; your code executes them safely
- Error handling, retries, and fallbacks are essential in production
- State management (the messages array) tracks context across turns

**Agent Patterns**
- ReAct interleaves reasoning traces (Thought) with actions (Action) and results (Observation)
- Guardrails must be layered: input → process → action → output
- Step limits, token budgets, tool allowlists, and human-in-the-loop protect against failures
- Observability (logging every thought, action, and observation) is critical for debugging and cost tracking

**RAG Fundamentals**
- Embeddings convert text to vectors; similarity search finds relevant documents
- Chunking strategy directly affects retrieval quality — recursive chunking with overlap is the standard
- Vector stores (Chroma, FAISS, Qdrant) enable fast approximate nearest-neighbour search
- Metadata enables filtering, ranking, and citation tracking

**Reranking and Grounding**
- Two-stage retrieval: fast vector search (bi-encoder) → precise reranking (cross-encoder)
- Hybrid search (vector + BM25 keyword) catches exact matches that semantic search misses
- Citation grounding makes answers verifiable — every claim traceable to a source chunk
- Evaluation metrics (Hit Rate, MRR, NDCG, Citation Precision) are essential for measuring and improving quality

### Architecture Overview

```
                    ┌─────────────────────────────┐
                    │         User Query          │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │    Query Transformation     │
                    │    (rewrite / HyDE / multi) │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                    │
              ▼                   ▼                    ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  Vector Search  │  │  BM25 Keyword   │  │  Metadata       │
    │  (semantic)     │  │  (lexical)      │  │  Filtering      │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
              │                   │                    │
              └───────────────────┼────────────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │      Hybrid Fusion          │
                    │  (weighted combination)     │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │      Reranker (top-20)      │
                    │   cross-encoder scores      │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │      Prompt Builder         │
                    │  Context chunks +           │
                    │  source labels + rules      │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │      LLM Generate           │
                    │  Answer with inline         │
                    │  citations [1], [2], [3]    │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │      Citation Verification  │
                    │  Every [N] → real chunk     │
                    │  Claim → chunk content ✓    │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │    Grounded Answer          │
                    │  + Sources + Citations      │
                    └─────────────────────────────┘
```

---

## Extension (Optional)

- **Multi-modal RAG**: Add image search using CLIP embeddings. Retrieve both text chunks and images, then use a vision-language model to generate answers.
- **Agentic RAG**: Combine agents with RAG. The agent decides when to retrieve (not every query needs RAG), what to retrieve (chooses which document collection), and what to do with results (summarize, compare, extract).
- **Streaming RAG**: Stream tokens from the LLM while progressively showing which chunks were retrieved. Users see the evidence building as the answer forms.
- **Feedback loop**: Add a thumbs up/down button. Log feedback with the query, chunks, and answer. Periodically review low-rated answers to improve chunking, retrieval, or prompts.
- **A/B testing framework**: Deploy two RAG configurations (different chunk sizes, different embedding models) to 50% of traffic each. Compare online metrics: user satisfaction, engagement, retention.
- **Self-reflective RAG**: After generating an answer, have the LLM rate its own confidence. If confidence is low, trigger re-retrieval with a refined query.
