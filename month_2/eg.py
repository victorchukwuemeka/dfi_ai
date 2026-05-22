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
