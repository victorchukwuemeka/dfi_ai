import json
from openai import OpenAI


def _get_client():
    return OpenAI()


def extract_between(text: str, start: str, end: str) -> str:
    start_idx = text.find(start)
    if start_idx == -1:
        return ""
    start_idx += len(start)
    end_idx = text.find(end, start_idx)
    if end_idx == -1:
        return text[start_idx:].strip()
    return text[start_idx:end_idx].strip()


def call_llm(messages, tools=None):
    return _get_client().chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto" if tools else None,
    )


def re_act(
    prompt: str,
    tools: list[dict],
    tool_handler: dict[str, callable],
    max_step: int = 10
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that can use tools to answer questions. "
                "You should think step by step about what information you need, "
                "use the appropriate tools to gather it, and then provide a final answer. "
                "Format your responses as:\n"
                "Thought: <your reasoning>\n"
                "Action: <tool_name>\n"
                "Action Input: <JSON args>\n"
                "Wait for the observation before continuing."
            )
        },
        {"role": "user", "content": prompt}
    ]

    for step in range(max_step):
        print(f"\n--- Step {step + 1} ---")

        response = call_llm(messages, tools=tools)
        content = response.choices[0].message.content

        print(f"LLM: {content}")

        if "Final:" in content or response.choices[0].finish_reason == "stop":
            return content.split("Final:")[-1].strip()

        if "Action:" in content:
            action_name = extract_between(content, "Action:", "\n")
            action_input_str = extract_between(content, "Action Input:", "\n")

            try:
                action_args = json.loads(action_input_str)
            except json.JSONDecodeError:
                action_args = {"input": action_input_str.strip()}

            handler = tool_handler.get(action_name)
            if handler:
                try:
                    result = handler(**action_args)
                except Exception as e:
                    result = f"Error executing tool: {str(e)}"
            else:
                result = f"Error: Unknown tool '{action_name}'"

            print(f"  -> Tool result: {result[:200]}...")

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Observation: {result}"})

        else:
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Please continue. What is your next step?"})

    return "I was unable to complete the task within the step limit."
