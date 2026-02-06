import argparse
import os
import sys
import json

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

tools =[
    {
        "type": "function",
        "function":{
            "name": "read_file",
            "description": "Read and return the contents of a file",
            "parameters":{
                "type": "object",
                "properties":{
                    "file_path":{
                        "type": "string",
                        "description": "The path to file to read",
                    }
                },
            "required": ["file_path"]
            }
        }
    },
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [{"role": "user", "content": args.p}],
    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages= messages,
            tools= tools
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        # You can use print statements as follows for debugging, they'll be visible when running tests.
        print("Logs from your program will appear here!", file=sys.stderr)
        message = chat.choices[0].message
        if message.tool_calls:
            for tool_call in messages.tool_calls
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                if function_name == "read_file":
                    file_path= arguments["file_path"]
                    with open(file_path, "r") as f:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f.read()
                        })
                
        else:
            print(message.content)
            break

if __name__ == "__main__":
    main()
