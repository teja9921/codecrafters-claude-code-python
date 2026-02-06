import argparse
import os
import sys
import json
import subprocess
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
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "writes content to the file specified.",
            "parameters":{
                "type": "object",
                "properties":{
                    "file_path":{
                        "type": "string",
                        "description": "the path of the file to write to"
                    },
                    "content":{
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function":{
            "name": "run_bash_command",
            "description":"Execute a shell command",
            "parameters":{
                "type": "object",
                "properties":{
                    "command":{
                        "type": "string",
                        "description": "The command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    }
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [{"role": "user", "content": args.p}]
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
            messages.append(message.model_dump(exclude_none=True))
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                if function_name == "read_file":
                    file_path= arguments["file_path"]
                    with open(file_path, "r") as f:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(f.read())
                        })

                if function_name == "write_file":
                    file_path = arguments["file_path"]
                    content = arguments["content"]
                    with open(file_path, "w") as f:
                        f.write(str(content))
                    messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(content)
                        })
                    
                if function_name == "run_bash_command":
                    command = arguments["command"]
                    try: 
                        # 'check=True' raises an error if the command fails
                        # 'capture_output=True' grabs stdout and stderr
                        # 'text=True' returns the output as a string instead of bytes

                        result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True)
                        messages.append({
                            "role": "tool",
                            "tool_id": tool_call.id,
                            "content": result.stdout
                        })
                    except subprocess.CalledProcessError as e:
                        messages.append({
                            "role": "tool",
                            "tool_id": tool_call.id,
                            "content": result.stderr
                        })

        else:
            print(message.content)
            break

if __name__ == "__main__":
    main()
