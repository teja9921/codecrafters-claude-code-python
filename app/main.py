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
            messages=messages,
            tools=tools
        )

        if not chat.choices:
            raise RuntimeError("no choices in response")

        message = chat.choices[0].message
        
        # FIX 1: OpenRouter/Bedrock often requires 'content' to not be empty
        msg_dict = message.model_dump(exclude_none=True)
        if not msg_dict.get("content"):
            msg_dict["content"] = None 
        messages.append(msg_dict)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                # FIX 2: Added basic error handling to file operations so the script doesn't crash
                try:
                    if function_name == "read_file":
                        with open(arguments["file_path"], "r") as f:
                            result_content = f.read()

                    elif function_name == "write_file":
                        with open(arguments["file_path"], "w") as f:
                            f.write(str(arguments["content"]))
                        result_content = "File written successfully."

                    elif function_name == "run_bash_command":
                        try:
                            res = subprocess.run(arguments["command"], check=True, capture_output=True, text=True, shell=True)
                            result_content = res.stdout if res.stdout else "Command executed (no output)."
                        except subprocess.CalledProcessError as e:
                            # FIX 3: Use 'e.stderr' because 'result' isn't defined on failure
                            result_content = e.stderr if e.stderr else f"Error code: {e.returncode}"

                    # FIX 4: Critical key fix: 'tool_call_id', NOT 'tool_id'
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result_content)
                    })
                except Exception as e:
                    # Provide the error back to the AI so it can try to fix its mistake
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error executing tool: {str(e)}"
                    })
        else:
            # Final output
            if message.content:
                print(message.content)
            break

if __name__ == "__main__":
    main()