import _thread as thread
import base64, requests
from typing import Any
import datetime, os
import hashlib
import hmac
import json
import re
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import websocket  # 使用websocket_client

from LLM_cache import DiskCache
from utils.utils import get_clock_time


def extract_content(text):
    # pattern = r"```(?!.*\n```$)\s(.*?)\s```"  # 这段regex好像不太好
    pattern = r"```.*\n([\s\S]*?)```"  # 捕获python代码块中的内容
    pattern = r"```python\s*\n([\s\S]*?)\s*```"  # 捕获python代码块中的内容
    # result = re.sub(r"[\u4e00-\u9fa5]+|[，。；；【】、！]+", "", text)    # 清除字符串中的中文
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 0:
        return [text]
    return matches


def chinese_filter(text):
    return re.sub(
        r"[\u4e00-\u9fa5]+|[“”‘’？》《￥，。；：【】、！]+", "", text
    )  # 清除字符串中的中文


class TGI:
    """
    class that warp the interface of TGI
    """

    def __init__(self, **kwargs) -> None:
        self._full_name = kwargs["type"] + kwargs["version"]
        self._url = kwargs["url"] + ":" + str(kwargs["port"])
        self._model_instruction = kwargs["model_instruction"]
        # self._load_cache = False
        self._temperature = 0.1
        if "system_instruction" not in kwargs:
            self._system_instruction = ""
        else:
            self._system_instruction = kwargs["system_instruction"]
        if "load_cache" in kwargs:
            self._load_cache = kwargs["load_cache"]
        self._cache_root_dir = kwargs["cache_root_dir"]
        self._cache_dir_base = os.path.join(
            self._cache_root_dir, self._full_name, get_clock_time()
        )
        self._cache_dir_base = os.path.join(
            self._cache_root_dir, self._full_name, "persistent_arm"
        )
        if self._load_cache and not os.path.exists(self._cache_dir_base):
            os.makedirs(self._cache_dir_base)
        self._cache = DiskCache(
            load_cache=self._load_cache, cache_dir=self._cache_dir_base
        )

    def __call__(self, **kwds: Any) -> Any:
        assert "prompt" in kwds.keys(), "engine call kwargs not contain messages"
        prompt, splited_prompt = kwds["prompt"]
        assert (
            len(splited_prompt) % 2 == 1
        ), f"len(splited_prompt)={len(splited_prompt)}, please ask assistant"
        use_cache = False  # whether or not checking cache before calling API online
        if "use_cache" in kwds and self._load_cache:
            use_cache = kwds["use_cache"]
        temperature = self._temperature  # get default temperature
        stop_tokens = []
        model_instruction = self._model_instruction  # get default model_instruction
        if "model_instruction" in kwds.keys():
            # override the model instruction but not override the system instruction
            # so we can adjust the model instruction for any call
            # TODO
            model_instruction = kwds["model_instruction"]
        if "stop" in kwds.keys():
            stop_tokens = kwds["stop"]
        if "temperature" in kwds.keys():
            # override the default temperature if it in kwds
            temperature = kwds["temperature"]
        messages = []
        messages.append(
            {
                "role": "system",
                "content": self._system_instruction,
            }
        )
        for idx, content in enumerate(splited_prompt):
            messages.append(
                {
                    "role": ["user", "assistant"][idx % 2],
                    "content": [model_instruction + "\n\n" + content + "\n\n", content][
                        idx % 2
                    ],
                }
            )
        inputs = f"{self._system_instruction} \n\n {prompt}"
        parameters = {
            "max_new_tokens": 512,
            "temperature": temperature,
            "stop": stop_tokens,
            "do_sample": True,
        }
        payload = json.dumps(
            {
                "model": "tgi",
                "messages": messages,
                "temperature": temperature,
                "stop": stop_tokens,
            }
        )
        payload = json.dumps({"inputs": inputs, "parameters": parameters})
        headers = {"Content-Type": "application/json"}
        # cache_key = {f"{self._full_name}": splited_prompt[-1]}
        cache_key = splited_prompt[-1]
        print(f"cache_key: {cache_key}")
        if use_cache:
            if cache_key in self._cache:
                print("(using cache)", end=" ")
                return self._cache[cache_key]
        try:
            response = requests.request(
                "POST",
                f"{self._url}/generate",
                headers=headers,
                data=payload,
            )
            code_str = response.json()["generated_text"]
        except Exception as e:
            print("Error:", e)
            print(f"{self._url}/generate")
            print(response.content)
            print(payload)
            exit(1)
            # todo: if reach the max length of API limit, need to switch to a shorter version

        code_segments = extract_content(code_str)
        if len(code_segments) > 0:
            ret = code_segments[0].strip()
        else:
            ret = chinese_filter(code_str).strip()

        # whatever caching the result
        if self._load_cache:
            self._cache[cache_key] = ret
            print(f"cached {cache_key}")
        return ret


class Ollama:
    """
    class that warp the interface of Ollama
    """

    def __init__(self, **kwargs) -> None:
        self._model_name = kwargs["model"]
        self._full_name = kwargs["type"] + kwargs["model"]
        self._url = kwargs["url"] + ":" + str(kwargs["port"])
        self._model_instruction = kwargs["model_instruction"]
        self._temperature = (
            0.1
            if "default_temperature" not in kwargs
            else kwargs["default_temperature"]
        )
        self._system_instruction = (
            "" if "system_instruction" not in kwargs else kwargs["system_instruction"]
        )
        self._load_cache = kwargs["load_cache"] if "load_cache" in kwargs else False
        self._cache_root_dir = kwargs["cache_root_dir"]
        self._cache_dir_base = os.path.join(
            self._cache_root_dir, self._full_name, get_clock_time()
        )
        if self._load_cache and not os.path.exists(self._cache_dir_base):
            os.makedirs(self._cache_dir_base)
        self._cache = DiskCache(
            load_cache=self._load_cache, cache_dir=self._cache_dir_base
        )

    def __call__(self, **kwds: Any) -> Any:
        assert "prompt" in kwds.keys(), "engine call kwargs not contain messages"
        prompt, splited_prompt = kwds["prompt"]
        assert (
            len(splited_prompt) % 2 == 1
        ), f"len(splited_prompt)={len(splited_prompt)}, please ask assistant"

        use_cache = (
            False
            if ("use_cache" not in kwds or not self._load_cache)
            else kwds["use_cache"]
        )
        temperature = (
            self._temperature if "temperature" not in kwds else kwds["temperature"]
        )
        model_instruction = (
            self._model_instruction
            if "model_instruction" not in kwds
            else kwds["model_instruction"]
        )
        stop_tokens = [] if "stop" not in kwds else kwds["stop"]

        # Treat example codes as conversation history
        messages = []
        messages.append(
            {
                "role": "system",
                "content": self._system_instruction,
            }
        )
        for idx, content in enumerate(splited_prompt):
            messages.append(
                {
                    "role": ["user", "assistant"][idx % 2],
                    "content": [model_instruction + "\n\n" + content + "\n\n", content][
                        idx % 2
                    ],
                }
            )

        payload = json.dumps(
            {
                "model": self._model_name,
                "messages": messages,
                "stream": False,
                "keep_alive": "5m",
                "options": {
                    "num_predict": 512,
                    "num_ctx": 2564,
                    "temperature": temperature,
                    "stop": stop_tokens,
                    "top_k": 40,
                    "top_p": 0.9,
                },
            }
        )
        headers = {"Content-Type": "application/json"}
        cache_key = {f"{self._full_name}": payload}
        if use_cache:
            if cache_key in self._cache:
                print("(using cache)", end=" ")
                return self._cache[cache_key]
        try:
            response = requests.request(
                "POST",
                f"{self._url}/api/chat",
                headers=headers,
                data=payload,
            )
            response = response.json()
            code_str = response["message"]["content"]

            # To calculate how fast the response is generated in tokens per second (token/s), divide eval_count / eval_duration * 10^9.
            generated_speed = response["eval_count"] / response["eval_duration"] * 10**9
            print(
                f"[engine_interface.py|Ollama] generated_speed: {generated_speed:.2f} token/s"
            )
        except KeyError as e:
            print("KeyError:", e)
            print(response.content)
            print(payload)
            exit(1)
            # TODO: if reach the max length of API limit, need to switch to a shorter version
        except Exception as e:
            print("KeyError:", e)
            print(response.content)
            print(payload)
            exit(1)

        code_segments = extract_content(code_str)
        if len(code_segments) > 0:
            ret = code_segments[0].strip()
        else:
            ret = chinese_filter(code_str).strip()

        # whatever caching the result
        if self._load_cache:
            self._cache[cache_key] = ret
        return ret


class Dummy:
    """Dunmmy engine for test, it will return the test code directly"""

    def __init__(self, **kwargs) -> None:
        self.answers = {
            "# Query: go to the table.": """
composer('move to 10cm above the table')
""",
            "# Query: move to 10cm above the table.": """
movable = parse_query_obj('quadricopter')
affordance_map = get_affordance_map('a point 10cm above the table')
execute(movable, affordance_map=affordance_map)
""",
            "# Query: a point 10cm above the table.": """
affordance_map = get_empty_affordance_map()
table_obj = parse_query_obj('table')
(min_x, min_y, min_z), (max_x, max_y, max_z) = table_obj.aabb
center_x, center_y, center_z = table_obj.position
x = center_x
y = center_y
z = max_z + cm2index(10, 'z')
affordance_map[x, y, z] = 1
ret_val = affordance_map
""",
            "# Query: table.": """
table = detect('table')
ret_val = table[0]
""",
            "# Query: quadricopter.": """
quadricopter = detect('quadricopter')
ret_val = quadricopter[0]
""",
        }

    def __call__(self, **kwargs):
        assert "prompt" in kwargs.keys(), "engine call kwargs not contain messages"
        prompt, splited_prompt = kwargs["prompt"]  # add other params
        last_prompt = splited_prompt[-1]
        # get the last line of last_prompt except the blank line
        for line in last_prompt.split("\n")[::-1]:
            if line != "\n" and line != "":
                last_prompt = line
                break
        assert (
            last_prompt in self.answers.keys()
        ), f"prompt [{last_prompt}] not in answers {self.answers.keys()}"
        return self.answers[last_prompt]


class GPT4:
    """
    def _cached_api_call(self, **kwargs):
    # check whether completion endpoint or chat endpoint is used
    if any([chat_model in kwargs['model'] for chat_model in ['gpt-3.5', 'gpt-4', 'SparkV3']]):
        # add special prompt for chat endpoint
        # user1 = kwargs.pop('prompt')
        # new_query = '# Query:' + user1.split('# Query:')[-1]
        # user1 = ''.join(user1.split('# Query:')[:-1]).strip()
        instruction = '续写代码，不要出现任何不是代码的语言，把续写的代码放在markdown格式中发给我，不要解释代码'
        instruction = '你现在是一个写代码专家，续写下列这段代码（尤其需要根据最后一行的注释完成接下去的代码），不要出现其他解释性语句，以最后一行注释开头'
        # user1 = f"I would like you to help me write Python code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."
        # assistant1 = f'Got it. I will complete what you give me next.'
        # user2 = new_query
        # handle given context (this was written originally for completion endpoint)
        # if user1.split('\n')[-4].startswith('objects = ['):
        #     obj_context = user1.split('\n')[-4]
        #     # remove obj_context from user1
        #     user1 = '\n'.join(user1.split('\n')[:-4]) + '\n' + '\n'.join(user1.split('\n')[-3:])
        #     # add obj_context to user2
        #     user2 = obj_context.strip() + '\n' + user2
        # messages=[
        #     {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."},
        #     {"role": "user", "content": user1},
        #     {"role": "assistant", "content": assistant1},
        #     {"role": "user", "content": user2},
        # ]
        messagesv2=[
            {"role": "user", "content": instruction+'\n'+kwargs.pop('prompt')}
        ]
        kwargs['messages'] = messagesv2
        if kwargs in self._cache:
            print('(using cache)', end=' ')
            return self._cache[kwargs]
        else:
            if self._engine_call is None:
                ret = openai.ChatCompletion.create(**kwargs)['choices'][0]['message']['content']
            else:
                ret = self._engine_call(**kwargs) # i wish every engine should define an function to call
            # post processing
            ret = ret.replace('```', '').replace('python', '').strip()
            self._cache[kwargs] = ret
            return ret
    else:
        if kwargs in self._cache:
            print('(using cache)', end=' ')
            return self._cache[kwargs]
        else:
            ret = openai.Completion.create(**kwargs)['choices'][0]['text'].strip()
            self._cache[kwargs] = ret
            return ret
    """

    pass
