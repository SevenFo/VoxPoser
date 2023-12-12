import _thread as thread
import base64
import datetime
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


class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(Spark_url).netloc
        self.path = urlparse(Spark_url).path
        self.Spark_url = Spark_url

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(
            self.APISecret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding="utf-8")

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(
            encoding="utf-8"
        )

        # 将请求的鉴权参数组合为字典
        v = {"authorization": authorization, "date": date, "host": self.host}
        # 拼接鉴权参数，生成url
        url = self.Spark_url + "?" + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url


class Spark:
    def __init__(self, **kwargs) -> None:
        websocket.enableTrace(False)
        # kwargs may contain appid,api_key,api_secret,url,doman
        self._appid = kwargs["secret"]["appid"]
        self._params = Ws_Param(
            APPID=self._appid,
            APIKey=kwargs["secret"]["api_key"],
            APISecret=kwargs["secret"]["api_secret"],
            Spark_url=kwargs["url"],
        )
        self._domain = kwargs["domain"]
        self._max_tokens = 512
        self.answer = ""
        self.message = []

    # 收到websocket错误的处理
    def on_error(self, ws, error):
        print("### error:", error)

    # 收到websocket关闭的处理
    def on_close(self, ws, one, two):
        print(" ")

    # 收到websocket连接建立的处理
    def on_open(self, ws):
        thread.start_new_thread(self.run, (ws,))

    def run(self, ws, *args):
        data = json.dumps(self.gen_params())
        # print(data)
        ws.send(data)

    # 收到websocket消息的处理
    def on_message(self, ws, message):
        data = json.loads(message)
        code = data["header"]["code"]
        if code != 0:
            print(f"请求错误: {code}, {data}")
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            # print(content,end ="")
            self.answer += content
            # print(1)
            if status == 2:
                ws.close()

    def gen_params(self):
        """
        通过appid和用户的提问来生成请参数
        """
        data = {
            "header": {"app_id": self._appid, "uid": "1234"},
            "parameter": {
                "chat": {
                    "domain": self._domain,
                    "temperature": 1.0,
                    "max_tokens": self._max_tokens,
                }
            },
            "payload": {"message": {"text": self.message}},
        }
        return data

    def extract_content(self, text):
        pattern = r"```python\s(.*?)\s```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    def __call__(self, **kwargs):
        self.answer = ""
        assert "messages" in kwargs.keys(), "engine call kwargs not contain messages"
        # add other params
        if "max_token" in kwargs.keys():
            self._max_tokens = kwargs["max_tokens"]
        for m in kwargs["messages"]:
            assert (
                "role" in m.keys() and "content" in m.keys()
            ), "messages not contain role or content"
            if not (m["role"] == "user" or m["role"] == "assistant"):
                continue
            self.message.append(m)
        assert len(self.message) > 0, "message is empty"
        wsUrl = self._params.create_url()
        ws = websocket.WebSocketApp(
            wsUrl,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
        )
        ws.run_forever(
            sslopt={"cert_reqs": ssl.CERT_NONE}
        )  # loop until all messages received
        code_segments = self.extract_content(self.answer)
        self.message = []
        if len(code_segments) > 0:
            return code_segments[0]
        else:
            return self.answer


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
