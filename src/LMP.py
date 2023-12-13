import openai
from time import sleep
from openai import RateLimitError, APIConnectionError
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from utils import load_prompt, DynamicObservation, IterableDynamicObservation
import time, textwrap, inspect, re
from LLM_cache import DiskCache


class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""

    def __init__(
        self,
        name,
        cfg,
        fixed_vars,
        variable_vars,
        debug=False,
        env="rlbench",
        engine_call_fn=None,
    ):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        self._stop_tokens = list(self._cfg["stop"])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ""
        self._context = None
        # self._cache = DiskCache(load_cache=self._cfg["load_cache"])
        self._engine_call = engine_call_fn
        self._history_file_name = f"code-history.txt"

    def clear_exec_hist(self):
        self.exec_hist = ""

    def split_prompt(self, prompt):
        if self._cfg["include_context"]:
            pattern = r"(objects = .*?\n# Query:.*?(?:$|\n))"
        else:
            pattern = r"(\n# Query:.*?(?:$|\n))"
        matches = re.split(pattern, prompt, flags=re.DOTALL)
        return [matches[0] + "\n\n" + matches[1]] + matches[2:-1]

    def build_prompt(self, query):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = (
                f"from utils import {', '.join(self._variable_vars.keys())}"
            )
        else:
            variable_vars_imports_str = ""
        prompt = self._base_prompt.replace(
            "{variable_vars_imports}", variable_vars_imports_str
        )

        if self._cfg["maintain_session"] and self.exec_hist != "":
            prompt += f"\n{self.exec_hist}"

        prompt += "\n"  # separate prompted examples with the query part

        if self._cfg["include_context"]:
            assert self._context is not None, "context is None"
            prompt += f"\n{self._context}"

        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f"\n{user_query}"

        return prompt, user_query, self.split_prompt(prompt=prompt)

    def _cached_api_call(self, **kwargs):
        # check whether completion endpoint or chat endpoint is used
        if any(
            [
                chat_model in kwargs["model"]
                for chat_model in ["gpt-3.5", "gpt-4", "SparkV3", "ERNIEV4"]
            ]
        ):
            ret = self._engine_call(
                **kwargs
            )  # i wish every engine should define an function to call
            return ret
        else:
            if kwargs in self._cache:
                print("(using cache)", end=" ")
                return self._cache[kwargs]
            else:
                ret = openai.Completion.create(**kwargs)["choices"][0]["text"].strip()
                self._cache[kwargs] = ret
                return ret

    def __call__(self, query, **kwargs):
        calling_level = len(inspect.getouterframes(inspect.currentframe())) // 3
        print("calling level:", calling_level)

        prompt, user_query, splited_prompt = self.build_prompt(query)

        start_time = time.time()
        while True:
            try:
                code_str = self._engine_call(
                    prompt=(prompt, splited_prompt),
                    stop=self._stop_tokens,
                    temperature=self._cfg["temperature"],
                    model=self._cfg["model"],
                    max_tokens=self._cfg["max_tokens"],
                    use_cache=self._cfg["load_cache"],
                )
                break
            except (RateLimitError, APIConnectionError) as e:
                print(f"OpenAI API got err {e}")
                print("Retrying after 3s.")
                sleep(3)
        print(f"*** OpenAI API call took {time.time() - start_time:.2f}s ***")

        if self._cfg["include_context"]:
            assert self._context is not None, "context is None"
            to_exec = f"{self._context}\n{code_str}"
            to_log = f"{self._context}\n{user_query}\n{code_str}"
        else:
            to_exec = code_str
            to_log = f"{user_query}\n{to_exec}"

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())

        if self._cfg["include_context"]:
            print(
                "#" * 40
                + f'\n## "{self._name}" generated code\n'
                + f'## context: "{self._context}"\n'
                + "#" * 40
                + f"\n{to_log_pretty}\n"
            )
        else:
            print(
                "#" * 40
                + f'\n## "{self._name}" generated code\n'
                + "#" * 40
                + f"\n{to_log_pretty}\n"
            )

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        # return function instead of executing it so we can replan using latest obs（do not do this for high-level UIs)
        if not self._name in ["composer", "planner"]:
            to_exec = "def ret_val():\n" + to_exec.replace("ret_val = ", "return ")
            to_exec = to_exec.replace("\n", "\n    ")

        with open(self._history_file_name, "r+") as f:
            file_content = f.read()

        if calling_level == 0:
            with open(self._history_file_name, "w") as f:
                f.write(f"{to_log.strip()}\n")
        else:
            with open(self._history_file_name, "a") as f:
                f.write(textwrap.indent(f"{to_log.strip()}\n", "    " * calling_level))

        if self._debug:
            # only "execute" function performs actions in environment, so we comment it out
            action_str = ["execute("]
            try:
                for s in action_str:
                    exec_safe(to_exec.replace(s, f"# {s}"), gvars, lvars)
            except Exception as e:
                print(f"Error: {e}")
                import pdb

                pdb.set_trace()
        else:
            exec_safe(to_exec, gvars, lvars)
            """
            Local variables in Python are those which are initialized inside a function and belong only to that particular function. 
            It cannot be accessed anywhere outside the function. 
            Global Variables are those which are defined outside any function and which are accessible throughout the program, 
            i.e., inside and outside of every function.
            If we initalize a local variable (in a function) with the same name with a global varible, it just "ignore" the globle variable,
            i.e., you cant change the value of a globle variable, i.e., globle variables are readonly in the function
            """

        self.exec_hist += f"\n{to_log.strip()}"

        if self._cfg["maintain_session"]:
            self._variable_vars.update(lvars)

        if self._cfg["has_return"]:
            if self._name == "parse_query_obj":
                try:
                    # there may be multiple objects returned, but we also want them to be unevaluated functions so that we can access latest obs
                    # 似乎 detect 函数返回的对象是一个函数，这样通过调用（evaluated）这个函数，便可以得到该对象的最新信息
                    return IterableDynamicObservation(
                        lvars[self._cfg["return_val_name"]]
                    )
                except AssertionError:
                    return DynamicObservation(lvars[self._cfg["return_val_name"]])
            return lvars[self._cfg["return_val_name"]]


def merge_dicts(dicts):
    return {k: v for d in dicts for k, v in d.items()}


def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ["import", "__"]
    for phrase in banned_phrases:
        assert phrase not in code_str, "banned phrases appear in model ret"

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([gvars, {"exec": empty_fn, "eval": empty_fn}])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f"Error executing code:\n{code_str}")
        raise e
