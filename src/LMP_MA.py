import warnings
import time
import textwrap
import inspect
import re
import openai
import logging
import os
from time import sleep
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

from utils import load_prompt, DynamicObservation, IterableDynamicObservation
from prompts import TEMPLATE, ROLE_DISCRIPTION, EXAMPLE_TEMPLATE, FEEDBACK_TEMPLATE,REFLECTION_TEMPLATE, SKILL_TEMPLATE


logger = logging.getLogger(__name__)


class StateError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
        
class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""

    def __init__(
        self,
        name,
        cfg,
        fixed_vars,
        variable_vars,
        skill_prompts = None,
        debug=False,
        env="pyrep_quadcopter",
        engine_call_fn=None,
        log_dir = "./"
    ):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        self._stop_tokens = list(self._cfg["stop"])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        # self._exec_hist = ""
        self._context = None
        self._engine_call = engine_call_fn
        self._history_file_name = os.path.join(log_dir,"code-history.txt")
        self._system_prompt = ROLE_DISCRIPTION.format(
            LMP_NAME=self._name, LMP_DISCRIPTION=self._cfg["discription"]
        )
        self._examples = EXAMPLE_TEMPLATE.format(examples=self._base_prompt)
        self._skills_prompt= SKILL_TEMPLATE.format(skills = load_prompt(f"{env}/skills.txt")) if skill_prompts is None else skill_prompts
        self._feedback = None
        self._memory = [
            {
                "role": "system",
                "content": self._system_prompt,
            },
            {
                "role": "user",
                "content": self._skills_prompt + '\n' + self._examples,
            },
            {
                "role": "system",
                "content": "OK, let's start coding!",
            },
        ]

    def clear_exec_hist(self):
        self._exec_hist = ""

    def split_prompt(self, prompt):
        if self._cfg["include_context"]:
            pattern = r"(objects = .*?\n# Query:.*?(?:$|\n))"
        else:
            pattern = r"(\n# Query:.*?(?:$|\n))"
        matches = re.split(pattern, prompt, flags=re.DOTALL)
        return [matches[0] + "\n\n" + matches[1]] + matches[2:-1]

    def build_prompt(self, query):
        prompt = TEMPLATE.format(
            # feedback=self._feedback,
            context=self._context if self._cfg["include_context"] else "None",
            query=query,
        )

        message = {"role": "user", "content": prompt}

        if self._cfg["maintain_session"]:
            self._memory.append(message)

        return message
    
    def build_feedback_prompt(self, query,history_code):
        prompt = REFLECTION_TEMPLATE.format(
            feedback=self._feedback,
            context=self._context if self._cfg["include_context"] else "None",
            query=query,
            history_code = history_code
        )
        message = {"role": "user", "content": prompt}

        if self._cfg["maintain_session"]:
            self._memory.append(message)

        return message
    
    def __call__(self, query, **kwargs):
        use_feedback = False
        history_code = None
        for try_count in range(5):
            calling_level = len(inspect.getouterframes(inspect.currentframe())) // 3
            print(f"[LMP.py | {self._name}] calling level:", calling_level)

            query_ = self._cfg["query_prefix"] + query + self._cfg["query_suffix"]
            message = self.build_prompt(query_) if not use_feedback else self.build_feedback_prompt(query,history_code)
            
            print(
                "-" * 40
                + f'\n## "{self._name}" latest prompt (maintain_session: {self._cfg["maintain_session"]})\n'
                + "-" * 40
                + f"\n{message}\n"
            )

            start_time = time.time()
            while True:
                try:
                    code_str = self._engine_call(
                        messages=self._memory
                        if self._cfg["maintain_session"]
                        else self._memory + [message],
                        stop=self._stop_tokens,
                        temperature=self._cfg["temperature"],
                        model=self._cfg["model"],
                        max_tokens=self._cfg["max_tokens"],
                        use_cache=self._cfg["load_cache"],
                    )
                    break
                except Exception as e:
                    print(f"LLM API got err {type(e)}: {e}")
                    print("Retrying after 3s.")
                    sleep(3)
            print(f"*** LLM API call took {time.time() - start_time:.2f}s ***")

            if self._cfg["maintain_session"]:
                self._memory.append({"role": "assistant", "content": code_str})

            if self._cfg["include_context"]:
                assert self._context is not None, "context is None"
                to_exec = f"{self._context}\n{code_str}"
                to_log = f"{self._context}\n{query_}\n{code_str}"
            else:
                to_exec = code_str
                to_log = f"{query_}\n{to_exec}"
            
            history_code = code_str

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
            if self._name not in ["composer", "planner"]:
                to_exec = "def ret_val():\n" + to_exec.replace("ret_val = ", "return ")
                to_exec = to_exec.replace("\n", "\n    ")

            if calling_level == 0:
                with open(self._history_file_name, "w+") as f:
                    f.write(f"{to_log.strip()}\n")
            else:
                with open(self._history_file_name, "a") as f:
                    f.write(
                        textwrap.indent(f"{to_log.strip()}" + "\n", "    " * calling_level)
                    )

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
                try:
                    exec_safe(to_exec, gvars, lvars)
                    
                    if self._cfg["maintain_session"]:
                        self._variable_vars.update(lvars)

                    print(f"[LMP.py | {self._name}] calling end")
                    if self._cfg["has_return"]:
                        if self._name == "parse_query_obj":
                            try:
                                # there may be multiple objects returned, but we also want them to be unevaluated functions so that we can access latest obs
                                # 由于parse_query_obj会被包装成一个函数，这样通过调用（evaluated）这个函数，便可以得到该对象的动态最新信息，（需要时再detect）
                                return IterableDynamicObservation(
                                    lvars[self._cfg["return_val_name"]]
                                )
                            except AssertionError:
                                return DynamicObservation(lvars[self._cfg["return_val_name"]])
                        return lvars[self._cfg["return_val_name"]]
                    return
                except StateError as e:
                    print(f"Stats error executing code:\n{to_exec}")
                    print(f"Error: {e}")
                    self._feedback = f"Other skills or LMP executed failed, reason: {e}\n You need to solve this problem"
                    use_feedback = True
                    continue
                except Exception as e:
                    print(f"Error executing code:\n{to_exec}")
                    print(f"Error: {e}")
                    self._feedback = f"Error: {e}"
                    use_feedback = True
                    continue
                    """
                    Local variables in Python are those which are initialized inside a function and belong only to that particular function. 
                    It cannot be accessed anywhere outside the function. 
                    Global Variables are those which are defined outside any function and which are accessible throughout the program, 
                    i.e., inside and outside of every function.
                    If we initalize a local variable (in a function) with the same name with a global varible, it just "ignore" the globle variable,
                    i.e., you cant change the value of a globle variable, i.e., globle variables are readonly in the function
                    """
        raise StateError(f"Unable to solve problem in {self._name}, feedback: {self._feedback}")



def merge_dicts(dicts):
    return {k: v for d in dicts for k, v in d.items()}


def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ["import ", "__"]
    for phrase in banned_phrases:
        assert phrase not in code_str, "banned phrases appear in model ret"
    
    if gvars is None:
        warnings.warn("gvars is None, creating empty dict")
        gvars = {}
    if lvars is None:
        warnings.warn("lvars is None, creating empty dict")
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([gvars, {"exec": empty_fn, "eval": empty_fn}])
    # try:
    exec(code_str, custom_gvars, lvars)
    # except Exception as e:
    #     print(f"Error executing code:\n{code_str}")
    #     raise e
