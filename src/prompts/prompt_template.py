EXAMPLE_TEMPLATE = """
# Examples
Here are a few examples (query pair with corrosponding expected output) to get you started:
- The import block is shared among all the examples.
- Each example contains a query and the expected output.
    - Start with `# Query ...`, end with `# done`.
    - Between `# Query ...` and `# done`, there is the code you need to write.
    - Each example is separated by a blank line.
```python
{examples}
```
"""

FEEDBACK_TEMPLATE = """
# Feedback
Here is the feedback for what you executed last time (None means no feedback):
{feedback}
If error occured, please change your code to solve it.
"""

SKILL_TEMPLATE = """
# Skills
Here are skills you can use directly
{skills}
"""
 
TEMPLATE = """
# Query and Context
Query is the task your code should solve and context is some possible objects in the environment.
`None` context means no context is provided.

## Context
{context}
## Query
{query}

# Output
Your code MUST follow the following template
    - assuming all the necessary imports have been done, do NOT include any imports
    - NO query or context is shown in your code
    - NO function definition is needed
    - Only one code block is allowed
    - You need output the code in [Your code here] block, no other text is allowed.
```python
[Your code here]
```
"""


REFLECTION_TEMPLATE = """
# Feedback
Here is the feedback for what you executed last time (None means no feedback):
{feedback}

# History code
Here is the code you written last time 
{history_code}

# Query and Context
Query is the task your code should solve and context is some possible objects in the environment.
`None` context means no context is provided.
## Context
{context}
## Query
{query}

# Output
You need to output new code based on the feedback from yourself
    - assuming all the necessary imports have been done, do NOT include any imports
    - NO query or context is shown in your code
    - NO function definition is needed
    - Only one code block is allowed
    - You need output the code in [Your code here] block, no other text is allowed.
```python
[Your code here]
```
"""

ROLE_DISCRIPTION = """You are a {LMP_NAME} LMP (language model programmer), you need to write PYTHON code to {LMP_DISCRIPTION}.
There are NO comments, imports, non-code text shown in your code. You can assume all the necessary imports have been done."""
