import json
import requests

headers = {"Content-Type": "application/json"}

def text_generation(data, ip='localhost', port=None):
    resp = requests.put(f'http://{ip}:{port}/generate', data=json.dumps(data), headers=headers)
    return resp.json()


def get_generation(prompt, greedy, add_BOS, token_to_gen, min_tokens, temp, top_p, top_k, repetition, batch=False):
    data = {
        "sentences": [prompt] if not batch else prompt,
        "tokens_to_generate": int(token_to_gen),
        "temperature": temp,
        "add_BOS": add_BOS,
        "top_k": top_k,
        "top_p": top_p,
        "greedy": greedy,
        "all_probs": False,
        "repetition_penalty": repetition,
        "min_tokens_to_generate": int(min_tokens),
        "end_strings": ["<|endoftext|>", "<extra_id_1>", "\x11", "<extra_id_1>User"],
    }
    sentences = text_generation(data, port=1424)['sentences']
    return sentences[0] if not batch else sentences

PROMPT_TEMPLATE = """<extra_id_0>System

<extra_id_1>User
{prompt}
<extra_id_1>Assistant
"""

question = "Write a poem on NVIDIA in the style of Shakespeare"
prompt = PROMPT_TEMPLATE.format(prompt=question)
print(prompt)

response = get_generation(prompt, greedy=True, add_BOS=False, token_to_gen=1024, min_tokens=1, temp=1.0, top_p=1.0, top_k=0, repetition=1.0, batch=False)
response = response[len(prompt):]
if response.endswith("<extra_id_1>"):
    response = response[:-len("<extra_id_1>")]
print(response)


question = """
Here is an extract from a python coding tutorial: 
```
import numpy as np from scipy.linalg import solve from helpers import Vehicle, show_trajectory A = np.random.random((3, 3)) b = np.random.random(3) x = solve(A, b) def JMT(start, end, T): si = start[0] si_dot = start[1] si_double_dot = start[2] sf = end[0] sf_dot = end[1] sf_double_dot = end[2] A = [ [T**3 , T**4 , T**5 ], [3*T**2, 4*T**3 , 5*T**4], [6*T , 12*T**2, 20*T**3] ] b = [sf - (si + si_dot*T + 0.5 * si_double_dot * T**2 ), sf_dot - (si_dot + si_double_dot*T), sf_double_dot - si_double_dot] x = solve(A,b) r = np.array([si, si_dot, 0.5*si_double_dot, x[0], x[1], x[2]]) return r #return [si, si_dot, 0.5*si_double_dot, x[0], x[1], x[2]] def close_enough(poly, target_poly, eps=0.01): if len(poly) != len(target_poly): print("your solution didn't have the correct number of terms") return False for i in range(len(poly)): diff = poly[i]-target_poly[i] if(abs(diff) > eps): print("at least one of your terms differed from target by more than ", eps ) return False return True answers = [[0.0, 10.0, 0.0, 0.0, 0.0, 0.0],[0.0,10.0,0.0,0.0,-0.625,0.3125],[5.0,10.0,1.0,-3.0,0.64,-0.0432]]; # create test cases tc1 = [[0,10,0], [10,10,0], [1]] tc2 = [[0,10,0],[20,15,20], [2]] tc3 = [[5,10,2],[-30,-20,-4], [5]] tc = [tc1,tc2,tc3] i = 0; total_correct = True for test in tc: jmt = JMT(test[0], test[1], test[2][0]) correct = close_enough(jmt, answers[i]) total_correct &= correct i += 1 if total_correct: print("Nice work!") else: print("Try again!") jmt = JMT(tc1[0], tc1[1], tc1[2][0]) show_trajectory(jmt[0], jmt[1], jmt[2])
```

Write an extensive and detailed textbook with interleaved text and code snippets related to the extract above. The textbook should promote reasoning and basic algorithmical skills, it should be suitable for an audience getting started with Python. Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth.
Try to:
- Ensure in-depth coverage of the concepts.
- Use a narrative thought-provoking style.
- Use LaTeX notation $$ for equations and ``` for Python code snippets. 
- Ensure valid Markdown output.
Do not include a title, introductory phrases or images. Do not use html for formatting. Write the content directly.
"""

prompt = PROMPT_TEMPLATE.format(prompt=question)
print(prompt)

response = get_generation(prompt, greedy=True, add_BOS=False, token_to_gen=1024, min_tokens=1, temp=1.0, top_p=1.0, top_k=0, repetition=1.0, batch=False)
response = response[len(prompt):]
if response.endswith("<extra_id_1>"):
    response = response[:-len("<extra_id_1>")]
print(response)