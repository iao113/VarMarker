# Modify from https://github.com/YBRua/SrcMarker
import os
import re
import sys
import json
import tree_sitter
from tqdm import tqdm
from collections import Counter
from mutable_tree.adaptors import JavaAdaptor, CppAdaptor, PythonAdaptor
from typing import List


def collect_tokens(root: tree_sitter.Node) -> List[str]:
    tokens: List[str] = []

    def _collect_tokens(node: tree_sitter.Node):
        if node.child_count == 0:
            tokens.append(node.text.decode())

        for ch in node.children:
            _collect_tokens(ch)

    _collect_tokens(root)
    return tokens


def remove_comments(source: str):
    def replacer(match):
        s = match.group(0)
        if s.startswith('#'):
            return ""  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(r'^ *#.*?$', re.DOTALL | re.MULTILINE)
    temp = []
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)


def remove_line_continuation(source: str):
    def replacer(match):
        return " "

    pattern = re.compile(r' *\\\n *', re.DOTALL | re.MULTILINE)
    temp = []
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)


def get_python_function_root(root: tree_sitter.Node):
    assert root.type == 'module'
    func_root_node = root.children[0]
    assert func_root_node.type == 'function_definition', func_root_node.type
    return func_root_node


def compare_tokens(old_tokens: List[str], new_tokens: List[str]):
    if len(old_tokens) != len(new_tokens):
        return False
    for old_token, new_token in zip(old_tokens, new_tokens):
        if old_token != new_token:
            return False
    return True

def print_node(node, indent=""):
    print(f"{indent}{node}")
    if len(node.children) == 0:
        return
    for n in node.children:
        print_node(n, indent+" ")

def function_round_trip_python(parser: tree_sitter.Parser, code: str, lang: str):
    wrapped = code

    tree = parser.parse(wrapped.encode('utf-8'))
    if tree.root_node.has_error:
        print_node(tree.root_node)
        return (False, 'original code has error')

    assert not tree.root_node.has_error
    func_root = get_python_function_root(tree.root_node)
    try:
        mutable_root = PythonAdaptor.convert_function_definition(func_root)
    except Exception as e:
        return (False, str(e))

    new_code = mutable_root.to_string()

    new_root = parser.parse(new_code.encode('utf-8')).root_node
    if new_root.has_error:
        print_node(new_root)
        return (False, 'new code has error')

    old_tokens = collect_tokens(tree.root_node)
    new_tokens = collect_tokens(new_root)
    if not compare_tokens(old_tokens, new_tokens):
        return (False, 'token mismatch')

    return (True, None)


def main(args):
    if len(args) != 2:
        print('Usage: python3 dataset_filter.py <lang> <file>')
        return
    lang, file_path = args

    parser = tree_sitter.Parser()
    parser.set_language(tree_sitter.Language('./parser/languages.so', lang))

    errors = []
    tot_good = 0
    tot_fail = 0
    tot_funcs = 0

    with open(file_path, 'r', encoding='utf-8') as fi:
        lines = fi.readlines()
        data_instances = [json.loads(line) for line in lines]

    filename_noext = os.path.splitext(os.path.basename(file_path))[0]
    output_file_path = os.path.join(os.path.dirname(file_path),
                                    f'{filename_noext}_filtered.jsonl')
    with open(output_file_path, 'w', encoding='utf-8') as fo:
        for data_instance in tqdm(data_instances):
            code = data_instance['original_string']
            code = remove_line_continuation(code)
            code = remove_comments(code)

            success, msg = function_round_trip_python(parser, code, lang)
            if not success:
                tot_fail += 1
                errors.append(msg)
                if "match[" not in code and "List[bytes] or List[str]" not in code and "# \\\\?\\[" not in code:
                    print(data_instance['original_string'])
                    print(code)
                    print(msg)
                    input()
                elif "match[" in code:
                    print("match[] error")
                else:
                    print("type hint error")
            else:
                data_instance['original_string'] = code
                fo.write(json.dumps(data_instance) + '\n')
                tot_good += 1
            tot_funcs += 1

        msg_counter = Counter(errors)
        print(f'Good: {tot_good}, Fail: {tot_fail}, Total: {tot_funcs}')
        for msg, count in msg_counter.items():
            print(f'{msg}: {count}')


if __name__ == '__main__':
    main(sys.argv[1:])
