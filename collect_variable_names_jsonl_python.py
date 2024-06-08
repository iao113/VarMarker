import random
import sys
import json
import pprint
import tree_sitter
from tqdm import tqdm
from collections import Counter
from data_processing import JsonlWMDatasetProcessor
from typing import List


def variable_collector(node: tree_sitter.Node) -> tuple[list[str], list[str]]:
    variable_names: List[str] = []
    
    def _record_aliased_import(node: tree_sitter.Node, is_local: bool, excepts: list[str]) -> None:
        after_as = False
        for n in node.children:
            if n.type == "as":
                after_as = True
            if is_local and after_as and n.type == "identifier":
                if n.text.decode("utf-8") not in variable_names and n.text.decode("utf-8") not in excepts:
                    variable_names.append(n.text.decode("utf-8"))
        assert after_as, node.text

    def _record_import_statement(node: tree_sitter.Node, is_local: bool, excepts: list[str]) -> None:
        for n in node.children:
            if n.type == "aliased_import":
                _record_aliased_import(n, is_local, excepts)

    def _record_import_from_statement(node: tree_sitter.Node, is_local: bool, excepts: list[str]) -> None:
        for n in node.children:
            if n.type == "aliased_import":
                _record_aliased_import(n, is_local, excepts)
    
    def _record_for_statement(node: tree_sitter.Node, is_local: bool, excepts: list[str]) -> None:
        after_for = False
        after_in = False
        for n in node.children:
            if n.type == "for":
                after_for = True
            if is_local and after_for and not after_in and n.type == "identifier":
                if n.text.decode("utf-8") not in variable_names and n.text.decode("utf-8") not in excepts:
                    variable_names.append(n.text.decode("utf-8"))
            if is_local and after_for and not after_in and n.type == "pattern_list":
                for i in n.children:
                    if i.type == "identifier" and i.text.decode("utf-8") not in variable_names and i.text.decode("utf-8") not in excepts:
                        variable_names.append(i.text.decode("utf-8"))
            if n.type == "in":
                after_in = True
            if after_for and after_in:
                _record_identifier(n, is_local, excepts)
        assert after_for, node.text
        assert after_in, node.text

    def _record_for_in_clause(node: tree_sitter.Node, is_local: bool, excepts: list[str]) -> None:
        after_for = False
        after_in = False
        for n in node.children:
            if n.type == "for":
                after_for = True
            if is_local and after_for and not after_in and n.type == "identifier":
                if n.text.decode("utf-8") not in variable_names and n.text.decode("utf-8") not in excepts:
                    variable_names.append(n.text.decode("utf-8"))
            if is_local and after_for and not after_in and n.type == "pattern_list":
                for i in n.children:
                    if i.type == "identifier" and i.text.decode("utf-8") not in variable_names and i.text.decode("utf-8") not in excepts:
                        variable_names.append(i.text.decode("utf-8"))
            if n.type == "in":
                after_in = True
            if after_for and after_in:
                _record_identifier(n, is_local, excepts)
        assert after_for, node.text
        assert after_in, node.text
                
    def _record_assignment(node: tree_sitter.Node, is_local: bool, excepts: list[str]) -> None:
        if is_local and node.children[0].type == "identifier":
            assert node.children[1].type in ["=", ":"], node.text
            if node.children[0].text.decode("utf-8") not in variable_names and node.children[0].text.decode("utf-8") not in excepts:
                variable_names.append(node.children[0].text.decode("utf-8"))
        if is_local and node.children[0].type == "pattern_list":
            assert node.children[1].type == "=", node.text
            for i in node.children[0].children:
                if i.type == "identifier" and i.text.decode("utf-8") not in variable_names and i.text.decode("utf-8") not in excepts:
                    variable_names.append(i.text.decode("utf-8"))
        for n in node.children[2:]:
            _record_identifier(n, is_local, excepts)
    
    def _record_class_definition(node: tree_sitter.Node, is_local: bool, excepts: list[str]) -> None:
        assert node.children[0].type == "class" and node.children[1].type == "identifier", node.text
        if is_local and node.children[0].type == "class" and node.children[1].type == "identifier":
            if node.children[1].text.decode("utf-8") not in variable_names and node.children[1].text.decode("utf-8") not in excepts:
                variable_names.append(node.children[1].text.decode("utf-8"))
        for n in node.children:
            if n.type == "block":
                _record_identifier(n, False, excepts)
    
    def _record_function_definition(node: tree_sitter.Node, is_local: bool, excepts: list[str]) -> None:
        assert (node.children[0].type == "def" and node.children[1].type == "identifier") \
            or (node.children[0].type == "async" and node.children[1].type == "def" and node.children[2].type == "identifier"), node.text
        if is_local and node.children[0].type == "def" and node.children[1].type == "identifier":
            if node.children[1].text.decode("utf-8") not in variable_names and node.children[1].text.decode("utf-8") not in excepts:
                variable_names.append(node.children[1].text.decode("utf-8"))
        if is_local and node.children[0].type == "async" and node.children[1].type == "def" and node.children[2].type == "identifier":
            if node.children[2].text.decode("utf-8") not in variable_names and node.children[2].text.decode("utf-8") not in excepts:
                variable_names.append(node.children[2].text.decode("utf-8"))
        new_excepts = []
        if node.children[0].type == "def" and node.children[2].type == "parameters":
            for n in node.children[2].children:
                if n.type == "identifier":
                    new_excepts.append(n.text.decode("utf-8"))
                if n.type == "typed_parameter":
                    assert n.children[0].type in ["identifier", "list_splat_pattern", "dictionary_splat_pattern"], node.text
                    if n.children[0].type == "identifier":
                        new_excepts.append(n.children[0].text.decode("utf-8"))
                    if n.children[0].type in ["list_splat_pattern", "dictionary_splat_pattern"]:
                        assert n.children[0].children[1].type == "identifier"
                        new_excepts.append(n.children[0].children[1].text.decode("utf-8"))
                if n.type == "default_parameter":
                    assert n.children[0].type in ["identifier", "list_splat_pattern", "dictionary_splat_pattern"], node.text
                    if n.children[0].type == "identifier":
                        new_excepts.append(n.children[0].text.decode("utf-8"))
                    if n.children[0].type in ["list_splat_pattern", "dictionary_splat_pattern"]:
                        assert n.children[0].children[1].type == "identifier"
                        new_excepts.append(n.children[0].children[1].text.decode("utf-8"))
                if n.type == "typed_default_parameter":
                    assert n.children[0].type == "identifier"
                    if n.children[0].type == "identifier":
                        new_excepts.append(n.children[0].text.decode("utf-8"))
                if n.type == "list_splat_pattern":
                    assert n.children[1].type == "identifier"
                    if n.children[1].type == "identifier":
                        new_excepts.append(n.children[1].text.decode("utf-8"))
                if n.type == "dictionary_splat_pattern":
                    assert n.children[1].type == "identifier"
                    if n.children[1].type == "identifier":
                        new_excepts.append(n.children[1].text.decode("utf-8"))
        if node.children[0].type == "async" and node.children[1].type == "def" and node.children[3].type == "parameters":
            for n in node.children[3].children:
                if n.type == "identifier":
                    new_excepts.append(n.text.decode("utf-8"))
                if n.type == "typed_parameter":
                    assert n.children[0].type in ["identifier", "list_splat_pattern", "dictionary_splat_pattern"], node.text
                    if n.children[0].type == "identifier":
                        new_excepts.append(n.children[0].text.decode("utf-8"))
                    if n.children[0].type in ["list_splat_pattern", "dictionary_splat_pattern"]:
                        assert n.children[0].children[1].type == "identifier"
                        new_excepts.append(n.children[0].children[1].text.decode("utf-8"))
                if n.type == "default_parameter":
                    assert n.children[0].type in ["identifier", "list_splat_pattern", "dictionary_splat_pattern"], node.text
                    if n.children[0].type == "identifier":
                        new_excepts.append(n.children[0].text.decode("utf-8"))
                    if n.children[0].type in ["list_splat_pattern", "dictionary_splat_pattern"]:
                        assert n.children[0].children[1].type == "identifier"
                        new_excepts.append(n.children[0].children[1].text.decode("utf-8"))
                if n.type == "typed_default_parameter":
                    assert n.children[0].type == "identifier"
                    if n.children[0].type == "identifier":
                        new_excepts.append(n.children[0].text.decode("utf-8"))
                if n.type == "list_splat_pattern":
                    assert n.children[1].type == "identifier"
                    if n.children[1].type == "identifier":
                        new_excepts.append(n.children[1].text.decode("utf-8"))
                if n.type == "dictionary_splat_pattern":
                    assert n.children[1].type == "identifier"
                    if n.children[1].type == "identifier":
                        new_excepts.append(n.children[1].text.decode("utf-8"))
        for n in node.children:
            if n.type == "block":
                _record_identifier(n, True, excepts + new_excepts)
    
    def _record_lambda(node: tree_sitter.Node, is_local: bool, excepts: list[str]) -> None:
        assert len(node.children) == 0 or (node.children[0].type == "lambda" and node.children[1].type in [":", "lambda_parameters"]), node.text
        if len(node.children) > 0 and node.children[0].type == "lambda":
            if node.children[1].type == "lambda_parameters":
                for i in node.children[1].children:
                    if i.type == "identifier" and i.text.decode("utf-8") not in variable_names and i.text.decode("utf-8") not in excepts:
                        variable_names.append(i.text.decode("utf-8"))
        for n in node.children[2:]:
            _record_identifier(n, is_local, excepts)
    
    def _record_named_expression(node: tree_sitter.Node, is_local: bool, excepts: list[str]) -> None:
        assert node.children[0].type == "identifier" and node.children[1].type == ":=", node.text
        if is_local and node.children[0].type == "identifier" and node.children[1].type == ":=":
            if node.children[0].text.decode("utf-8") not in variable_names and node.children[0].text.decode("utf-8") not in excepts:
                variable_names.append(node.children[0].text.decode("utf-8"))
        for n in node.children[2:]:
            _record_identifier(n, is_local, excepts)
    
    def _record_as_pattern(node: tree_sitter.Node, is_local: bool, excepts: list[str]) -> None:
        assert node.children[1].type == "as" and node.children[2].type == "as_pattern_target", node.text
        assert node.children[2].children[0].type in ["tuple", "identifier", "parenthesized_expression", "attribute"], node.text
        if is_local and node.children[1].type == "as" and node.children[2].type == "as_pattern_target":
            if node.children[2].children[0].type == "identifier":
                if node.children[2].children[0].text.decode("utf-8") not in variable_names and node.children[2].children[0].text.decode("utf-8") not in excepts:
                    variable_names.append(node.children[2].children[0].text.decode("utf-8"))
            if node.children[2].children[0].type in ["tuple", "parenthesized_expression"]:
                for n in node.children[2].children[0].children:
                    if n.type == "identifier" and n.text.decode("utf-8") not in variable_names and n.text.decode("utf-8") not in excepts:
                        variable_names.append(n.text.decode("utf-8"))
        _record_identifier(node.children[0], is_local, excepts)
    
    def _record_type_alias_statement(node: tree_sitter.Node, is_local: bool, excepts: list[str]) -> None:
        assert node.children[0].type == "type" and node.children[1].type == "type", node.text
        assert node.children[1].children[0].type == "identifier", node.text
        if is_local and node.children[0].type == "type" and node.children[1].type == "type":
            if node.children[1].children[0].type == "identifier":
                if node.children[1].children[0].text.decode("utf-8") not in variable_names and node.children[1].children[0].text.decode("utf-8") not in excepts:
                    variable_names.append(node.children[1].children[0].text.decode("utf-8"))
        for n in node.children[2:]:
            _record_identifier(n, is_local, excepts)
    
    def _record_identifier(node: tree_sitter.Node, is_local: bool, excepts: list[str]) -> None:
        if node.type == "import_statement":  # import ... as (n)
            _record_import_statement(node, is_local, excepts)
        elif node.type == "import_from_statement":  # from ... import ... as (n)
            _record_import_from_statement(node, is_local, excepts)
        elif node.type == "for_statement":  # for (n), (n) in ...
            _record_for_statement(node, is_local, excepts)
        elif node.type == "for_in_clause":  # [... for (n), (n) in ...]
            _record_for_in_clause(node, is_local, excepts)
        elif node.type == "assignment":  # (n), (n) = ..., (n): ..., (n): ... = ...
            _record_assignment(node, is_local, excepts)
        elif node.type == "class_definition":  # class (n)...
            _record_class_definition(node, is_local, excepts)
        elif node.type == "function_definition":  # def (n)(...)
            _record_function_definition(node, is_local, excepts)
        elif node.type == "lambda":  # lambda (n), (n): ...
            _record_lambda(node, is_local, excepts)
        elif node.type == "named_expression":  # (n) := ...
            _record_named_expression(node, is_local, excepts)
        elif node.type == "as_pattern":  # with ... as (n), ... as (n): ..., except ... as (n): ...
            _record_as_pattern(node, is_local, excepts)
        # elif node.type == "type_alias_statement":  # type (n) = ...
        #     _record_type_alias_statement(node, is_local, excepts)
        else:
            if len(node.children) == 0:
                return
            for n in node.children:
                _record_identifier(n, is_local, excepts)

    def _variable_collector(node: tree_sitter.Node):
        assert node.type == 'module'
        for child in node.children:
            assert child is not None
            _record_identifier(child, False, [])

    _variable_collector(node)
    
    identifiers = []
    
    def _identifier_collector(node: tree_sitter.Node):
        if node.type == "identifier":
            identifiers.append(node.text.decode("utf-8"))
        for n in node.children:
            _identifier_collector(n)
    
    _identifier_collector(node)
    
    if variable_names == 0:
        return ["a"], []
    return variable_names, list(set(identifiers) - set(variable_names))


def main(args):
    if len(args) != 1:
        print('Usage: python collect_variable_names_jsonl.py <dataset>')
        return

    DATASET = args[0]
    if DATASET in {'csn_python', 'mbpp'}:
        LANG = 'python'
    else:
        raise ValueError(f'Unknown dataset: {DATASET}')

    DATA_DIR = f'./datasets/{DATASET}'

    parser = tree_sitter.Parser()
    parser_lang = tree_sitter.Language('./parser/languages.so', LANG)
    parser.set_language(parser_lang)

    data_processor = JsonlWMDatasetProcessor(LANG)
    if DATASET in {'mbpp'}:
        all_instances = data_processor._load_jsonl_fast(DATA_DIR, split='test')
    else:
        instances = data_processor.load_jsonls_fast(DATA_DIR, show_progress=False)
        train_instances = instances['train']
        valid_instances = instances['valid']
        test_instances = instances['test']

        all_instances = train_instances + valid_instances + test_instances

    all_variable_names = set()
    variable_names_per_file = dict()
    identifiers_per_file = dict()

    for instance in tqdm(all_instances):
        code = instance.source
        tree = parser.parse(code.encode("utf-8"))
        variable_names, identifiers = variable_collector(tree.root_node)
        variable_names_per_file[instance.id] = dict(Counter(variable_names))
        identifiers_per_file[instance.id] = dict(Counter(identifiers))
        all_variable_names.update(variable_names)
        if random.random() < 0.00001:
            print(code)
            print(variable_names)
            input()

    res_dict = {
        'all_variable_names': list(all_variable_names),
        'variable_names_per_file': variable_names_per_file,
        'identifiers_per_file': identifiers_per_file,
    }

    json.dump(res_dict, open(f'./datasets/variable_names_{DATASET}.json', 'w'), indent=2)
    variable_names_per_file = res_dict['variable_names_per_file']

    var_counts = []
    tot_files = 0
    for instance_id, instance_list in variable_names_per_file.items():
        var_counts.append(len(instance_list))
        tot_files += 1

    var_counter = Counter(var_counts)
    pprint.pp(var_counter)

    tot_vars = sum(var_counts)
    print(f'total files: {tot_files}')
    print(f'total var names: {len(res_dict["all_variable_names"])}')
    print(f'total var count: {tot_vars}')
    print(f'avg vars per file: {tot_vars / tot_files:.2f}')


if __name__ == '__main__':
    main(sys.argv[1:])

