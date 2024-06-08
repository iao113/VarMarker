import tree_sitter
# from code_transform_provider import CodeTransformProvider
# from data_processing.data_instance import DataInstance
# from data_processing.dataset_processor import JsonlWMDatasetProcessor
# from runtime_data_manager import InMemoryJitRuntimeDataManager


class PythonVarnameSub:
    def __init__(self, lang: str) -> None:
        self.lang = lang
    
        # initialize parser
        self.parser = tree_sitter.Parser()
        self.parser_lang = tree_sitter.Language('./parser/languages.so', self.lang)
        self.parser.set_language(self.parser_lang)
    
    def _variable_location_collector(self, node: tree_sitter.Node, new_words: list[str], old_words: list[str]) -> list[tuple[str, str, int, int]]:
        words_map = dict(zip(old_words, new_words))
        old_words_set = set(old_words)
        variable_locations = []
        
        def _identifier_collector(node: tree_sitter.Node, is_local: bool, excepts: list[str]):
            if is_local and node.type == "identifier" and (t := node.text.decode("utf-8")) in old_words_set:
                if t not in excepts:
                    variable_locations.append((words_map[t], t, node.start_point[0], node.start_point[1]))
            if node.type == "attribute":
                _identifier_collector(node.children[0], is_local or n.type == "block", excepts)
            elif node.type == "keyword_argument":
                _identifier_collector(node.children[2], is_local or n.type == "block", excepts)
            else:
                new_excepts = []
                if node.type == "function_definition" and node.children[0].type == "def" and node.children[2].type == "parameters":
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
                if node.type == "function_definition" and node.children[0].type == "async" and node.children[1].type == "def" and node.children[3].type == "parameters":
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
                    _identifier_collector(n, is_local or n.type == "block", excepts + new_excepts)
        
        _identifier_collector(node, False, [])
        return variable_locations
    
    def substitute(self, instance_code: str, new_words: list[str], old_words: list[str]) -> tuple[str, str]:
        tree = self.parser.parse(instance_code.encode("utf-8"))
        variable_locations = self._variable_location_collector(tree.root_node, new_words, old_words)
        variable_locations = sorted(variable_locations, key=lambda x: (x[2], x[3]), reverse=True)
        code_lines = instance_code.split("\n")
        for neww, oldw, row, col in variable_locations:
            code_line = code_lines[row]
            code_lines[row] = code_line[:col] + neww + code_line[col + len(oldw):]
        return "\n".join(code_lines), instance_code


def main():
    LANG = "python"
    sub = PythonVarnameSub(LANG)
    oric = """def split_phylogeny(p, level=\"s\"):
    level = level+\"__\"
    result = p.split(level)
    return result[0]+level+result[1].split(\";\")[0]"""
    newc, _ = sub.substitute(oric, ["res"], ["result"])
    print(newc)


if __name__ == "__main__":
    main()

