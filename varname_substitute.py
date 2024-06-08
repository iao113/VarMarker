# Modify from https://github.com/YBRua/SrcMarker
import json
import os
import tree_sitter
from code_transform_provider import CodeTransformProvider
from data_processing.data_instance import DataInstance
from data_processing.dataset_processor import JsonlWMDatasetProcessor
from runtime_data_manager import InMemoryJitRuntimeDataManager


class VarnameSub:
    def __init__(self, lang: str, dataset: str, dataset_dir: str, fast: bool) -> None:
        self.lang = lang
        self.dataset = dataset
        self.dataset_dir = dataset_dir

        # load datasets
        self.dataset_processor = JsonlWMDatasetProcessor(lang=lang)
        if self.dataset == "mbjp" or fast:
            self.test_instances = self.dataset_processor.load_jsonl(dataset_dir, split='test')
            self.all_instances = self.test_instances
        else:
            self.instance_dict = self.dataset_processor.load_jsonls(dataset_dir)
            self.train_instances = self.instance_dict['train']
            self.valid_instances = self.instance_dict['valid']
            self.test_instances = self.instance_dict['test']
            self.all_instances = self.train_instances + self.valid_instances + self.test_instances

        self.vocab = self.dataset_processor.build_vocab(self.all_instances)
    
        # initialize transform computers
        self.parser = tree_sitter.Parser()
        self.parser_lang = tree_sitter.Language('./parser/languages.so', lang)
        self.parser.set_language(self.parser_lang)
        self.code_transformers = []
        self.transform_computer = CodeTransformProvider(lang, self.parser, self.code_transformers)
        self.transform_manager = InMemoryJitRuntimeDataManager(self.transform_computer,
                                                        self.all_instances,
                                                        lang=lang)
        self.transform_manager.register_vocab(self.vocab)
        self.transform_manager.load_transform_mask(f'datasets/feasible_transform_{dataset}.json')
        self.transform_manager.load_varname_dict(f'datasets/variable_names_{dataset}.json')
    
    def substitute(self, instance_id: str, new_words: list[str], old_words: list[str]) -> tuple[DataInstance, DataInstance]:
        ori_instance = self.transform_manager.get_original_instance(instance_id)
        # new_word = transform_manager.vocab.get_token_by_id(word_pred)
        # old_word = transform_manager.vocab.get_token_by_id(old_pred)
        tmp_instance = ori_instance
        for new_word, old_word in zip(new_words, old_words):
            tmp_instance, update = self.transform_manager._jit_varname_substitution(tmp_instance, new_word, old_word, mode='replace')
        new_instance = tmp_instance
        return new_instance, ori_instance



def main():
    LANG = "java"
    DATASET = "csn_java"
    DATASET_DIR = "datasets/csn_java"
    ORIGINAL_DIR = "data/java_def"
    klass = "test"
    WATERMARKED_DIR = f"data/output_SW10_java_8b_10t_varl_mask_random05_oneturn_full_msgjava_def_{klass}"
    SAVE_DIR = "datasets/dewatermark/java_8b"
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    sub = VarnameSub(LANG, DATASET, DATASET_DIR, False)
    
    with open(os.path.join(WATERMARKED_DIR, "name.txt"), "r") as f:
        new_words = [x.split() for x in f.readlines()]
    with open(os.path.join(ORIGINAL_DIR, f"{klass}.txt"), "r") as f:
        old_words = [x.split() for x in f.readlines()]

    with open(os.path.join(SAVE_DIR, f"{klass}.jsonl"), "w") as f:
        for i, zi in enumerate(zip(old_words, new_words)):
            old_word, new_word = zi
            instance_id = f"{klass}#{i}"
            new_instance, ori_instance = sub.substitute(instance_id, new_word, old_word)

            f.write(json.dumps({
                "original_string": ori_instance.source,
                "after_watermark": new_instance.source,
            }) + "\n")


if __name__ == "__main__":
    main()

