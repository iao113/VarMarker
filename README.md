# VARMARKER

Following SrcMarker(https://github.com/YBRua/SrcMarker) to initial the CSN-Java, Github-C, MBJP and MBCPP datasets.
Initial the CSN-Python and MBPP datasets in a similar way by using the ```collect_variable_names_jsonl_python.py```.

Use ```datasets_formater.py``` to prepare the datasets to train.

Use ```main_train_dropout.py``` to train the models.

Use ```main_evaluate.py``` to embed watermarks in variable names, then use ```varname_substitute.py``` and ```varname_substitute_python.py``` to substitute variable names in code.

Use ```main_evaluate_decode.py``` for extracting watermarks only.
