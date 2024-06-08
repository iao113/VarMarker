import random
import traceback

STEP = 1
LANG = "java"
RANGE = 100000000 + 1
klass = "test"
SUFFIX = "def"
BIT = 8
# OUTPUT = f"output_SW10_{LANG}_{BIT}b_10t_varl_mask_random05_oneturn_full_msg{LANG}_{SUFFIX}_{klass}"
OUTPUT = f"decode_SW10_{LANG}_8b_10t_varl_mask_random05_oneturn_full_msg_{LANG}_def_t8III50"
DATATEST = f"data/{LANG}_{SUFFIX}/{klass}.txt"
# UNKTOKEN = f"data/{LANG}_def/unktoken.txt"
# DATATEST = f"data/{LANG}test_4b/{klass}.txt"
UNKTOKEN = f"data/{LANG}_{SUFFIX}/unktoken.txt"

def show():
    for i in range(0, RANGE, STEP):
        q = input()
        if q: break
        with open(f'data/{OUTPUT}/data/data{i}.txt', 'r') as f:
            print(f.read().replace('Ġ', ' ').replace('Ċ', '').replace('<eos>', '\n'))
        with open(f'data/{OUTPUT}/data_out/data_out{i}.txt', 'r') as f:
            print(f.read().replace('Ġ', ' ').replace('Ċ', '').replace('<eos>', '\n'))
        with open(f'data/{OUTPUT}/msgs/msgs{i}.txt', 'r') as f:
            print([round(float(x)) for x in f.read().split()])
        with open(f'data/{OUTPUT}/msgs_out_sig/msgs_out_sig{i}.txt', 'r') as f:
            print([round(float(x)) for x in f.read().split()])


def show1():
    for i in range(0, RANGE, STEP):
        q = input()
        if q: break
        with open(f'data/{OUTPUT}/data/data{i}.txt', 'r') as f1, open(f'data/{OUTPUT}/data_out/data_out{i}.txt', 'r') as f2:
            d1 = f1.read().replace('Ġ', ' ').replace('Ċ', '').replace('<eos>', '\n').split("\n")
            d2 = f2.read().replace('Ġ', ' ').replace('Ċ', '').replace('<eos>', '\n').split("\n")
            for di1, di2 in zip(d1, d2):
                print(di1.strip(), di2.strip(), sep="\n")
        with open(f'data/{OUTPUT}/msgs/msgs{i}.txt', 'r') as f:
            print([round(float(x)) for x in f.read().split()])
        with open(f'data/{OUTPUT}/msgs_out_sig/msgs_out_sig{i}.txt', 'r') as f:
            print([round(float(x)) for x in f.read().split()])


def split():
    with open(DATATEST, "r") as f:
        testcases = [list(filter(lambda x: x, x.strip().split())) for x in f.readlines()]
    with open(UNKTOKEN, "r") as f:
        unktoken = [x.strip() for x in f.readlines()]
    testlens = [len(x) for x in testcases]
    testid = 0
    with open(f'data/{OUTPUT}/name.txt', "w") as rf:
        for i in range(0, RANGE, STEP):
            try:
                with open(f'data/{OUTPUT}/data/data{i}.txt', 'r') as of, open(f'data/{OUTPUT}/data_out/data_out{i}.txt', 'r') as f:
                    od = list(filter(lambda x: x, of.read().split()))
                    d = list(filter(lambda x: x, f.read().split()))
                for j in range(len(d)):
                    if d[j] == "<unk>":
                        d[j] = random.choice(unktoken)
                while d:
                    if testlens[testid] <= len(od):
                        for j in range(testlens[testid]):
                            if od[j] == "<unk>":
                                testcases[testid][j] = "<unk>"
                        assert od[:testlens[testid]] == testcases[testid], (i, od[:testlens[testid]], testcases[testid])
                        rf.write(" ".join(d[:testlens[testid]]) + "\n")
                        d = d[testlens[testid]:]
                        od = od[testlens[testid]:]
                    else:
                        for j in range(len(od)):
                            if od[j] == "<unk>":
                                testcases[testid][j] = "<unk>"
                        assert od == testcases[testid][:len(d)], f"{i}, {od}, {testcases[testid][:len(d)]}"
                        rf.write(" ".join(d + testcases[testid][len(d):]) + "\n")
                        d = []
                        od = []
                    testid += 1
            except:
                traceback.print_exc()
                break
                


def calc():
    total = 0
    total_adv = 0
    total_true = 0
    total_true_array = [0 for _ in range(100)]
    total_false = 0
    adv_true = 0
    adv_false = 0
    total_line = 0
    total_msg_true = 0
    total_msg_true_array = [0 for _ in range(100)]
    total_single = 0
    total_file = 0
    diff_file = set()
    tokens = set()
    diff_token = [set() for _ in range(100)]
    diff_all = [0 for _ in range(100)]
    diff_single = [0 for _ in range(100)]
    token_length = [0 for _ in range(100)]
    same = [[0 for _ in range(100)] for _ in range(100)]
    msg_length = 100000000
    for i in range(0, RANGE, STEP):
        try:
            with open(f'data/{OUTPUT}/data_out/data_out{i}.txt', 'r') as f2:
                d2 = f2.read().replace('Ġ', ' ').replace('Ċ', '').replace('<eos>', '\n').split("\n")
                total_file += 1
                diff_file.add(" ".join(d2).strip())
                dtoken = " ".join(d2).strip().split(" ")
                token_length[len(dtoken) - 1] += 1
                # assert len(dtoken) == 10, f"{len(dtoken)}"
                for ii, di in enumerate(dtoken):
                    diff_token[ii].add(di)
                    diff_all[ii] += 1
                    tokens.add(di)
                total_line += 1
                if len(dtoken) == len(set(dtoken)):
                    total_single += 1
                    diff_single[len(dtoken) - 1] += 1
                for x in range(len(dtoken)):
                    for y in range(len(dtoken)):
                        if dtoken[x] == dtoken[y]:
                            same[x][y] += 1
            with open(f'data/{OUTPUT}/msgs/msgs{i}.txt', 'r') as f:
                msgs = [round(float(x)) for x in f.read().split()]
            # with open(f'data/{OUTPUT}/msgs_adv/msgs_adv{i}.txt', 'r') as f:
            #     msgs_adv = [round(float(x)) for x in f.read().split()]
            with open(f'data/{OUTPUT}/msgs_out_sig/msgs_out_sig{i}.txt', 'r') as f:
                msgs_out = [round(float(x)) for x in f.read().split()]
            # with open(f'data/{OUTPUT}/msgs_out_sig/msgs_out_sig_adv{i}.txt', 'r') as f:
            #     msgs_out_adv = [round(float(x)) for x in f.read().split()]
            all_true = True
            msg_length = min(msg_length, len(msgs))
            for m0, m1 in zip(msgs, msgs_out):
                total += 1
                if m0 == m1:
                    total_true += 1
                    total_true_array[len(dtoken) - 1] += 1
                else:
                    total_false += 1
                    all_true = False
            # for m0, m1 in zip(msgs_adv, msgs_out_adv):
            #     total_adv += 1
            #     if m0 == m1:
            #         adv_true += 1
            #     else:
            #         adv_false += 1
            if all_true:
                total_msg_true += 1
                total_msg_true_array[len(dtoken) - 1] += 1
        except:
            traceback.print_exc()
            break
    assert total == total_true + total_false
    with open(DATATEST, "r") as f:
        freadlines = len(f.readlines())
        print(f"Bits: {msg_length * total_line / freadlines}, {msg_length} {total_line} {freadlines}")
    print(f"Accuracy: {total_true / total}, {total_true}, {total}")
    # print(f"Adv Accuracy: {adv_true / total_adv}, {adv_true}, {total_adv}")
    print(f"Msg Accuracy: {total_msg_true / total_line}, {total_msg_true}, {total_line}")
    print(f"Right: {total_single / total_line}, {total_single}, {total_line}")
    print(f"Same: {same}")
    print(f"Diff: {len(diff_file) / total_file}, {len(diff_file)}, {total_file}")
    for i in range(50):
        print(f"Bit Acc {i}: {total_true_array[i] / (token_length[i] * msg_length) if token_length[i] != 0 else 0}, {total_true_array[i]}, {token_length[i]}")
        print(f"Msg Acc {i}: {total_msg_true_array[i] / token_length[i] if token_length[i] != 0 else 0}, {total_msg_true_array[i]}, {token_length[i]}")
        print(f"Diff {i}: {len(diff_token[i]) / min(len(tokens), diff_all[i]) if min(len(tokens), diff_all[i]) != 0 else 0}, {len(diff_token[i])}, {len(tokens)}, {diff_all[i]}")
        print(f"Right {i}: {diff_single[i] / token_length[i] if token_length[i] != 0 else 0}, {diff_single[i]}, {token_length[i]}")


if __name__ == '__main__':
    calc()
    split()
    show1()

