# coding:utf-8
import os
import utils.sem_constant_own as constant

relations = constant.LABEL_TO_ID

def write_results(predictions, result_file="./temp_result.txt"):
    start_no = 8001
    # id_to_relation = dict(zip(relations.values(), relations.keys()))
    with open(result_file, "w") as file:
        for idx, rel in enumerate(predictions):
            # rel = id_to_relation[id]
            file.write('%d\t%s\n' % (start_no + idx, rel))


def convert_labels(labels):
    labels = list(labels)
    labels = map(lambda x: x.argmax(), labels)
    return labels


def show_result(result):
    begin = result.rfind("MACRO-averaged result (excluding Other)")
    end = result.rfind("\n\n\n\n")
    return result[begin: end]


def get_marco_f1(predict_labels, real_labels, gpu_idx=0, show_res=True):
    write_results(predict_labels, result_file="./temp_result_{}.txt".format(gpu_idx))
    write_results(real_labels, result_file="./gold_result_{}.txt".format(gpu_idx))
    command = "perl utils/scorer.pl gold_result_{}.txt temp_result_{}.txt".format(gpu_idx, gpu_idx)
    result = os.popen(command).read()
    if show_res:
        print(show_result(result))
    begin_index = result.find("macro-averaged F1")
    result = result[begin_index + 19:-6]
    try:
        return float(result) / 100
    except:
        return 0


if __name__ == "__main__":
    pass