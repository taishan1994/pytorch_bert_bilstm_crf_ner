from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np



def calculate_metric(gt, predict):
    """
    计算 tp fp fn
    """
    tp, fp, fn = 0, 0, 0
    for entity_predict in predict:
        flag = 0
        for entity_gt in gt:
            if entity_predict[0] == entity_gt[0] and entity_predict[1] == entity_gt[1]:
                flag = 1
                tp += 1
                break
        if flag == 0:
            fp += 1

    fn = len(gt) - tp

    return np.array([tp, fp, fn])


def get_p_r_f(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.array([p, r, f1])

def classification_report(metrics_matrix, label_list, id2label, total_count, digits=2, suffix=False):
    name_width = max([len(label) for label in label_list])
    last_line_heading = 'micro-f1'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for label_id, label_matrix in enumerate(metrics_matrix):
        type_name = id2label[label_id]
        p,r,f1 = get_p_r_f(label_matrix[0],label_matrix[1],label_matrix[2])
        nb_true = total_count[label_id]
        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'
    mirco_metrics = np.sum(metrics_matrix, axis=0)
    mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
    # compute averages
    print('precision:{:.4f} recall:{:.4f} micro_f1:{:.4f}'.format(mirco_metrics[0],mirco_metrics[1],mirco_metrics[2]))
    report += row_fmt.format(last_line_heading,
                             mirco_metrics[0],
                             mirco_metrics[1],
                             mirco_metrics[2],
                             np.sum(s),
                             width=width, digits=digits)

    return report