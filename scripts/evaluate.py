#!/usr/bin/env python

import re
import rospy

if __name__ == '__main__':
    rospy.init_node('create_kb')
    test_path = rospy.get_param('~test', './test.txt')
    gt_path = rospy.get_param('~ground_truth', './ground_truth.txt')

    print('Comparing {} and {}.'.format(test_path, gt_path))

    ground_truth = dict()
    regex = re.compile(r"~[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}:[0-9]{2}:[0-9]{2}", re.IGNORECASE)

    with open(gt_path, 'r') as f:
        for l in f:
            clean_line = l.strip()

            if len(clean_line) > 0:
                clean_line = regex.sub('', clean_line)
                ground_truth.update({clean_line: False})

    test_counter = 0
    false_positives = 0
    with open(test_path, 'r') as f:
        for l in f:
            clean_line = l.strip()

            if len(clean_line) > 0:
                test_counter += 1
                clean_line = regex.sub('', clean_line)

                if clean_line in ground_truth.keys():
                    ground_truth[clean_line] = True
                else:
                    false_positives += 1

    positives = 0
    negatives = 0

    for val in ground_truth.values():
        if val:
            positives += 1
        else:
            negatives += 1

    total_gt = positives + negatives

    print("Evaluation result:")
    print("\ttotal ground truth data: {} predicates".format(total_gt))
    print("\ttotal test data: {} predicates".format(test_counter))
    print("\tmatched ground truth data (true positives): {0:.2f}%".format(float(positives) / float(total_gt)))
    print("\tmissed ground truth data (false negatives): {0:.2f}%".format(float(negatives) / float(total_gt)))
    print("\texceeding (wrong) data over total test (false positives): {0:.2f}%".format(
        float(false_positives) / float(test_counter)))
