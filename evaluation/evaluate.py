#!/usr/bin/env python
import sys, os, os.path
from ChalearnLAPEvaluation import evalGesture

#input_dir = sys.argv[1]
#output_dir = sys.argv[2]
input_dir = './'
output_dir = './result'

submit_dir = os.path.join(input_dir, 'prediction')
truth_dir = os.path.join(input_dir, 'reference')

print(submit_dir)
print(truth_dir)

if not os.path.isdir(submit_dir):
    print ("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')

    # Call evaluation for this track
    score=evalGesture(submit_dir,truth_dir)
    print("Score: %f" % score)

    # Store the score
    output_file.write("Overlap: %f" % score)

    output_file.close()
