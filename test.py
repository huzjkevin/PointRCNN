import matplotlib.pyplot as plt
import numpy as np

result_file = 'output/rcnn/default/eval/epoch_30/val/final_result/output_jrdb_full_dataset.txt'
with open(result_file, "r") as f:
    precisions = f.readlines()

overall_precision = precisions[0].split(sep=',')
ap = overall_precision[1]
precision = np.asarray(overall_precision[2:], dtype=np.float)
recall = np.linspace(0, 1, 41)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.set_title("overal precision recall curve. AP: {}".format(ap))
ax.set_xlabel("recall")
ax.set_ylabel("precision")

ax.plot(recall, precision)

plt.savefig("tmp_img/pr_curve_full_dataset_2.png")

plt.show()
