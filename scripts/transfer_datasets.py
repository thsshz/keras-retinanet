import csv


def transfer_annotations(input_name, output_name, pre_dir):
    input_file = open(input_name, 'r')
    input_lines = input_file.readlines()
    output_lines = []
    for input_line in input_lines:
        input_line = input_line.rstrip()
        annotations = input_line.split(' ')
        image_name = pre_dir + annotations[0]
        for k in range(len(annotations)):
            if k == 0:
                continue
            t = (k - 1) % 5
            if t == 0:
                label = annotations[k]
            elif t == 1:
                x1 = annotations[k]
            elif t == 2:
                y1 = annotations[k]
            elif t == 3:
                x2 = str(int(x1) + int(annotations[k]))
            elif t == 4:
                y2 = str(int(y1) + int(annotations[k]))
                output_lines.append((image_name, x1, y1, x2, y2, label))
        if k == 0:
            output_lines.append((image_name, '', '', '', '', ''))
    input_file.close()
    with open(output_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for output_line in output_lines:
            csv_writer.writerow(output_line)


def transfer_classes(output_name):
    output_lines = [('1', 0), ('2', 1)]
    with open(output_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for output_line in output_lines:
            csv_writer.writerow(output_line)


def main():
    transfer_annotations("../pedestrian_detection_trainval/train_annotations.txt", "../pedestrian_detection_trainval/train_annotations.csv", "train/")
    transfer_classes("../pedestrian_detection_trainval/train_classes.csv")
    transfer_annotations("../pedestrian_detection_trainval/val.txt", "../pedestrian_detection_trainval/val.csv", "val/")
    transfer_classes("../pedestrian_detection_trainval/val_classes.csv")

if __name__ == '__main__':
    main()

