import numpy as numpy

def read_data(n_step):
    csv_file = numpy.loadtxt("voice_data.txt", dtype=numpy.str, delimiter=' ')
    raw_data = []
    data = []
    raw_label = []
    label = []

    for row in csv_file:
        vector_row = row.astype(numpy.float64)
        raw_data.append(vector_row[1:])
        label.append(vector_row[0])

    raw_data = numpy.array(raw_data)

    for i in range(raw_data.shape[0]):
        if i + 8 < raw_data.shape[0]:
            data.append(raw_data[i: i + n_step])
            label.append(raw_label[i: i + n_step])
            #print(numpy.array(data).shape)
            #print(numpy.array(label).shape)
    return data, label

