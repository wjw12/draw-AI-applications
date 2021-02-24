from torch.utils.data import Dataset, DataLoader
from scipy import signal
import numpy as np

def load_strokes(file_path):  
    strokes = []
    with open(file_path) as f:
        start = False
        stroke = []
        for line in f.readlines():
            if line and line.startswith("stroke"):
                start = True
                stroke = []
                continue
            if start:
                if len(line) == 1:
                    start = False # reach the end of stroke
                    strokes.append(stroke.copy())
                    stroke = []
                elif len(line) > 30:
                    stroke.append(line)
        return strokes
        
def convert_strokes_to_points(strokes):
    processed_strokes = []
    for stroke in strokes:
        points = []
        for point in stroke:
            points.append([float(i) for i in point.split()])
        #print("length = ", len(points))
        processed_strokes.append(points.copy())
    return processed_strokes
    
# output format
# ready to use in neural networks
# 0 delta x
# 1 delta y
# 2 pressure
# 3 rotation
# 4 tilt X
# 5 tilt Y
# 6 x0
# 7 y0

def normalize_strokes(processed_strokes, sequence_length):
    normalized = []
    for stroke in processed_strokes:
        if len(stroke) > sequence_length:
            # need to resample to the max length
            continue 
        points = []
        x0 = stroke[0][2]
        y0 = stroke[0][3]
        lastx = x0
        lasty = y0
        for p in stroke:
            points.append([p[2] - lastx, p[3] - lasty, p[4], p[5], p[6], p[7], x0, y0])
            lastx, lasty = p[2], p[3]
            
        for j in range(sequence_length - len(points)):
            points.append([0, 0, 0, 0, 0, 0, 0, 0])
        normalized.append(points.copy())
    return normalized
    
def rebuild_strokes(normalized_strokes, delta_t):
    result = []
    for stroke in normalized_strokes:
        x0, y0 = stroke[0][6], stroke[0][7]
        lastx, lasty = x0, y0
        index = 0
        time = 0.0
        points = []
        for point in stroke:
            new_point = [index, time, lastx + point[0], lasty + point[1], point[2], point[3], point[4], point[5]]
            points.append(new_point)
            time += delta_t
            index += 1
            lastx, lasty = new_point[2], new_point[3]
        result.append(points.copy())
    return result

class CynData(Dataset):
    def __init__(self, filename, sequence_length):
        strokes = load_strokes(filename)
        stroke_points = convert_strokes_to_points(strokes)
        self.training_data = np.transpose(np.array(normalize_strokes(stroke_points)), (0, 2, 1))[:, :6, :]
        
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        # dimension 0: point index in a stroke
        # dimension 1: pen stroke dim ( = 6)
        return self.training_data[idx]
        