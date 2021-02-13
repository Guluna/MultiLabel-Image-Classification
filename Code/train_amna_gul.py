# print("exam2")
import cv2
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

# use GPU if available else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # %% --------------------------------------- Load Image Data to numpy arrays --------------------------------------------------------------------

# if "train" not in os.listdir():
#     os.system("wget https://storage.googleapis.com/exam-deep-learning/train-Exam2.zip")
#     os.system("unzip train-Exam2.zip")

DATA_DIR = os.getcwd() + "/train/"
RESIZE_TO = 150
SEED = 42

# x, y = [], []
# for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".png"]:
#     # print(path)
#     x.append(cv2.resize(cv2.imread(DATA_DIR + path, cv2.IMREAD_GRAYSCALE), (RESIZE_TO, RESIZE_TO)))  # for grayscale & resizing
#     with open(DATA_DIR + path[:-4] + ".txt", "r") as s: # reading labels
#         label = s.read().splitlines()
#     y.append(label)
# x, y = np.array(x), np.array(y)     # (# of images, 50, 50, 3) (# of images,)
#
#
# # # %% --------------------------------------- One Hot Encoding of labels --------------------------------------------------------------------
#
# # Create MultiLabelBinarizer object
# one_hot = MultiLabelBinarizer()
# # One-hot encode data
# y_one_hot = one_hot.fit_transform(y)
#
# y_rearranged = y_one_hot.copy()
# # rearranging y_one_hot columns according to Exam2 description format (instead of default alphabetical order)
# permutation = [1, 2, 6, 0, 4, 5, 3]
# idx = np.empty_like(permutation)
# idx[permutation] = np.arange(len(permutation))
# y_rearranged[:] = y_rearranged[:, idx]  # in-place modification of y_rearranged, shape = (# of images, # of cell_types i.e. 7)
#
#
# # # %% --------------------------------------- Saving & loading .npy to/from local disk --------------------------------------------------------------------
#
# # saving loaded data to .npy so that I dont have to run above code everytime I run this script
# np.save("x_train.npy", x); np.save("y_train.npy", y_rearranged)
# # np.save("x_test.npy", x_test); np.save("y_test.npy", y_test)

x, y = np.load("x_train.npy"), np.load("y_train.npy")   # (929, 150, 150)
# print(x.shape)  # (929, 50, 50)
# print(y.shape)  # (929, 7)
# # displaying images
# plt.imshow(x[0])
# plt.show()

# # %% --------------------------------------- Converting numpy to Tensor object, Normalize X --------------------------------------------------------------------

X = torch.FloatTensor(x).to(device)   # (929, 150, 150)
X = X/255.0
Y = torch.FloatTensor(y).to(device)



# # %% --------------------------------------- Model Architecture, Loss, Optimizer --------------------------------------------------------------------

class MultiLabelModel(nn.Module):
    def __init__(self):
        super(MultiLabelModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(25920, 1024)
        self.fc2 = nn.Linear(1024, 7)
        # self.fc3 = nn.Logit()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # x = self.fc3(x)
        return x

model = MultiLabelModel().to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()

# # %% --------------------------------------- Training --------------------------------------------------------------------

min_loss = 10000
for epoch in range(100):
    print("Epoch", epoch)
    model.train()
    for batch in range(len(X)):
        inds = slice(batch*1, (batch+1)*1)
        optimizer.zero_grad()
        output = model((X[inds]).view(-1, 1, RESIZE_TO, RESIZE_TO))
        # logits = model(X[inds])
        # print(logits)
        loss = criterion(output, Y[inds])
        logits = torch.sigmoid(output)

        loss.backward()
        optimizer.step()
        logits = torch.round(logits)
        # print(logits)

    if loss < min_loss:
        traced_model = torch.jit.script(model)
        torch.jit.save(traced_model, "model_amna_gul.pt")
        min_loss = loss


    print(loss)



model.eval()
# print(model)



# loaded_model = torch.jit.load("traced.pt")


