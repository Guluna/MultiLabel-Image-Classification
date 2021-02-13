# This script shows an example of how to test your predict function before submission
# 0. To use this, replace x_test (line 26) with a list of absolute image paths. And
# 1. Replace predict with your predict function
# or
# 2. Import your predict function from your predict script and remove the predict function define here
# Example: from predict_username (predict_ajafari) import predict
# %% -------------------------------------------------------------------------------------------------------------------
import numpy as np
import torch
import os
os.system("sudo pip install opencv-python")
import cv2



def predict(x):
    # On the exam, x will be a list of all the paths to the images of our held-out set
    images = []
    RESIZE_TO = 150
    for img_path in x:
        # Here you would write code to read img_path and preprocess it

        images.append(cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE),
                                (RESIZE_TO, RESIZE_TO)))  # cv2.imread(path, cv2.IMREAD_GRAYSCALE) for grayscale


    x = torch.FloatTensor(np.array(images)).cuda()
    x = x / 255.0

    # Here you would load your model (.pt) and use it on x to get y_pred, and then return y_pred
    model_1 = torch.jit.load("model_amna_gul.pt")
    # print(model_1.state_dict())
    y_pred = model_1(x.view(-1, 1, RESIZE_TO, RESIZE_TO))
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.detach()
    y_pred = torch.round(y_pred)
    # print(y_pred)
    return y_pred.cpu()



# # # # %% -------------------------------------------------------------------------------------------------------------------
# x_test = ['cells_1327.png', 'cells_12.png', 'cells_1000.png','cells_1003.png','cells_1004.png','cells_1005.png','cells_1006.png',
#           'cells_1007.png','cells_1008.png','cells_1009.png','cells_1010.png','cells_1011.png','cells_1013.png','cells_1015.png',
#           'cells_1016.png','cells_1017.png','cells_1019.png','cells_1021.png','cells_1022.png','cells_1023.png',
#           'cells_1024.png','cells_1025.png','cells_1026.png','cells_1027.png','cells_1029.png','cells_1031.png','cells_1032.png',
#           'cells_1034.png','cells_1035.png','cells_1036.png','cells_1037.png','cells_1039.png','cells_1041.png','cells_1044.png',
#           'cells_1045.png','cells_1046.png','cells_1048.png','cells_1049.png','cells_1050.png','cells_1051.png','cells_1054.png',
#           'cells_1055.png','cells_1056.png','cells_1057.png','cells_1058.png','cells_1060.png','cells_1062.png','cells_1063.png',
#           'cells_1064.png','cells_1067.png','cells_1068.png','cells_1070.png','cells_1071.png','cells_1072.png','cells_1073.png',
#           'cells_1074.png','cells_1076.png','cells_1077.png','cells_1078.png','cells_1079.png','cells_1080.png','cells_1081.png',
#           'cells_1082.png','cells_1083.png','cells_1084.png','cells_1087.png','cells_1088.png','cells_1089.png','cells_1091.png',
#           'cells_1092.png','cells_1093.png','cells_1096.png','cells_1098.png','cells_1099.png','cells_1100.png','cells_1102.png',
#           'cells_1103.png','cells_1104.png','cells_1105.png','cells_1107.png','cells_1108.png','cells_1109.png','cells_1110.png',
#           'cells_1111.png','cells_1112.png','cells_1113.png','cells_1116.png','cells_1117.png','cells_1118.png','cells_1120.png',
#           'cells_1122.png','cells_1125.png','cells_1126.png','cells_1127.png','cells_1129.png','cells_1130.png','cells_1137.png',
#           'cells_1138.png','cells_1142.png','cells_1143.png','cells_1146.png','cells_1148.png','cells_1150.png','cells_1151.png',
#           'cells_1152.png','cells_1153.png','cells_1154.png','cells_1155.png','cells_1156.png','cells_1157.png','cells_1158.png',
#           'cells_1159.png','cells_1162.png','cells_1164.png','cells_1165.png','cells_1166.png','cells_1167.png','cells_1169.png',
#           'cells_1171.png','cells_1172.png','cells_1173.png','cells_1174.png','cells_1175.png','cells_1179.png','cells_1180.png',
#           'cells_1181.png','cells_1182.png','cells_1183.png','cells_1186.png','cells_1187.png','cells_1188.png','cells_1189.png',
#           'cells_1190.png','cells_1192.png','cells_1193.png','cells_1195.png','cells_1196.png','cells_1197.png','cells_1200.png',
#           'cells_1201.png','cells_1202.png','cells_1203.png','cells_1204.png','cells_1206.png','cells_1210.png','cells_1212.png',
#           'cells_1214.png','cells_1215.png','cells_1216.png','cells_1218.png','cells_1219.png','cells_1220.png','cells_1221.png',
#           'cells_1222.png','cells_1223.png','cells_1225.png','cells_1226.png','cells_1227.png','cells_1228.png','cells_1230.png',
#           'cells_1231.png','cells_1232.png','cells_1236.png','cells_1237.png','cells_1238.png','cells_1239.png','cells_1240.png','cells_1241.png','cells_1242.png','cells_1245.png','cells_1246.png','cells_1247.png','cells_1248.png','cells_1250.png','cells_1251.png','cells_1252.png','cells_1253.png','cells_1254.png','cells_1255.png','cells_1256.png','cells_1260.png','cells_1262.png','cells_1264.png','cells_1265.png','cells_1266.png','cells_1267.png','cells_1268.png','cells_1270.png','cells_1271.png','cells_1272.png','cells_1273.png','cells_1274.png','cells_1275.png','cells_1277.png','cells_1278.png','cells_1279.png','cells_1281.png','cells_1282.png','cells_1283.png','cells_1284.png','cells_1285.png','cells_1286.png','cells_1288.png','cells_1291.png','cells_1292.png','cells_1293.png','cells_1294.png','cells_1295.png','cells_1297.png','cells_1298.png','cells_1299.png','cells_1301.png','cells_1302.png','cells_1303.png','cells_1305.png','cells_1307.png','cells_1308.png','cells_1309.png','cells_1310.png','cells_1312.png','cells_1313.png','cells_1314.png','cells_1316.png','cells_1317.png','cells_1318.png','cells_1320.png','cells_1321.png','cells_1323.png','cells_1324.png','cells_1326.png', 'cells_1327.png',
#           'cells_0.png','cells_1.png','cells_2.png','cells_6.png','cells_7.png','cells_8.png','cells_10.png','cells_12.png','cells_15.png','cells_16.png','cells_17.png','cells_18.png','cells_20.png','cells_22.png','cells_23.png','cells_24.png','cells_25.png','cells_26.png','cells_28.png','cells_29.png','cells_30.png','cells_32.png','cells_33.png','cells_35.png','cells_36.png','cells_37.png','cells_38.png','cells_39.png','cells_40.png','cells_41.png','cells_42.png','cells_45.png','cells_47.png','cells_50.png','cells_51.png','cells_54.png','cells_55.png','cells_56.png','cells_57.png','cells_58.png','cells_59.png','cells_60.png','cells_61.png','cells_63.png','cells_65.png','cells_66.png','cells_68.png','cells_69.png','cells_70.png','cells_71.png','cells_73.png','cells_74.png','cells_75.png','cells_76.png','cells_77.png','cells_78.png','cells_80.png','cells_81.png','cells_83.png','cells_84.png','cells_89.png','cells_91.png','cells_93.png','cells_94.png','cells_96.png','cells_97.png','cells_98.png','cells_99.png','cells_100.png','cells_104.png','cells_105.png','cells_108.png','cells_109.png','cells_111.png','cells_112.png','cells_113.png','cells_114.png','cells_116.png','cells_117.png','cells_118.png','cells_120.png','cells_124.png','cells_125.png','cells_126.png','cells_127.png','cells_128.png','cells_129.png','cells_131.png','cells_132.png','cells_134.png','cells_135.png','cells_136.png','cells_138.png','cells_139.png','cells_141.png','cells_142.png','cells_143.png','cells_145.png','cells_148.png','cells_149.png','cells_150.png','cells_151.png','cells_152.png','cells_153.png','cells_154.png','cells_155.png','cells_156.png','cells_157.png','cells_158.png','cells_162.png','cells_163.png','cells_165.png','cells_166.png','cells_167.png','cells_168.png','cells_169.png','cells_170.png','cells_172.png','cells_174.png','cells_175.png','cells_176.png','cells_178.png','cells_179.png','cells_180.png','cells_181.png','cells_183.png','cells_184.png','cells_185.png','cells_186.png','cells_188.png','cells_189.png','cells_191.png','cells_192.png','cells_193.png','cells_195.png','cells_197.png','cells_198.png','cells_199.png','cells_200.png','cells_201.png','cells_202.png','cells_203.png','cells_204.png','cells_205.png','cells_206.png','cells_207.png','cells_209.png','cells_210.png','cells_211.png','cells_212.png','cells_213.png','cells_214.png','cells_215.png','cells_218.png','cells_220.png','cells_221.png','cells_222.png','cells_223.png','cells_224.png','cells_227.png','cells_228.png','cells_229.png','cells_230.png','cells_231.png','cells_233.png','cells_235.png','cells_238.png','cells_240.png','cells_241.png','cells_242.png','cells_243.png','cells_244.png','cells_245.png']  # Dummy image path list placeholder
#
# y_test_pred = predict(x_test)


# # %% -------------------------------------------------------------------------------------------------------------------
# assert isinstance(y_test_pred, type(torch.Tensor([1])))  # Checks if your returned y_test_pred is a Torch Tensor
# assert y_test_pred.dtype == torch.float  # Checks if your tensor is of type float
# assert y_test_pred.device.type == "cpu"  # Checks if your tensor is on CPU
# assert y_test_pred.requires_grad is False  # Checks if your tensor is detached from the graph
# assert y_test_pred.shape == (len(x_test), 7)  # Checks if its shape is the right one
# # Checks whether the your predicted labels are one-hot-encoded
# assert set(list(np.unique(y_test_pred))) in [{0}, {1}, {0, 1}]
# print("All tests passed!")
#
# a = y_test_pred.cpu().numpy()
