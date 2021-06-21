import utils

filePath = '/home/visionlab/Wenjing/myCodes/tcn_Predict/data/readings.csv'
T_train = 24 * 7 * 2 
T_predict = 24 * 7
savePath = '/home/visionlab/Wenjing/data/' + str(T_train)+'_'+str(T_predict)+'_Z/'
#step 1
utils.generate_Seq_npy_data_Z(filePath, savePath, T_train, T_predict)
#step 2
utils.generate_Seq_csv_data(savePath)
