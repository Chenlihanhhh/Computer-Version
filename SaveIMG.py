import matplotlib
import mne
import matplotlib.pyplot as plt
import numpy as np
import os

# 文件路径和文件名称列表
data_folder = r'D:\ECG\BCICIV_2b_gdf'

# 事件的对应关系
eventDescription = {'276': "eyesOpen", '277': "eyesClosed", '768': "startTrail", '769': "cueLeft", '770': "cueRight",
'781': "feedback", '783': "cueUnknown",
'1023': "rejected", '1077': 'horizonEyeMove', '1078': "verticalEyeMove", '1079': "eyeRotation",
'1081': "eyeBlinks", '32766': "startRun"}

# 循环处理每个文件
for filename in os.listdir(data_folder):
   if filename.endswith(".gdf"):
      # 读取原始数据
      raw_data_gdf = mne.io.read_raw_gdf(os.path.join(data_folder, filename), preload=True,
      eog=['EOG:ch01', 'EOG:ch02', 'EOG:ch03'])

# 定义数据通道和通道类型
ch_types = ['eeg', 'eeg', 'eeg']
ch_names = ['EEG:Cz', 'EEG:C3', 'EEG:C4']

# 创建数据的描述信息
info = mne.create_info(ch_names=ch_names, sfreq=raw_data_gdf.info['sfreq'], ch_types=ch_types)

# 创建数据结构体
data = np.squeeze(np.array(
[raw_data_gdf['EEG:Cz'][0], raw_data_gdf['EEG:C3'][0], raw_data_gdf['EEG:C4'][0]]))

print(raw_data_gdf[0])
# 创建RawArray类型的数据
raw_data = mne.io.RawArray(data, info)

# 创建文件夹保存图片
output_folder = os.path.join(data_folder, 'output_images')
os.makedirs(output_folder, exist_ok=True)

# # 绘制并保存原始数据图像
# raw_data.plot()
plt.savefig(os.path.join(output_folder, filename + '_raw.png'),format='png')
plt.close()

#获取事件
event, _ = mne.events_from_annotations(raw_data_gdf)
print(event)

event_id = {}
for i in _:
   event_id[eventDescription[i]] = _[i]

# 提取epoch
   epochs = mne.Epochs(raw_data, event, event_id, tmax=1.5, event_repeated='merge')

# 绘制并保存epoch图像
   epochs.plot()
# plt.savefig(os.path.join(output_folder, filename + '_epochs.png'))
# plt.close()


print("Processing and saving images complete.")

