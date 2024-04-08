'''
step1:读取数据
step2:滤波
step3：去伪迹
step4：重参考
step5：分段
step6：叠加平均
step7：时频分析
step8：提取数据
'''

# MNE

# 导入原始数据
import numpy as np
import mne
import matplotlib.pyplot as plt  # new
from matplotlib import pyplot as plt
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet

# 数据地址，读取数据
data_path = r'D:\桌面\BCICIV_2b_gdf\B0101T.gdf'
raw = mne.io.read_raw_eeglab(data_path, preload=True)

# step1=====================================

# 查看原始数据信息
print(raw)
print(raw.info)

# 电极定位
locs_info_path = "E:\\Matlab\\toolbox\\eeglab14_0_0b\\sample_locs\\sample_data\\eeglab_chan32.locs"  # locs文件地址
montage = mne.channels.read_custom_montage(locs_info_path)  # 读取电极位置信息
new_chan_names = np.loadtxt(locs_info_path, dtype=str, usecols=3)  # 读取正确的导联名称
old_chan_names = raw.info["ch_names"]  # 读取旧的导联名称
chan_names_dict = {old_chan_names[i]: new_chan_names[i] for i in range(32)}  # 创建字典，匹配新旧导联名称
raw.rename_channels(chan_names_dict)  # 更新数据中的导联名称
raw.set_montage(montage)  # 传入数据的电极位置信息

# 设定导联类型为eeg和eog
chan_types_dict = {new_chan_names[i]: "eeg" for i in range(32)}
chan_types_dict = {"EOG1": "eog", "EOG2": "eog"}
raw.set_channel_types(chan_types_dict)

# 查看修改后的数据相关信息
print(raw.info)

# 可视化原是数据
# 绘制原始数据波形图
raw.plot(duration=5, n_channels=32, clipping=None)

# 绘制原始数据功率谱图
raw.plot_psd(average=True)

# 绘制电极拓扑图
raw.plot_sensors(ch_type='eeg', show_names=True)

# 绘制原始数据拓扑图
raw.plot_psd_topo()

# step2=====================================

# 陷波滤波---用陷波滤波器去掉工频（step1中的功率谱图显示60Hz处可能存在噪声，比较特殊因为大多在50Hz处。所以切记根据功率谱图判断）
raw = raw.notch_filter(freqs=(60))
raw.plot_psd(average=True)  # 绘制功率谱图

# 高低通滤波
raw = raw.filter(l_freq=0.1, h_freq=30)  # 默认method为FIR，括号内加method：(l_freq=0.1, h_freq=30,method='iir')可修改滤波方法为IIR
raw.plot_psd(average=True)

# step3=====================================

# 去伪迹

# 去坏段---手动标记maker？？？
# matplotlib.use('TkAgg') # 无限弹窗
# fig = raw.plot(duration=5, n_channels=32, clipping=None)
# fig.canvas.key_press_event('a')

# 去坏道
raw.info['bads'].append('FC5')  # 坏道标记,多个坏道则raw.info['bads'].extend(['FC5','C3'])
print(raw.info['bads'])  # 打印坏道

# 坏道插值重建
raw = raw.interpolate_bads()  # 对标记为bad的导联进行了信号重建

# 独立成分分析 ICA
# 运行ICA
ica = ICA(max_iter='auto')
raw_for_ica = raw.copy().filter(l_freq=1, h_freq=None)
ica.fit(raw_for_ica)

ica.plot_sources(raw_for_ica)  # 绘制各成分的时序信号图
ica.plot_components()  # 绘制各成分地形图

ica.plot_overlay(raw_for_ica, exclude=[1])  # 查看去掉某一成分前后信号差异
ica.plot_properties(raw, picks=[1, 16])  # 单独可视化每个成分

# 剔除成分
ica.exclude = [1]  # 设定要剔除的成分序号
ica.apply(raw)  # 应用到脑电数据上

# 绘制ICA后的数据波形图
raw.plot(duration=5, n_channels=32, clipping=None)

# step4======================================
# 重参考

# raw.set_eeg_reference(ref_channels=['TP9','TP10']) # 以TP9、TP10为参考电极的话
# 报错：Missing channels from ch_names required by include:['TP9', 'TP10']，改成第一步执行信息有的“T3T4”也不行

raw.set_eeg_reference(ref_channels='average')  # 使用平均参考,不报错
'''
raw.set_eeg_reference(ref_channels='REST') # 使用REST参考
raw_bip_ref = mne.set_bipolar_reference(raw, anode=['EEG X'], cathode=['EEG Y']) # 使用双极参考，EEG X和EEG Y对应于参考的阳极和阴极导联
'''

# step5======================================
# 分段

print(raw.annotations)  # 提取事件信息
print(raw.annotations.duration)  # 基于annotations打印数据的事件持续时长
print(raw.annotations.description)  # 基于annotations打印数据的事件描述信息
print(raw.annotations.onset)  # 基于annotations打印数据的事件开始时间

# 事件信息数据类型转换
events, event_id = mne.events_from_annotations(raw)  # 将annotation类型的事件转换为events类型
print(events.shape, event_id)  # 打印event矩阵的shape和event_id（不同markers对应整型字典信息），输出(154,3),共154个markers
# 报错 ModuleNotFoundError: No module named 'events'

# 数据分段
epochs = mne.Epochs(raw, events, event_id=2, tmin=-1, tmax=2, baseline=(-0.5, 0),
                    preload=True, reject=dict(eeg=2e-4))  # 清空plot区后再执行没有错
print(epochs)

# 分段数据可视化
epochs.plot(n_epochs=4)

epochs.plot_psd(picks='eeg')  # 绘制功率谱图（逐导联）

bands = [(4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta')]
epochs.plot_psd_topomap(bands=bands, vlim='joint')  # 绘制功率拓扑图（分三频段）

# step6========================================
# 叠加平均

evoked = epochs.average()  # 数据叠加平均

evoked.plot()  # 绘制逐导联的时序信号图

times = np.linspace(0, 2, 5)  # 绘制0, 0.5, 1, 1.5, 2s处的地形图
evoked.plot_topomap(times=times, colorbar=True)

evoked.plot_topomap(times=0.8, average=0.1)  # 绘制特定时刻（0.8s处）,取0.75-0.85s的均值

evoked.plot_joint()  # 绘制联合图

evoked.plot_image()  # 绘制逐导联热力图

evoked.plot_topo()  # 绘制拓扑时序信号图

mne.viz.plot_compare_evokeds(evokeds=evoked, combine='mean')  # 绘制平均所有电极后的ERP

mne.viz.plot_compare_evokeds(evokeds=evoked, picks=['O1', 'Oz', 'O2'], combine='mean')  # 绘制枕叶电极的平均ERP

# step7========================================

# 时频分析
freqs = np.logspace(*np.log10([4, 30]), num=10)  # 频段选取4-30Hz
n_cycles = freqs / 2.
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True)

# 时频结果绘制
power.plot(picks=['O1', 'Oz', 'O2'], baseline=(-0.5, 0), mode='logratio', title='auto')  # 枕叶导联的power结果
power.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average power')  # 绘制power拓扑图

# 绘制不同频段的power拓扑图，以Theta,Alpha为例
power.plot_topomap(tmin=0, tmax=0.5, fmin=4, fmax=8, baseline=(-0.5, 0), mode='logratio', title='Theta')
power.plot_topomap(tmin=0, tmax=0.5, fmin=8, fmax=12, baseline=(-0.5, 0), mode='logratio', title='Alpha')

# 绘制联合图
power.plot_joint(baseline=(-0.5, 0), mode='mean', tmin=-0.5, tmax=1.5, timefreqs=[(0.5, 10), (1, 8)])

# ITC结果绘制类似，以拓扑图为例，先不写


# step8=========================================
# 提取数据

# get_data()的使用
epochs_array = epochs.get_data()  # 以epoch为例
print(epochs_array.shape)  # 查看获取的数据
print(epochs_array)

# .data的使用
power_array = power.data
print(power_array.shape)
print(power_array)










