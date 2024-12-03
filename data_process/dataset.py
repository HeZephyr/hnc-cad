from utils import process_solid, process_profile, process_loop, process_model
from tqdm import tqdm
import multiprocessing
import json 
from pathlib import Path
from glob import glob
import itertools


class Loader():
    """ 处理数据集的类 """

    def __init__(self, datapath, bit, format):
        """
        初始化Loader对象，指定数据集路径、位深度和数据格式。
        
        参数：
        datapath (str): 数据集的路径。
        bit (int): 要处理的位深度。
        format (str): 数据的格式（如 solid、profile、loop、model）。
        """
        self.datapath = datapath
        self.bit = bit
        self.format = format

    def load_all_obj(self):
        """
        加载所有对象（模型、轮廓、循环或固体），并进行处理，将数据拆分为训练集、验证集和测试集。
        
        返回：
        tuple: 包含三个列表的元组：训练数据、测试数据和验证数据。
        """
        print(f"Processing {self.format} data...")
        
        # 从 JSON 文件加载训练、验证和测试集的分割信息
        with open('data_process/train_val_test_split.json') as f:
            data_split = json.load(f)
        
        project_folders = []  # 用于存储所有项目文件夹的路径

        # 遍历项目文件夹（假设有 100 个子文件夹，命名为 0000 到 0099）
        for i in range(0, 100):
            cur_dir = Path(self.datapath) / str(i).zfill(4)
            # 将项目文件夹中的所有子文件夹路径添加到 project_folders 列表
            project_folders += sorted(glob(str(cur_dir) + '/*/'))

        # 准备并行处理的数据
        iter_data = zip(
            project_folders,  # 输入数据（文件夹路径）
            itertools.repeat(self.bit),  # 为每次迭代重复位深度
        )

        # 将数据格式映射到相应的处理函数
        process_func = {
            "solid": process_solid,
            "profile": process_profile,
            "loop": process_loop,
            "model": process_model
        }

        samples = []  # 用于存储处理后的样本

        # 获取可用的 CPU 数量
        num_cpus = multiprocessing.cpu_count()
        # 使用多进程池并行处理数据
        load_iter = multiprocessing.Pool(num_cpus).imap(process_func[self.format], iter_data)

        # 并行处理每个数据样本，并将结果收集到 samples 列表中
        for data_sample in tqdm(load_iter, total=len(project_folders)):
            samples += data_sample
        
        # 初始化训练集、测试集和验证集的列表
        train_samples = []
        test_samples = []
        val_samples = []

        # 遍历所有样本，将它们分配到相应的数据集
        for data in tqdm(samples):
            if data['name'] in data_split['train']:
                train_samples.append(data)
            elif data['name'] in data_split['test']:
                test_samples.append(data)
            elif data['name'] in data_split['validation']:
                val_samples.append(data)
            else:
                train_samples.append(data)  # 如果没有匹配，则默认分配到训练集

        # 打印数据集的摘要信息
        print(f"Data Summary")
        print(f"\tTraining data: {len(train_samples)}")
        print(f"\tValidation data: {len(val_samples)}")
        print(f"\tTest data: {len(test_samples)}")

        # 返回训练集、验证集和测试集
        return train_samples, test_samples, val_samples
