import torch
import psutil  # 用于获取CPU和内存使用情况
import GPUtil  # 用于获取GPU使用情况


'''
GPU和CPU分配算法：
1. 识别任务类型：首先，根据任务的类型确定是应该由CPU还是GPU来处理。例如，计算机视觉任务通常需要大量的图像处理和计算，因此最好将其分配给GPU。

2. 确定可用资源：确定当前系统中可用的CPU和GPU资源数量以及它们的负载情况，以便更好地分配任务。

3. 分配任务：根据任务类型和可用资源，将任务分配给CPU或GPU。如果任务可以在CPU和GPU之间并行处理，则可以使用并行处理技术同时利用两者的优势。

4. 监控任务执行：始终监控任务的执行情况，并确保任务在预期时间内完成。如果任务执行时间超过了预期，则可以尝试重新分配资源或对任务进行优化。

5. 动态调整资源分配：随着时间推移和任务类型的变化，可能需要动态调整资源分配策略。例如，当CPU和GPU负载不平衡时，可以尝试重新分配任务以平衡负载。
'''
def allocation():
    print("=====GPUandCPU_allocation=====")
    # 获取系统CPU和内存使用情况
    cpu_percent = psutil.cpu_percent()
    mem_percent = psutil.virtual_memory().percent
    print("cpu_percent: ", cpu_percent)
    print("mem_percent: ", mem_percent)
    # 获取所有可用GPU使用情况
    gpu_list = GPUtil.getGPUs()
    for gpu in gpu_list:
        gpu_percent = gpu.load * 100
        print(f"GPU {gpu.id}: {gpu_percent}%")
    # 根据任务类型和资源负载等因素，分配任务给CPU或GPU。
    my_task = 'semantic_reasoning'
    gpu_task_type = {'computer_vision', 'deep_learning', 'neural_networks', 'speech_recognition', 'data_analysis',
                     'data_visualization', 'semantic_reasoning'}
    try:
        if my_task in gpu_task_type:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检测是否存在GPU
            if device == 'GPU' and len(gpu_list) > 0 and gpu_percent < 80:
                print("属于GPU任务")
                print(device)
                print("=====end=====")
                # 如果有可用GPU且GPU负载低于80%，则将任务分配给GPU
                # TODO：编写将任务分配给GPU的代码
                return device;
            else:
                # 如果没有可用GPU或GPU负载过高，则将任务分配给CPU
                # TODO：编写将任务分配给CPU的代码
                print("属于CPU任务")
                print(device)
                print("=====end=====")
                device = 'CPU'
                return device;

    except KeyError:
        print("属于CPU任务")
        print(device)
        print("=====end=====")
        # 如果任务类型不属于GPU任务，则将任务分配给CPU
        # TODO：编写将任务分配给CPU的代码
        device = 'CPU'
        return device
















