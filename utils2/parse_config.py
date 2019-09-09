import numpy as np


def parse_model_config(path):
    """ Parses thr yolo-v3 layer configuration file and returns module definitions """
    file = open(path, 'r') # 只读模式  打开文件
    lines = file.read().split('\n')  # 以 回车键 作为分隔符 读取文件  保存在lines 列表中 ; file.read()返回一个str ; split将str分离并返回分离后的list
    lines = [x for x in lines if x and not x.startswith('#')]  # 去除空行 和 以#开头的行
    lines = [x.rstrip().lstrip() for x in lines] # .rstrip() 删除字符串末尾的指定的字符，默认空格  .lstrip() 开头...

    module_defs = []
    for line in lines:
        if line.startswith('['): # 如果 一行是以 [ 开头（通常，这代表一个layer的开始，除了第一个) ，则创建字典以便于储存layer信息
            module_defs.append({})
            module_defs[-1]['type'] = line[1: -1].rstrip() # 去掉 [] 和可能有的空格
            if module_defs[-1]['type'] == 'convolutional': # 如果是卷积层， 令'batch_normalize'=0
                module_defs[-1]['batch_normalize'] = 0
        else: # 描述参数的行
            key, value = line.split('=') # 以=分离 分别作为上面定义的字典的 key 和 value
            value = value.strip() # 删除str 开头和结尾 可能存在的空格
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs   # 返回一个list  这个list 储存的是描述网络结构的 字典 dict 第一个字典描述网络整体信息


def parse_data_config(path):
    """ Parses the data configuration file """
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options