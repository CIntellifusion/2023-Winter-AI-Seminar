class DatasetConfig:
    n_range = [1,5]     # 阶梯数范围
    s0_range = [2,5]    # 起步里程范围 
    c0_range = [5,15]   # 起步价范围
    ci_range = [2,5]    # 单位里程价格范围
    steplen_range = [20,100]    # 阶梯区间长度范围
    ep_mu = 10
    ep_sigma = 25   # 误差项标准差
    x_ubound = 200 # 里程上界
