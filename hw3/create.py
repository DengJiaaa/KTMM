import numpy as np
import pandas as pd

RE = 1.496e11

def generate_and_save_particle_data(filename, nn):
    m = np.random.rand(nn) * 1000 * 5.965e24 * 6.67e-11
    x = np.random.rand(nn) * 10 * RE
    y = np.random.rand(nn) * 10 * RE
    v_x = np.random.rand(nn) * 100 * 1000
    v_y = np.random.rand(nn) * 100 * 1000
    data = {
        'm': m,
        'x': x,
        'y': y,
        'v_x': v_x,
        'v_y': v_y
    }
    df = pd.DataFrame(data)

    # 将 DataFrame 写入 CSV 文件
    df.to_csv(filename, index=False)

n = 9

filename = f"data_{n}.csv"
# 使用示例
generate_and_save_particle_data(filename, n)
