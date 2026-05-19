"""
Grasp Filtering for YuanChair Part 1 (Seat)
===========================================
加载 Part 1 (Seat) 的抓取规划数据，过滤出仅保留从上往下抓（基座坐标系 Z 轴负方向）的抓取，
并按照要求保存并覆盖为 demo_yuanchair-part2_grasps.pickle 文件。
"""

import os
import pickle
import numpy as np
from wrs.grasping.grasp import GraspCollection


def filter_by_orientation(grasp_collection, axis_idx=2, direction="down", threshold=0.0):
    """
    过滤抓取姿态，仅保留夹爪控制轴指向特定方向的抓取。

    axis_idx=2 代表夹爪的接近轴（Z轴）。
    ac_rotmat[2, axis_idx] 获取该轴在世界/基座坐标系 Z 轴上的分量。
    当 direction="down" 且分量 < 0 时，说明夹爪是从上往下接近物体的。
    """
    filtered = GraspCollection()
    for grasp in grasp_collection:
        z_component = grasp.ac_rotmat[2, axis_idx]
        if direction == "down" and z_component < -threshold:
            filtered.append(grasp)
    return filtered


def main():
    # 1. 确定输入输出路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(current_dir, "_output")

    # 输入：yuanchair-part1.stl 对应的原抓取文件
    input_path = os.path.join(out_dir, "demo_yuanchair-part1_grasps.pickle")
    # 输出：用户指定保存并覆盖的目标文件 (part2)
    output_path = os.path.join(out_dir, "demo_yuanchair-part2_grasps.pickle")

    # 2. 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"\n[ERROR] 找不到 Part 1 的抓取规划文件: {input_path}")
        print("请确保已经运行过 planning.py 生成了基础抓取数据。")
        return

    # 3. 加载抓取数据
    print(f"\n[INFO] 正在读取原椅座(Part 1)抓取数据: {os.path.relpath(input_path)}")
    with open(input_path, 'rb') as f:
        grasp_collection = pickle.load(f)
    print(f"成功加载，原始抓取总数: {len(grasp_collection)}")

    # 4. 执行定向过滤（从上往下抓）
    print("\n[INFO] 正在执行方向过滤（只保留基座坐标系 Z 轴负方向 / 从上往下抓）...")
    filtered_collection = filter_by_orientation(
        grasp_collection,
        axis_idx=2,       # 检查夹爪的接近轴(Z轴)
        direction="down", # 方向朝下
        threshold=0.0     # 严格小于0
    )
    print(f"过滤完成，符合要求的朝下抓取总数: {len(filtered_collection)}")

    if len(filtered_collection) == 0:
        print("[WARN] 过滤后的抓取数量为 0，请检查原始数据或放宽阈值。")
        return

    # 5. 保存并覆盖目标文件
    print(f"\n[INFO] 正在写入并覆盖目标文件: {os.path.relpath(output_path)}")
    filtered_collection.save_to_disk(file_name=output_path)
    print("=" * 50)
    print("[OK] 任务成功完成！")
    print(f"已成功将 Part 1 过滤后的朝下抓取覆盖至 Part 2 缓存中。")
    print("=" * 50)


if __name__ == "__main__":
    main()