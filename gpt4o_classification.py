#!/usr/bin/env python3
"""
使用 GPT-4o 对星系图片进行分类
Classify galaxy images using GPT-4o API

均匀选取 100 张图片进行测试
Uniformly sample 100 images for testing
"""

import os
import base64
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import random

# 需要安装: pip install openai
from openai import OpenAI


def encode_image_to_base64(image_path):
    """将图片编码为 base64 格式"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def classify_galaxy_with_gpt4o(client, image_path, max_retries=3):
    """
    使用 GPT-4o 对单张星系图片进行分类

    Args:
        client: OpenAI client
        image_path: 图片路径
        max_retries: 最大重试次数

    Returns:
        prediction: 预测的类别名称 (edge_on, featured, smooth, spiral)
    """

    # 编码图片
    base64_image = encode_image_to_base64(image_path)

    # 构建 prompt
    prompt = """You are an expert astronomer specializing in galaxy morphology classification.

Please classify this galaxy image into ONE of the following four categories:
1. "spiral" - Spiral galaxies with clear spiral arms
2. "smooth" - Smooth elliptical galaxies without features
3. "edge_on" - Edge-on disk galaxies (seen from the side)
4. "featured" - Galaxies with features but not clearly spiral (e.g., irregular, disturbed)

Important:
- Respond with ONLY the category name: spiral, smooth, edge_on, or featured
- Do not include any explanation or additional text
- Be as accurate as possible based on the visual features

Your answer (one word only):"""

    # 调用 GPT-4o API
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # 使用 GPT-4o 模型
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=50,
                temperature=0.0,  # 使用低温度以获得稳定结果
            )

            # 提取预测结果
            prediction = response.choices[0].message.content.strip().lower()

            # 验证预测结果是否在有效范围内
            valid_categories = ['spiral', 'smooth', 'edge_on', 'featured']
            if prediction in valid_categories:
                return prediction
            else:
                # 尝试从响应中提取有效类别
                for category in valid_categories:
                    if category in prediction:
                        return category
                print(f"Warning: Invalid prediction '{prediction}', retrying...")

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # 等待后重试
            else:
                return "unknown"  # 如果所有尝试都失败，返回 unknown

    return "unknown"


def select_balanced_samples(csv_path, dataset_dir, total_samples=100):
    """
    从每个类别均匀选取样本

    Args:
        csv_path: 标签 CSV 文件路径
        dataset_dir: 数据集目录
        total_samples: 总样本数

    Returns:
        selected_samples: 选中的样本列表
    """

    # 读取标签
    df = pd.read_csv(csv_path)

    # 按类别分组
    label2id = {"edge_on": 0, "featured": 1, "smooth": 2, "spiral": 3}
    samples_by_class = {label: [] for label in label2id.keys()}

    print("Loading samples...")
    for idx, row in df.iterrows():
        galaxy_id = str(row['ID'])
        label_name = row['label']

        # 找到图片文件
        img_path = Path(dataset_dir) / label_name / f"{galaxy_id}.jpg"

        if img_path.exists():
            samples_by_class[label_name].append({
                'id': galaxy_id,
                'path': str(img_path),
                'label': label_name,
            })

    # 计算每个类别应选取的数量
    num_classes = len(label2id)
    samples_per_class = total_samples // num_classes

    print(f"\nSampling strategy:")
    print(f"  Total samples: {total_samples}")
    print(f"  Classes: {num_classes}")
    print(f"  Samples per class: {samples_per_class}")

    # 从每个类别随机选取
    selected_samples = []
    random.seed(42)  # 设置随机种子以保证可重复性

    for label_name, samples in samples_by_class.items():
        available = len(samples)
        to_select = min(samples_per_class, available)

        selected = random.sample(samples, to_select)
        selected_samples.extend(selected)

        print(f"  {label_name:12s}: {to_select:3d} selected from {available:4d} available")

    # 如果总数不够 100，从剩余样本中补充
    if len(selected_samples) < total_samples:
        remaining = total_samples - len(selected_samples)
        print(f"\n  补充 {remaining} 个样本...")

        all_samples = [s for samples in samples_by_class.values() for s in samples]
        selected_ids = {s['id'] for s in selected_samples}
        available_samples = [s for s in all_samples if s['id'] not in selected_ids]

        if len(available_samples) >= remaining:
            additional = random.sample(available_samples, remaining)
            selected_samples.extend(additional)

    # 打乱顺序
    random.shuffle(selected_samples)

    print(f"\nTotal selected: {len(selected_samples)}")
    return selected_samples


def main():
    """主函数"""

    print("="*70)
    print("GPT-4o Galaxy Classification".center(70))
    print("="*70)

    # ========== 配置 ==========
    # !!! 在这里填入你的 OpenAI API Key !!!
    API_KEY = "your-api-key"  # <--- 填入你的 API Key
    BASE_URL = "https://api.openai.com/v1"  # OpenAI API base URL (default)

    # 输入输出路径
    csv_path = "/home/shiqiu/dinov3-tng50-finetune/gz2_top1000_high_confidence.csv"
    dataset_dir = "/home/shiqiu/dinov3-tng50-finetune/gz2_dataset_for_dinov3"
    output_csv = "/home/shiqiu/dinov3-tng50-finetune/gpt4o_classification_results.csv"

    num_samples = 100  # 测试样本数量

    print(f"\n配置 Configuration:")
    print(f"  输入 CSV:     {csv_path}")
    print(f"  数据集目录:    {dataset_dir}")
    print(f"  输出 CSV:     {output_csv}")
    print(f"  测试样本数:    {num_samples}")
    print(f"  API Key:      {API_KEY[:10]}..." if len(API_KEY) > 10 else "  API Key: (too short)")
    print(f"  Base URL:     {BASE_URL}")

    # ========== 初始化 OpenAI 客户端 ==========
    print(f"\n初始化 OpenAI 客户端...")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # ========== 选择样本 ==========
    print(f"\n{'='*70}")
    print("选择测试样本 Selecting Test Samples".center(70))
    print(f"{'='*70}")

    samples = select_balanced_samples(csv_path, dataset_dir, num_samples)

    # ========== 运行分类 ==========
    print(f"\n{'='*70}")
    print("运行 GPT-4o 分类 Running GPT-4o Classification".center(70))
    print(f"{'='*70}")

    results = []

    print(f"\n处理 {len(samples)} 张图片...")
    print("注意: 这将调用 OpenAI API，可能需要一些时间和费用\n")

    for i, sample in enumerate(tqdm(samples, desc="Classifying")):
        galaxy_id = sample['id']
        image_path = sample['path']
        true_label = sample['label']

        # 调用 GPT-4o
        try:
            prediction = classify_galaxy_with_gpt4o(client, image_path)
            is_correct = (prediction == true_label)

            results.append({
                'ID': galaxy_id,
                'answer': true_label,
                'gpt4o_answer': prediction,
                'TRUE/FALSE': 'TRUE' if is_correct else 'FALSE',
            })

            # 每 10 张图片保存一次（防止中断导致丢失）
            if (i + 1) % 10 == 0:
                df_temp = pd.DataFrame(results)
                df_temp.to_csv(output_csv + ".temp", index=False)

        except Exception as e:
            print(f"\nError processing {galaxy_id}: {e}")
            results.append({
                'ID': galaxy_id,
                'answer': true_label,
                'gpt4o_answer': 'error',
                'TRUE/FALSE': 'FALSE',
            })

        # 添加延迟以避免 API 速率限制
        time.sleep(0.5)

    # ========== 保存结果 ==========
    print(f"\n{'='*70}")
    print("保存结果 Saving Results".center(70))
    print(f"{'='*70}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)

    # 删除临时文件
    temp_file = Path(output_csv + ".temp")
    if temp_file.exists():
        temp_file.unlink()

    print(f"\nResults saved to: {output_csv}")

    # ========== 统计结果 ==========
    total = len(df_results)
    correct = (df_results['TRUE/FALSE'] == 'TRUE').sum()
    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*70}")
    print("统计结果 Statistics".center(70))
    print(f"{'='*70}")

    print(f"\n  总样本数 Total:      {total}")
    print(f"  正确数 Correct:      {correct}")
    print(f"  错误数 Incorrect:    {total - correct}")
    print(f"  准确率 Accuracy:     {accuracy:.4f} ({100*accuracy:.2f}%)")

    # 各类别统计
    print(f"\n  各类别准确率 Per-class Accuracy:")
    for label in ['edge_on', 'featured', 'smooth', 'spiral']:
        class_df = df_results[df_results['answer'] == label]
        if len(class_df) > 0:
            class_correct = (class_df['TRUE/FALSE'] == 'TRUE').sum()
            class_acc = class_correct / len(class_df)
            print(f"    {label:12s}: {class_correct:3d}/{len(class_df):3d} = {100*class_acc:5.1f}%")

    # 显示前 10 行
    print(f"\n  前 10 行结果 First 10 rows:")
    print("  " + "-"*66)
    print(df_results.head(10).to_string(index=False))

    print(f"\n{'='*70}")
    print("Complete!".center(70))
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
