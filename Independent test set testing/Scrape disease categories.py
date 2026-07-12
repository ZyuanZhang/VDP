# -*- coding: utf-8 -*-
"""
功能：
读取包含 DiseaseMeshID 和 MajorDiseaseCategory 两列的表格；
根据 DiseaseMeshID 访问 NCBI MeSH 页面；
提取 MeSH 层级路径中 Diseases Category 后面一层的所有类别；
用 "/" 拼接后写入 MajorDiseaseCategory 列；
最后保存为新表格。

安装依赖：
pip install pandas requests beautifulsoup4 lxml openpyxl tqdm
"""

import re
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from pathlib import Path


# ===================== 1. 参数设置 =====================

INPUT_FILE = "./Associations_test.xlsx"          # 改成你的输入文件名，例如 disease.xlsx 或 disease.csv
OUTPUT_FILE = "Associations_test.xlsx" # 输出文件名

ID_COL = "DiseaseMeshID"
RESULT_COL = "MajorDiseaseCategory"

BASE_URL = "https://www.ncbi.nlm.nih.gov/mesh/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; mesh-category-scraper/1.0; "
        "mailto:your_email@example.com)"
    )
}

SLEEP_MIN = 0.4
SLEEP_MAX = 1.0

MAX_RETRIES = 3
TIMEOUT = 20


# ===================== 2. 读取和保存表格 =====================

def read_table(file_path: str) -> pd.DataFrame:
    """
    自动读取 xlsx/xls/csv 文件。
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path, dtype=str)
    elif suffix == ".csv":
        return pd.read_csv(file_path, dtype=str, encoding="utf-8-sig")
    else:
        raise ValueError("仅支持 .xlsx、.xls、.csv 文件")


def save_table(df: pd.DataFrame, output_file: str):
    """
    根据后缀自动保存。
    """
    output_file = Path(output_file)
    suffix = output_file.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        df.to_excel(output_file, index=False)
    elif suffix == ".csv":
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
    else:
        raise ValueError("输出文件仅支持 .xlsx、.xls、.csv")


# ===================== 3. MeSH ID 清洗 =====================

def clean_mesh_id(value):
    """
    清洗 DiseaseMeshID。

    支持以下情况：
    D004819
    mesh:D004819
    https://www.ncbi.nlm.nih.gov/mesh/?term=D004819

    如果一个单元格里有多个 ID，本代码默认取第一个 D 编号。
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    match = re.search(r"D\d{6}", text, flags=re.IGNORECASE)
    if match:
        return match.group(0).upper()

    return ""


# ===================== 4. 请求 NCBI MeSH 页面 =====================

def fetch_mesh_html(mesh_id: str) -> str:
    """
    请求 NCBI MeSH 页面，带简单重试。
    """
    params = {"term": mesh_id}

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                BASE_URL,
                params=params,
                headers=HEADERS,
                timeout=TIMEOUT
            )
            response.raise_for_status()
            return response.text

        except Exception as e:
            last_error = e
            wait_time = attempt * 2
            print(f"[警告] {mesh_id} 第 {attempt} 次请求失败：{e}，等待 {wait_time}s 后重试")
            time.sleep(wait_time)

    raise RuntimeError(f"{mesh_id} 请求失败：{last_error}")


# ===================== 5. 提取 MeSH 层级路径 =====================

def extract_hierarchy_paths(soup: BeautifulSoup):
    """
    提取页面中的所有 MeSH 层级路径。

    页面中类似：
    All MeSH Categories > Diseases Category > Infections > Virus Diseases ...
    All MeSH Categories > Diseases Category > Skin and Connective Tissue Diseases > Skin Diseases ...

    返回：
    [
        ["All MeSH Categories", "Diseases Category", "Infections", ...],
        ["All MeSH Categories", "Diseases Category", "Skin and Connective Tissue Diseases", ...]
    ]
    """

    # NCBI 页面中层级路径主要以链接形式出现，直接取所有文本更稳妥
    tokens = [x.strip() for x in soup.stripped_strings if x.strip()]

    paths = []
    current_path = None

    stop_words = {
        "Supplemental Content",
        "Follow NCBI",
        "National Library of Medicine"
    }

    for token in tokens:
        # 新层级路径开始
        if token == "All MeSH Categories":
            if current_path and len(current_path) > 1:
                paths.append(current_path)
            current_path = [token]
            continue

        if current_path is not None:
            # 到这些内容说明层级路径区域结束
            if token in stop_words or token.startswith("Supplemental Content"):
                if current_path and len(current_path) > 1:
                    paths.append(current_path)
                current_path = None
                continue

            current_path.append(token)

    if current_path and len(current_path) > 1:
        paths.append(current_path)

    # 去重，保持原顺序
    unique_paths = []
    seen = set()

    for path in paths:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)

    return unique_paths


# ===================== 6. 提取 Diseases Category 后一层 =====================

def extract_major_disease_categories(paths):
    """
    提取所有路径中 Diseases Category 后面一层的类别。

    例如：
    All MeSH Categories > Diseases Category > Infections > Virus Diseases
    提取：
    Infections

    All MeSH Categories > Diseases Category > Skin and Connective Tissue Diseases > Skin Diseases
    提取：
    Skin and Connective Tissue Diseases

    最终返回：
    Infections/Skin and Connective Tissue Diseases
    """

    categories = []

    for path in paths:
        if "Diseases Category" in path:
            idx = path.index("Diseases Category")

            if idx + 1 < len(path):
                category = path[idx + 1].strip()

                # 防止异常空值
                if category:
                    categories.append(category)

    # 去重，保持顺序
    unique_categories = []
    seen = set()

    for c in categories:
        if c not in seen:
            seen.add(c)
            unique_categories.append(c)

    return "/".join(unique_categories)


# ===================== 7. 单个 MeSH ID 处理 =====================

def get_major_disease_category(mesh_id: str) -> str:
    """
    输入一个 MeSH ID，返回 Diseases Category 后一层类别，用 / 拼接。
    """
    mesh_id = clean_mesh_id(mesh_id)

    if not mesh_id:
        return ""

    html = fetch_mesh_html(mesh_id)
    soup = BeautifulSoup(html, "lxml")

    paths = extract_hierarchy_paths(soup)
    major_categories = extract_major_disease_categories(paths)

    return major_categories


# ===================== 8. 批量处理主函数 =====================

def fill_major_disease_category(input_file: str, output_file: str):
    """
    批量读取表格并填充 MajorDiseaseCategory。
    """

    df = read_table(input_file)

    if ID_COL not in df.columns:
        raise ValueError(f"输入表格中找不到列：{ID_COL}")

    if RESULT_COL not in df.columns:
        df[RESULT_COL] = ""

    # 可选：保留原始 MajorDiseaseCategory
    backup_col = RESULT_COL + "_original"
    if backup_col not in df.columns:
        df[backup_col] = df[RESULT_COL]

    # 清洗 ID
    df[ID_COL] = df[ID_COL].apply(clean_mesh_id)

    # 缓存，避免重复 MeSH ID 反复访问
    cache = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing MeSH IDs"):
        mesh_id = row[ID_COL]

        if not mesh_id:
            df.at[idx, RESULT_COL] = ""
            continue

        if mesh_id in cache:
            df.at[idx, RESULT_COL] = cache[mesh_id]
            continue

        try:
            major_category = get_major_disease_category(mesh_id)
            cache[mesh_id] = major_category
            df.at[idx, RESULT_COL] = major_category

        except Exception as e:
            print(f"[错误] {mesh_id} 处理失败：{e}")
            cache[mesh_id] = ""
            df.at[idx, RESULT_COL] = ""

        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    save_table(df, output_file)

    print(f"\n处理完成，结果已保存到：{output_file}")
    return df


# ===================== 9. 运行 =====================

if __name__ == "__main__":
    result_df = fill_major_disease_category(INPUT_FILE, OUTPUT_FILE)
    print(result_df[[ID_COL, RESULT_COL]].head())