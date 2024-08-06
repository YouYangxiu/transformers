# -*- coding: utf-8 -*-
"""
@Time: 2024/08/06/16/36
@Author: josephyou
@Email: josephyou@tencent.com
"""

# 遍历当前文件夹和所有的子文件夹，删除所有以Identifier结尾的文件
import os

def delete_identifier_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('Identifier'):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    current_directory = os.getcwd()
    delete_identifier_files(current_directory)