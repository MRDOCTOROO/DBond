"""
使用Pyteomics库解析MGF文件
================================

本模块演示如何使用pyteomics库读取和解析MGF（Mascot Generic Format）文件
用于质谱数据分析。

安装说明：
-------------
使用pip安装pyteomics包：
    pip install pyteomics

示例MGF文件结构：
---------------------------
BEGIN IONS
TITLE=id=PXD003709;F001262.dat-pride.xml;spectrum=3670
PEPMASS=475.24009
CHARGE=2+
TAXONOMY=9606
SEQ=KGNYAER|KGNYAER|KGNYAER|KGNYAER|KGNYAER|KGNYAER|KGNYAER
USER03=0-MOD:01894,1-MOD:01894|0-MOD:01894,1-MOD:01894|0-MOD:01894,1-MOD:01894|0-MOD:01894,1-MOD:01894|0-MOD:01894,1-MOD:01894|0-MOD:01894,1-MOD:01894|0-MOD:01894,1-MOD:01894
140.1069	4619000.0
241.1545	1140000.0
335.1347	142900.0
375.1968	70430.0
537.2758	67600.0
652.3079	71560.0
709.3286	1936000.0
END IONS
"""

from pyteomics import mgf
import sys


def parse_mgf_file(file_path):
    """
    解析MGF文件并提取谱图信息。
    
    参数:
        file_path (str): 要解析的MGF文件路径
        
    返回:
        list: 谱图数据字典列表
    """
    try:
        spectra = []
        for spectrum in mgf.read(file_path):
            spectra.append(spectrum)
            
            # 提取参数
            params = spectrum.get('params', {})
            
            # 打印基本谱图信息
            print("=" * 50)
            print("谱图数据:")
            print(f"完整谱图: {spectrum}")
            
            # 提取并显示关键参数
            if params:
                # 获取所有可用键
                param_keys = list(params.keys())
                print(f"\n可用参数: {param_keys}")
                
                # 提取感兴趣的特定参数
                title = params.get('title', '无标题')
                seq = params.get('seq', '无序列')
                pepmass = params.get('pepmass', '无肽段质量')
                charge = params.get('charge', '无电荷')
                taxonomy = params.get('taxonomy', '无分类')
                
                print(f"\n关键参数:")
                print(f"标题: {title}")
                print(f"序列: {seq}")
                print(f"肽段质量: {pepmass}")
                print(f"电荷: {charge}")
                print(f"分类: {taxonomy}")
                
                # 获取第4个键（索引3），如原始示例中所示
                if len(param_keys) > 3:
                    key_4th = param_keys[3]
                    print(f"第4个参数键: {key_4th}")
                    print(f"第4个参数值: {params[key_4th]}")
            
            # 提取质荷比和强度数组
            mz_array = spectrum.get('m/z array', [])
            intensity_array = spectrum.get('intensity array', [])
            
            print(f"\n质荷比（m/z）值: {mz_array}")
            print(f"强度值: {intensity_array}")
            print("=" * 50)
            
        return spectra
        
    except FileNotFoundError:
        print(f"错误：未找到文件 '{file_path}'。")
        return []
    except Exception as e:
        print(f"解析MGF文件时出错: {e}")
        return []


def main():
    """
    主函数，演示MGF文件解析。
    """
    print("使用Pyteomics库解析MGF文件")
    print("=" * 40)
    
    # 检查是否通过命令行参数提供了文件路径
    if len(sys.argv) > 1:
        mgf_file = sys.argv[1]
        print(f"正在解析MGF文件: {mgf_file}")
    else:
        print("请提供MGF文件路径作为命令行参数。")
        print("使用方法: python mgf_cover.py <mgf_file_path>")
        return
    
    # 解析MGF文件
    spectra = parse_mgf_file(mgf_file)
    
    if spectra:
        print(f"\n成功从MGF文件解析了 {len(spectra)} 个谱图。")
    else:
        print("\n未能从MGF文件解析出任何谱图。")


if __name__ == "__main__":
    main()
