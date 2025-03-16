import os
import argparse
import subprocess
import time

def create_directories():
    """创建项目所需的目录"""
    directories = [
        'data/yelp/images',
        'models',
        'results',
        'results/visualizations'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ 目录创建成功: {directory}")

def run_script(script_name, description, args=None):
    """运行指定的Python脚本"""
    print(f"\n{'='*80}")
    print(f"执行: {description}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    cmd = ['python', script_name]
    if args:
        cmd.extend(args)
    
    result = subprocess.run(cmd, text=True, capture_output=True)
    
    if result.returncode == 0:
        print(f"✅ {description}成功完成")
        print(result.stdout)
    else:
        print(f"❌ {description}执行失败")
        print(f"错误信息: {result.stderr}")
        return False
    
    duration = time.time() - start_time
    print(f"⏱️ 耗时: {duration:.2f}秒 ({duration/60:.2f}分钟)")
    return True

def main():
    parser = argparse.ArgumentParser(description='多模态情感分析系统 - Yelp数据集版本')
    parser.add_argument('--skip-preprocess', action='store_true', help='跳过数据预处理步骤')
    parser.add_argument('--skip-train', action='store_true', help='跳过训练步骤')
    parser.add_argument('--skip-eval', action='store_true', help='跳过评估步骤')
    parser.add_argument('--skip-viz', action='store_true', help='跳过可视化步骤')
    parser.add_argument('--data-dir', type=str, default='yelp_dataset', help='Yelp数据集所在目录')
    parser.add_argument('--output-dir', type=str, default='data/yelp', help='输出数据目录')
    parser.add_argument('--sample-size', type=int, default=5000, help='处理的样本数量')
    parser.add_argument('--batch-size', type=int, default=16, help='训练批量大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU训练')
    
    args = parser.parse_args()
    
    # 创建目录
    create_directories()
    
    # 1. 数据预处理
    if not args.skip_preprocess:
        preprocess_args = [
            f'--data_dir={args.data_dir}',
            f'--output_dir={args.output_dir}',
            f'--sample_size={args.sample_size}'
        ]
        success = run_script('preprocess_yelp.py', '数据预处理', preprocess_args)
        if not success:
            print("数据预处理失败，终止执行")
            return
    else:
        print("跳过数据预处理步骤")
    
    # 2. 模型训练
    if not args.skip_train:
        train_args = [
            f'--data_dir={args.output_dir}',
            f'--batch_size={args.batch_size}',
            f'--epochs={args.epochs}'
        ]
        if args.cpu:
            train_args.append('--cpu')
            
        success = run_script('train.py', '模型训练', train_args)
        if not success:
            print("模型训练失败，终止执行")
            return
    else:
        print("跳过模型训练步骤")
    
    # 3. 模型评估
    if not args.skip_eval:
        eval_args = [
            f'--data_dir={args.output_dir}',
            f'--batch_size={args.batch_size * 2}',  # 评估时可以用更大的批量
            '--visualize_attributions'
        ]
        if args.cpu:
            eval_args.append('--cpu')
            
        success = run_script('evaluate.py', '模型评估', eval_args)
        if not success:
            print("模型评估失败，终止执行")
            return
    else:
        print("跳过模型评估步骤")
    
    # 4. 结果可视化
    if not args.skip_viz:
        success = run_script('visualize_results.py', '结果可视化')
        if not success:
            print("结果可视化失败")
    else:
        print("跳过结果可视化步骤")
    
    print("\n🎉 多模态情感分析系统执行完成!")
    print(f"- 数据保存在: {os.path.abspath(args.output_dir)}")
    print(f"- 模型保存在: {os.path.abspath('models')}")
    print(f"- 评估结果在: {os.path.abspath('results')}")
    print(f"- 可视化结果在: {os.path.abspath('results/visualizations')}")
    print("\n如需查看汇总仪表板，请打开:")
    print(f"{os.path.abspath('results/multimodal_sentiment_dashboard.png')}")

if __name__ == "__main__":
    main()