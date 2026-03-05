"""
RevBayes 系统发育树推断模块
纯净版：根据命令行传入的 image_type 和 model_type 自动精准匹配文件。
解除了参数的名字限制，同时保留了动态抽样频率。
"""

import os
import subprocess
import argparse

# RevBayes 脚本模板 (Brownian Motion Model / REML PIC)
REV_TEMPLATE = """## This script is generated automatically by revbayes.py
fl <- "{nex_path}"  
outtr <- "{out_dir}/OUTTREEFILE.trees" 
outlf <- "{out_dir}/OUTLOGFILE" 
outsm <- "{out_dir}/OUTSUMFILE.tre" 

contData <- readContinuousCharacterData(fl)

numTips = contData.ntaxa()
numNodes = numTips * 2 - 1
names = contData.names()
diversification ~ dnLognormal(0,1)
turnover = 0 
speciation := diversification + turnover
extinction := turnover 
sampling_fraction <- 1

psi ~ dnBirthDeath(lambda=speciation, mu=extinction, rho=sampling_fraction, rootAge=1, taxa=names) 
mvi = 0 

moves[++mvi] = mvSubtreeScale(psi, weight=5.0)
moves[++mvi] = mvNodeTimeSlideUniform(psi, weight=5.0) 
moves[++mvi] = mvNNI(psi, weight=5.0)
moves[++mvi] = mvFNPR(psi, weight=5.0)

# 动态计算的文件保存频率
monitors[1] = mnFile(filename=outtr, printgen={file_printgen}, separator = TAB, psi)

logSigma ~ dnNormal(0,1) 
sigma := 10^logSigma
moves[++mvi] = mvSlide(logSigma, delta=1.0, tune=true, weight=2.0)

traits ~ dnPhyloBrownianREML(psi, branchRates=1.0, siteRates=sigma, nSites=contData.nchar())

traits.clamp(contData) 
# 动态计算的屏幕打印频率
monitors[2] = mnScreen(printgen={screen_printgen}, sigma)
monitors[3] = mnFile(filename=outlf, printgen={file_printgen}, separator = TAB, sigma)
bmv = model(sigma) 

chain = mcmc(bmv, monitors, moves)
chain.burnin(generations={burnin}, tuningInterval=500)
chain.run({generations})

treetrace = readTreeTrace(file=outtr, treetype="clock")
map = mapTree(file=outsm, treetrace)
q()
"""

def run_revbayes(image_type, model_type, base_output_dir, rb_executable, burnin, generations):
    """根据传入的参数，精准拼装路径并执行 RevBayes"""
    
    # === 核心逻辑：完全依赖传入的两个参数拼接文件名 ===
    nex_filename = f"{image_type}_{model_type}_continuous_traits.nex"
    nex_path = os.path.join(base_output_dir, "features", nex_filename)
    
    if not os.path.exists(nex_path):
        raise FileNotFoundError(f"\n找不到特征文件！\n期望的路径是: {nex_path}\n请先确保 main.py 已经成功生成了该文件。")
        
    if not os.path.exists(rb_executable):
        raise FileNotFoundError(f"找不到 RevBayes 执行文件: {rb_executable}")

    out_dir = os.path.join(base_output_dir, "trees", "revbayes", f"{image_type}_{model_type}")
    os.makedirs(out_dir, exist_ok=True)
    
    # 动态计算抽样频率（防止文件过大或刷屏）
    screen_printgen = max(100, generations // 100)
    file_printgen = max(10, generations // 5000)

    rev_script_content = REV_TEMPLATE.format(
        nex_path=os.path.abspath(nex_path),
        out_dir=os.path.abspath(out_dir),
        burnin=burnin,
        generations=generations,
        screen_printgen=screen_printgen,
        file_printgen=file_printgen
    )
    
    rev_script_path = os.path.join(out_dir, "run_continuous_traits.rev")
    with open(rev_script_path, "w") as f:
        f.write(rev_script_content)
        
    print("=" * 60)
    print("RevBayes 系统发育树构建启动")
    print("=" * 60)
    print(f"正在读取文件: {nex_filename}")
    print(f"输出结果目录: {out_dir}")
    print(f"MCMC 代数: Burn-in = {burnin}, Generations = {generations}")
    print(f"抽样频率: 屏幕每 {screen_printgen} 代打印, 文件每 {file_printgen} 代保存")
    print("=" * 60)
    
    try:
        process = subprocess.Popen(
            [rb_executable, rev_script_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            universal_newlines=True
        )
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode == 0:
            print("\n" + "=" * 60)
            print(f"✓ RevBayes 运行成功！")
            print(f"✓ 最终树文件: {os.path.join(out_dir, 'OUTSUMFILE.tre')}")
            print("=" * 60)
        else:
            print(f"\nRevBayes 运行异常终止，返回码: {process.returncode}")
            
    except Exception as e:
        print(f"\n执行 RevBayes 时发生错误: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 RevBayes 构建系统发育树")
    
    # 解除了 choices 限制，你想传入什么名字都可以
    parser.add_argument('--image_type', type=str, default='specimen', help='图像类型')
    parser.add_argument('--model_type', type=str, default='resnet50', help='模型类型')
    
    parser.add_argument('--output_dir', type=str, default='/data/yutong/line/codee/outputs', 
                        help='基础输出路径')
    parser.add_argument('--rb_path', type=str, default='/data/yutong/software/revbayes-v1.2.0/bin/rb', 
                        help='rb 程序的绝对路径')
    parser.add_argument('--burnin', type=int, default=1000, help='老化期代数')
    parser.add_argument('--gen', type=int, default=10000, help='运行代数')
    
    args = parser.parse_args()
    
    run_revbayes(
        image_type=args.image_type,
        model_type=args.model_type,
        base_output_dir=args.output_dir,
        rb_executable=args.rb_path,
        burnin=args.burnin,
        generations=args.gen
    )