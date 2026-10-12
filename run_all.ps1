# 复现 Table 1：6 个迁移方向 × 5 次运行，Deep CORAL 与无适配基线（λ=0）各跑一遍
# 用法：powershell -ExecutionPolicy Bypass -File run_all.ps1 [-Lambda 1] [-Iters 2000] [-Seeds 0,1,2,3,4] [-Gpu 0]
param(
    [double]$Lambda = 1.0,
    [int]$Iters = 2000,
    [int[]]$Seeds = @(0, 1, 2, 3, 4),
    [int]$Gpu = 0
)
Set-Location $PSScriptRoot
$shifts = @(@("A","D"), @("A","W"), @("D","A"), @("D","W"), @("W","A"), @("W","D"))
foreach ($seed in $Seeds) {
    foreach ($s in $shifts) {
        foreach ($lam in @($Lambda, 0)) {
            python train.py --source $s[0] --target $s[1] --lambda_coral $lam --iters $Iters --seed $seed --gpu $Gpu
        }
    }
}
python plot.py --table
python plot.py --shift A2W --lambda_coral $Lambda --seed 0
