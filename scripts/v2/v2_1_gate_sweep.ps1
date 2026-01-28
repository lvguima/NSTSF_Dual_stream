param(
    [ValidateSet("E1-A", "E1-B", "E1-C", "all")]
    [string]$Stage = "E1-A",
    [string]$BestGateMode = "scalar",
    [float]$BestGateInit = -4.0,
    [int]$GraphLogInterval = 200
)

$python = "python"
$baseArgs = @(
    "--task_name", "long_term_forecast",
    "--is_training", "1",
    "--model", "DynamicGraphMixer",
    "--data", "ETTm1",
    "--root_path", "./datasets",
    "--data_path", "ETTm1.csv",
    "--features", "M",
    "--target", "OT",
    "--freq", "t",
    "--seq_len", "96",
    "--label_len", "48",
    "--pred_len", "96",
    "--e_layers", "2",
    "--d_model", "128",
    "--d_ff", "256",
    "--enc_in", "7",
    "--dec_in", "7",
    "--c_out", "7",
    "--batch_size", "64",
    "--train_epochs", "15",
    "--patience", "3",
    "--use_norm", "1",
    "--graph_scale", "8",
    "--graph_rank", "8",
    "--graph_smooth_lambda", "0",
    "--temporal_encoder", "tcn",
    "--tcn_kernel", "3",
    "--tcn_dilation", "2",
    "--graph_source", "content_mean",
    "--graph_base_mode", "mix",
    "--graph_base_alpha_init", "-8",
    "--graph_base_l1", "0.001",
    "--adj_sparsify", "topk",
    "--adj_topk", "6",
    "--graph_log_interval", $GraphLogInterval.ToString(),
    "--graph_log_topk", "5",
    "--graph_log_num_segments", "2",
    "--graph_log_dir", "./graph_logs"
)

function Invoke-Run {
    param(
        [string]$ExpId,
        [string]$ModelId,
        [string[]]$ExtraArgs
    )
    $args = @("--model_id", $ModelId, "--graph_log_exp_id", $ExpId) + $baseArgs + $ExtraArgs
    Write-Host "Running $ExpId"
    & $python "run.py" @args
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed: $ExpId"
    }
}

$gateModes = @("none", "scalar", "per_var", "per_token")
$gateInits = @(-8, -6, -4, -2, 0)
$graphScales = @(1, 2, 4, 8, 16)

if ($Stage -eq "E1-A" -or $Stage -eq "all") {
    foreach ($mode in $gateModes) {
        $expId = "v2_1_E1-A_gate_${mode}"
        $modelId = "DynamicGraphMixer_TCN_ETTm1_96_96_E1A_${mode}"
        Invoke-Run $expId $modelId @("--gate_mode", $mode, "--gate_init", "-4")
    }
}

if ($Stage -eq "E1-B" -or $Stage -eq "all") {
    foreach ($mode in $gateModes) {
        foreach ($init in $gateInits) {
            $expId = "v2_1_E1-B_${mode}_init${init}"
            $modelId = "DynamicGraphMixer_TCN_ETTm1_96_96_E1B_${mode}_init${init}"
            Invoke-Run $expId $modelId @("--gate_mode", $mode, "--gate_init", $init.ToString())
        }
    }
}

if ($Stage -eq "E1-C" -or $Stage -eq "all") {
    foreach ($scale in $graphScales) {
        $expId = "v2_1_E1-C_${BestGateMode}_scale${scale}"
        $modelId = "DynamicGraphMixer_TCN_ETTm1_96_96_E1C_${BestGateMode}_scale${scale}"
        Invoke-Run $expId $modelId @(
            "--gate_mode", $BestGateMode,
            "--gate_init", $BestGateInit.ToString(),
            "--graph_scale", $scale.ToString()
        )
    }
}
