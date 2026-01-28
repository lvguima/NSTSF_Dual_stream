# Auto-generated ablation script for national_illness

$common = @(
  "--task_name", "long_term_forecast",
  "--is_training", "1",
  "--model", "DynamicGraphMixer",
  "--data", "custom",
  "--root_path", "./datasets",
  "--data_path", "national_illness.csv",
  "--features", "M",
  "--target", "OT",
  "--freq", "w",
  "--seq_len", "96",
  "--label_len", "48",
  "--pred_len", "96",
  "--enc_in", "7",
  "--dec_in", "7",
  "--c_out", "7",
  "--e_layers", "2",
  "--d_model", "128",
  "--d_ff", "256",
  "--batch_size", "64",
  "--train_epochs", "15",
  "--patience", "3",
  "--use_norm", "1",
  "--temporal_encoder", "tcn",
  "--tcn_kernel", "3",
  "--tcn_dilation", "2",
  "--graph_rank", "8",
  "--graph_scale", "8",
  "--graph_smooth_lambda", "0",
  "--adj_sparsify", "topk",
  "--adj_topk", "6",
  "--graph_base_mode", "mix",
  "--graph_base_alpha_init", "-8",
  "--graph_base_l1", "0.001",
  "--gate_mode", "per_var",
  "--graph_log_interval", "200",
  "--graph_log_topk", "5",
  "--graph_log_num_segments", "2",
  "--graph_log_dir", "./graph_logs",
)

function Run-Exp($id, $extra) {
  $modelId = "DGmix_national_illness_96_96_${id}"
  $logId = "national_illness_${id}"
  $args = $common + @("--model_id", $modelId, "--graph_log_exp_id", $logId) + $extra
  python -u run.py @args
}

Run-Exp "B0" @("--decomp_mode", "none", "--trend_head", "none", "--graph_map_norm", "none", "--gate_init", "-20")
Run-Exp "B1" @("--decomp_mode", "none", "--trend_head", "none", "--graph_map_norm", "none", "--gate_init", "-6")
Run-Exp "B2" @("--decomp_mode", "none", "--trend_head", "none", "--graph_map_norm", "ma_detrend", "--graph_map_window", "16", "--gate_init", "-6")
Run-Exp "B3" @("--decomp_mode", "ema", "--decomp_alpha", "0.1", "--trend_head", "linear", "--trend_head_share", "1", "--graph_map_norm", "none", "--gate_init", "-20")
Run-Exp "B4" @("--decomp_mode", "ema", "--decomp_alpha", "0.1", "--trend_head", "linear", "--trend_head_share", "1", "--graph_map_norm", "none", "--gate_init", "-6")
Run-Exp "B5" @("--decomp_mode", "ema", "--decomp_alpha", "0.1", "--trend_head", "linear", "--trend_head_share", "1", "--graph_map_norm", "ma_detrend", "--graph_map_window", "16", "--gate_init", "-6")
