sample_rate = 44100
n_mels = 128
n_fft = 2048
hop_length = 1024
win_length = 2048
fmin = 0
clip_frames = 500
fmax = sample_rate / 2

TAU_audio_root_path = "/mnt2/lwy/zelda/datasets/TAU-urban-acoustic-scenes-2020-mobile-development/"
TAU_meta_csv_path = "/mnt2/lwy/zelda/datasets/TAU-urban-acoustic-scenes-2020-mobile-development/meta.csv"
TAU_fea_root_path = "/mnt2/yyp/ICME2024/resnet/features"

CAS_audio_root_path = "/mnt2/yyp/ICME2024/baseline/ICME2024ASC/data/ICME2024_GC_ASC_dev"
CAS_meta_csv_path = "/mnt2/yyp/ICME2024/baseline/ICME2024ASC/data/ICME2024_ASC_dev_label.csv"
CAS_fea_root_path = "/mnt2/yyp/ICME2024/resnet/CAS_features"

outpath = "output/"

eval_audio_root_path = "/mnt2/yyp/ICME2024/evaluation/ICME2024_GC_ASC_eval"
eval_fea_root_path = "/mnt2/yyp/ICME2024/evaluation/features"
eval_meta_csv_path = "/mnt2/yyp/ICME2024/evaluation/ICME2024_ASC_eval.csv"
