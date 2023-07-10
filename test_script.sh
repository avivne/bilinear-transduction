python bilinear_transduction_regression.py --model-type bc
python bilinear_transduction_imitation.py --config-name configs/reach_metaworld.yaml --model-type bc
python bilinear_transduction_imitation.py --config-name configs/push_metaworld.yaml --model-type bc
python bilinear_transduction_imitation.py --config-name configs/adroit.yaml --model-type bc
python bilinear_transduction_imitation.py --config-name configs/slider.yaml --model-type bc

python bilinear_transduction_regression.py
python bilinear_transduction_imitation.py --config-name configs/reach_metaworld.yaml
python bilinear_transduction_imitation.py --config-name configs/push_metaworld.yaml
python bilinear_transduction_imitation.py --config-name configs/adroit.yaml
python bilinear_transduction_imitation.py --config-name configs/slider.yaml
