export WIDTH=768
export HEIGHT=224
export WEIGHTS=best

cd yolov5
./data/scripts/download_weights.sh #modify 'python' to 'python3' if needed
python3 export.py --weights ./$WEIGHTS.pt --img-size $HEIGHT $WIDTH --simplify

python3 /opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer/mo.py  --input_model $WEIGHTS.onnx  --input_shape [1,3,$WIDTH,$HEIGHT]  --output_dir ./openvino  --data_type FP32  --output Conv_245,Conv_294,Conv_343
openvino2tensorflow --model_path ./openvino/$WEIGHTS.xml --model_output_path tflite --output_pb --output_saved_model --output_no_quant_float32_tflite 
cd ..
cd convert_model/
python3 quantize.py --input_size $HEIGHT $WIDTH --pb_path /workspace/yolov5/tflite/model_float32.pb --output_path /workspace/yolov5/tflite/model_quantized.tflite
