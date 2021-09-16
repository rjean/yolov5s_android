#include <string.h>
#include <jni.h>
#include <math.h>
#include <iostream>
#include "nms.h"

using namespace std;

float sigmoid(float f){
    return (float)(1.0f / (1.0f + exp(-f)));
}

float revsigmoid(float f){
    const float eps = 1e-8;
    return -1.0f * (float)log((1.0f / (f + eps)) - 1.0f);
}

#define CLASS_NUM 80
#define max_wh 4096
void detector(
        vector<bbox>* bbox_candidates,
        JNIEnv *env,
        jobjectArray input,
        const int gridnum_y,
        const int gridnum_x,
        const int strides,
        const int (&anchorgrid)[3][2],
        const float conf_thresh,
        const int num_categories
        ){
    float revsigmoid_conf = revsigmoid(conf_thresh);
    //Warning: For now, we assume batch_size is always 1.
    for(int bi = 0; bi < 1; bi++){
        jobjectArray ptr_d0 = (jobjectArray)env->GetObjectArrayElement(input , bi);
        for(int gy = 0; gy < gridnum_y; gy++){
            jobjectArray ptr_d1 = (jobjectArray)env->GetObjectArrayElement(ptr_d0 ,gy);
            for(int gx = 0; gx < gridnum_x; gx++){
                jobjectArray ptr_d2 = (jobjectArray)env->GetObjectArrayElement(ptr_d1 ,gx);
                auto elmptr = env->GetFloatArrayElements((jfloatArray)ptr_d2 , nullptr);
                for(int ch = 0; ch < 3; ch++){
                    int offset = (num_categories+5) * ch;
                    auto elmptr_ch = elmptr + offset;
                    //don't apply sigmoid to all bbox candidates for efficiency
                    float obj_conf_unsigmoid = elmptr_ch[4];
                    //if (sigmoid(obj_conf_unsigmoid) < conf_thresh) continue;
                    if (obj_conf_unsigmoid >= revsigmoid_conf) {
                        //get maximum conf class
                        float max_class_conf = elmptr_ch[5];
                        int max_class_idx = 0;
                        for(int class_idx = 1; class_idx < num_categories; class_idx++){
                            float class_conf = elmptr_ch[class_idx + 5];
                            if (class_conf > max_class_conf){
                                max_class_conf = class_conf;
                                max_class_idx = class_idx;
                            }
                        }
                        // class conf filter
                        float bbox_conf = sigmoid(max_class_conf) * sigmoid(obj_conf_unsigmoid);
                        //if (bbox_conf < conf_thresh) continue;
                        // xywh2xyxy
                        // batched nms (by adding class * max_wh to coordinates,
                        // we can get nms result for all classes by just one nms call)
                        //grid[gridnum][gy][gx][0] = gx
                        //grid[gridnum][gy][gx][1] = gy
                        float cx = ((sigmoid(elmptr_ch[0]) * 2.0f) - 0.5f + (float)gx) * (float)strides;
                        float cy = ((sigmoid(elmptr_ch[1]) * 2.0f) - 0.5f + (float)gy) * (float)strides;
                        float w  = (sigmoid(elmptr_ch[2]) * sigmoid(elmptr_ch[2])) * 4.0f * (float)anchorgrid[ch][0];
                        float h  = (sigmoid(elmptr_ch[3]) * sigmoid(elmptr_ch[3])) * 4.0f * (float)anchorgrid[ch][1];
                        float x1 = cx - w / 2.0f + max_wh * max_class_idx;
                        float y1 = cy - h / 2.0f + max_wh * max_class_idx;
                        float x2 = cx + w / 2.0f + max_wh * max_class_idx;
                        float y2 = cy + h / 2.0f + max_wh * max_class_idx;
                        bbox box = bbox(x1, y1, x2, y2, bbox_conf, max_class_idx);
                        bbox_candidates->push_back(box);
                    }
                }
                env->ReleaseFloatArrayElements((jfloatArray)ptr_d2, elmptr, 0);
                env->DeleteLocalRef(ptr_d2);
            }
            env->DeleteLocalRef(ptr_d1);
        }
        env->DeleteLocalRef(ptr_d0);
    }
    env->DeleteLocalRef(input);
}

extern "C" jobjectArray Java_com_example_tflite_1yolov5_1test_TfliteRunner_postprocess (
        JNIEnv *env,
        jobject /* this */,
        jobjectArray input1,//80x80 or 40x40
        jobjectArray input2, //40x40 or 20x20,
        jobjectArray input3, //20x20 or 10x10
        jint input_height,
        jint input_width,
        jfloat conf_thresh,
        jfloat iou_thresh,
        jint num_categories
        ){
    //conf

    int anchorgrids[3][3][2]  = {
        {{10, 13}, {16, 30}, {33, 23}}, //80
        {{30, 61}, {62, 45}, {59, 119}}, //40
        {{116, 90}, {156, 198}, {373, 326}} //20
    };
    int anchorgrids_sl[3][3][2]  = {
            {{5, 7}, {13, 16}, {21, 25}}, //80
            {{30, 36}, {45, 55}, {60, 77}}, //40
            {{88, 110}, {138, 167}, {215, 275}} //20
    };
    if (num_categories==1)
    {
        for (int i=0;i<3;i++)
        {
            for (int j=0;j<3;j++)
            {
                for (int k=0;k<2;k++)
                {
                    anchorgrids[i][j][k] = anchorgrids_sl[i][j][k];
                }
            }
        }
    }

    const int strides[3] = {8, 16, 32};

    vector<bbox> bbox_candidates; //TODO: reserve
    //Detector
    detector(&bbox_candidates, env, input1,  input_height/ 8, input_width/ 8, strides[0], anchorgrids[0], conf_thresh, num_categories);
    detector(&bbox_candidates, env, input2, input_height / 16, input_width/ 16, strides[1], anchorgrids[1], conf_thresh, num_categories);
    detector(&bbox_candidates, env, input3, input_height / 32, input_width/ 32, strides[2], anchorgrids[2], conf_thresh, num_categories);
    //non-max-suppression
    vector<bbox> nms_results = nms(bbox_candidates, iou_thresh);

    //return 2-dimension array [detected_box][6(x1, y1, x2, y2, conf, class)]
    jobjectArray objArray;
    jclass floatArray = env->FindClass("[F");
    if (floatArray == NULL) return NULL;
    int size = nms_results.size();
    objArray = env->NewObjectArray(size, floatArray, NULL);
    if (objArray == NULL) return NULL;
    for(int i = 0; i < nms_results.size(); i++){
        int class_idx = nms_results[i].class_idx;
        float x1 = nms_results[i].x1 - class_idx * max_wh;
        float y1 = nms_results[i].y1 - class_idx * max_wh;
        float x2 = nms_results[i].x2 - class_idx * max_wh;
        float y2 = nms_results[i].y2 - class_idx * max_wh;
        float conf = nms_results[i].conf;
        float boxres[6] = {x1, y1, x2, y2, conf, (float)class_idx};
        jfloatArray iarr = env->NewFloatArray((jsize)6);
        if (iarr == NULL) return NULL;
        env->SetFloatArrayRegion(iarr, 0, 6, boxres);
        env->SetObjectArrayElement(objArray, i, iarr);
        env->DeleteLocalRef(iarr);
    }
    return objArray;
}