# comfyui_segformer_b2_sleeves

This model can be used if you want to segment part of clothing , like sleeves, collars , torso. All credit to yolo12138/segformer-b2-cloth-parse-9 

This model is a fine-tuned version of mattmdjaga/segformer_b2_clothes on the cloth_parsing_mix dataset. It achieves the following results on the evaluation set:

Accuracy Background: 0.9964
Accuracy Upper Torso: 0.9857
Accuracy Left Pants: 0.9654
Accuracy Right Patns: 0.9664
Accuracy Skirts: 0.9065
Accuracy Left Sleeve: 0.9591
Accuracy Right Sleeve: 0.9662
Accuracy Outer Collar: 0.6491
Accuracy Inner Collar: 0.8015

install:

1.download and put on custom_nodes 

2.download models from https://huggingface.co/yolo12138/segformer-b2-cloth-parse-9 and put into model/segformer-b2-sleeves

