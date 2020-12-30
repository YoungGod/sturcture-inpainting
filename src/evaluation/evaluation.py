import subprocess
from metrics import evaluate
import os
from glob import glob

# Test model dirs


# Evaluation Data dirs: test images and masks
result_dir = './result'
model_names = os.listdir(result_dir)
data_dir = os.path.join(result_dir, model_names[0])


# """##### Evaluation #####"""
print('Start Evaluation...')
# For saving evaluation results
with open('evaluation.csv', mode='a') as f:
    f.write("model,     l1/mae,     pnsr,   ssim,   fid, uqi, vif\n")


# Our model
print("Our Models:")
for model_name in model_names:
    data_dir = os.path.join(result_dir, model_name)
    path_true = data_dir+'/sample_images'
    path_pred = data_dir+'/inpainted_images'

    l1, psnr, ssim, fid, uqi, vif = evaluate(path_true, path_pred)
    print("l1/mae:{:.4f}, psnr:{:.4f}, ssim:{:.4f}, fid:{:.4f},  uqi:{:.4f}, vif:{:.4f}".format(l1, psnr, ssim, fid, uqi, vif))
    with open('evaluation.csv', mode='a') as f:
        f.write("{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(model_name.split('-')[-1], l1, psnr, ssim, fid, uqi, vif))

with open('evaluation.csv', mode='a') as f:
    f.write("\n")
