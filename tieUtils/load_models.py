import torch
import segmentation_models_pytorch as smp
from mmocr.apis.inferencers import MMOCRInferencer

def load_masker_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = smp.Unet('resnet50', classes=1, encoder_weights='imagenet').to(device)
    model.load_state_dict(torch.load("weights/model_weights_50.pth", map_location=device))
    model.eval()
    return model, device

def load_MMOCR():
    det_config = 'mmocr_dev/configs/textdet/dbnetpp/dbnetpp_swinv2_base_w16_in21k.py'  # noqa
    det_weight = 'checkpoints/mmocr/db_swin_mix_pretrain.pth'
    rec_config = 'mmocr_dev/configs/textrecog/abinet/abinet_20e_st-an_mj.py'
    rec_weight = 'checkpoints/mmocr/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # BUILD MMOCR
    mmocr_inferencer = MMOCRInferencer(
        det_config, det_weight, rec_config, rec_weight, device=device)
    
    return mmocr_inferencer