from yolact_onnx import Yolact
"""
yolact_onnx 相对原来的 yolact 改动如下：
1) 最开始： use_jit = False
2) FPN:
   original: class FPN(ScriptModuleWrapper):
   修改为: class FPN(nn.Module):
3) 修改 YOLACT 的 forward 函数返回值：
   original: return self.detect(pred_outs, self)
   修改为: return pred_outs
"""

from utils.functions import SavePath
from data import cfg, set_cfg
import torch
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--onnx_path', default=None)

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)


if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)


    with torch.no_grad():
        print('Loading model...', end='')
        model = Yolact()

        # model.load_weights(args.trained_model)
        # model = torch.load(args.trained_model)
        # model.load_state_dict(args.trained_model)

        state_dict = torch.load(args.trained_model)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]
        
            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]
        model.load_state_dict(state_dict)

 
        

        # Input
        imgsz = [cfg.max_size, cfg.max_size]  
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # im = torch.zeros(1, 3, *imgsz).to(device)  
        im = torch.zeros(1, 3, *imgsz)

        # Exports
        if not args.onnx_path:
            f = args.trained_model[:args.trained_model.rfind(".")] + ".onnx"
        else:
            f = args.trained_model
        
        torch.onnx.export(
            # model.modules,
            model,
            im, 
            f,
            opset_version = 12,
        )

        print('='*80)
        print(f'onnx export success : {f}')
        print('='*80)
        print()


