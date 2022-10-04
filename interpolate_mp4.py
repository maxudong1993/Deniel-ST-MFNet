import argparse
import torch
import torchvision.transforms.functional as TF
import models
import os
import skvideo.io
import cv2
from PIL import Image
from tqdm import tqdm
from utility import read_frame_yuv2rgb, tensor2rgb, FoldUnfold


parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--net', type=str, default='STMFNet')
parser.add_argument('--checkpoint', type=str, default='train_results/checkpoint/model_epoch070.pth')
parser.add_argument('--mp4_path', type=str, help='path of the input MP4 file')
parser.add_argument('--size', type=str, default='1920x1080', help='resolution of the input mp4 file')
parser.add_argument('--patch_size', type=int, default=None, help='patch size for block-wise eval')
parser.add_argument('--overlap', type=int, default=None, help='overlap between patches for block-wise eval, SHOULD BE EVEN NUMBER')
parser.add_argument('--batch_size', type=int, default=None, help='batch size for block-wise eval')
parser.add_argument('--out_fps', type=int, default=30, help='fps of the output mp4')
parser.add_argument('--out_dir', type=str, default='.', help='dir to store output video')

# model parameters
parser.add_argument('--featc', nargs='+', type=int, default=[64, 128, 256, 512])
parser.add_argument('--featnet', type=str, default='UMultiScaleResNext')
parser.add_argument('--featnorm', type=str, default='batch')
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)
parser.add_argument('--finetune_pwc', dest='finetune_pwc', default=False,  action='store_true')


def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    if args.patch_size != 0:
        print(f'====Note: Using block-wise evaluation with block size={args.patch_size}, overlap={args.overlap}, batch size={args.batch_size}')
        print('====This may generate unwanted block artefacts.')

    # Initiate the model
    model = getattr(models, args.net)(args).cuda()
    print('Loading the model...')
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Setup output file
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    _, fname = os.path.split(args.mp4_path)
    seq_name = fname.split('.')[0]
    width, height = args.size.split('x')
    # bit_depth = 16 if '16bit' in fname else 10 if '10bit' in fname else 8
    # pix_fmt = '444' if '444' in fname else '420'
    try:
        width = int(width)
        height = int(height)
    except:
        print('Invalid size, should be \'<width>x<height>\'')
        return 
    img_array = []
    outname = '{}_{}x{}_{}fps_{}.mp4'.format(seq_name, width, height, args.out_fps, args.net)
    # writer = skvideo.io.FFmpegWriter(os.path.join(args.out_dir, outname) ,
    #     inputdict={
    #         '-r': str(args.out_fps)
    #     },
    #     outputdict={
    #         '-pix_fmt': 'yuv420p',
    #         # '-c:v':'libx264',
    #         '-s': '{}x{}'.format(width,height),
    #         '-r': str(args.out_fps),
    #         '-vcodec': 'libx264',  #use the h.264 codec
    #         '-crf': '0',           #set the constant rate factor to 0, which is lossless
    #         '-preset':'veryslow'   #the slower the better compression, in princple, try
    #                                #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    #     }
    # )

    # Start interpolation
    print('Using model {} to upsample file {}'.format(args.net, fname))
    # stream = open(args.mp4_path, 'r')
    # file_size = os.path.getsize(args.yuv_path)
    video = cv2.VideoCapture(args.mp4_path)
    video_fps = video.get(cv2.CAP_PROP_FPS)
    print('video fps is: {}'.format(video_fps))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for t in tqdm(range(0, num_frames-3)):
        print('t: {}'.format(t))
        print(' frame position before: {}'.format(video.get(cv2.CAP_PROP_POS_FRAMES)))
        video.set(cv2.CAP_PROP_POS_FRAMES, t)
        _, rawFrame0 = video.read()
        _, rawFrame1 = video.read()
        _, rawFrame2 = video.read()
        _, rawFrame3 = video.read()
        print(' frame position after: {}'.format(video.get(cv2.CAP_PROP_POS_FRAMES)))
        print('Frame Format: {}. size: {}'.format(rawFrame0.__class__,rawFrame0.shape))


        frame0 = TF.to_tensor(rawFrame0)[None, ...].cuda()
        frame1 = TF.to_tensor(rawFrame1)[None, ...].cuda()
        frame2 = TF.to_tensor(rawFrame2)[None, ...].cuda()
        frame3 = TF.to_tensor(rawFrame3)[None, ...].cuda()

        with torch.no_grad():
            if args.patch_size != None:
                #### (NOT RECOMMENDED) block-wise evaluation if GPU memory is not enough
                #### This will produce artefacts near block edges
                patch_maker = FoldUnfold(height, width, patch_size=args.patch_size, overlap=args.overlap)
                patches0, patches1, patches2, patches3 = patch_maker.fold_to_patches(frame0, frame1, frame2,
                                                                                     frame3)  # (num_patches,3,patch_size,patch_size)
                patches_out = torch.empty_like(patches0)
                # interpolate each patch
                for i in range(0, patches0.shape[0], args.batch_size):
                    out = model(patches0[i:i + args.batch_size], patches1[i:i + args.batch_size],
                                patches2[i:i + args.batch_size], patches3[i:i + args.batch_size])
                    patches_out[i:i + args.batch_size] = out
                # form frame from patches
                out = patch_maker.unfold_to_frame(patches_out)
            else:
                out = model(frame0, frame1, frame2, frame3)
        # print('frame0 shape {}'.format(frame0.shape))
        # print('tensor2rgb frame0 shape {}'.format(tensor2rgb(frame0).shape))

        # write to output video
        if t == 0:
            img_array.append(tensor2rgb(frame0)[0])
            img_array.append(tensor2rgb(frame0)[0])
            img_array.append(tensor2rgb(frame1)[0])
            # writer.writeFrame(tensor2rgb(frame0)[0])
            # writer.writeFrame(tensor2rgb(frame0)[0])  # repeat the first frame
            # writer.writeFrame(tensor2rgb(frame1)[0])
        img_array.append(tensor2rgb(out)[0])
        img_array.append(tensor2rgb(frame2)[0])
        # writer.writeFrame(tensor2rgb(out)[0])
        # writer.writeFrame(tensor2rgb(frame2)[0])
        if t == num_frames - 4:
            img_array.append(tensor2rgb(frame3)[0])
            img_array.append(tensor2rgb(frame3)[0])
            # writer.writeFrame(tensor2rgb(frame3)[0])
            # writer.writeFrame(tensor2rgb(frame3)[0])  # repeat the last frame

    video.release()
    cv2writer = cv2.VideoWriter(outname,cv2.VideoWriter_fourcc(*'DIVX'),args.out_fps,(width,height))
    for i in range(len(img_array)):
        cv2writer.write(img_array[i])
    cv2.writer.release()
    # writer.close()  # close the writer


if __name__ == "__main__":
    main()
