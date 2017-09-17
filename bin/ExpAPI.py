import glob
import os
import shutil
import argparse
import ExpressionAPI 
import numpy as np
from skimage.io import imread, imsave 
from subprocess import check_output
import math


def run_pipeline_on_video_clip(args):
    batch_size = 20
    FE = ExpressionAPI.Feature_Extractor(output='flatten_1', verbose=0, batch_size=batch_size )
    print('processing video clip:', args.input)

    shutil.rmtree(args.output, ignore_errors=True)
    os.makedirs(args.output+'/.tmp')

    # # split video to frames
    cmd = [ 'ffmpeg', '-i', args.input, args.output+'/.tmp/%06d.png' ]
    check_output(cmd)

    # apply ExpressionAPI for each frame
    frames = sorted(list(glob.glob(args.output+'/.tmp/*.png')))
    t0, t1  = 0, batch_size
    nb_batches = math.ceil((len(frames)/batch_size))
    for i in range(nb_batches):
        img_batch = [imread(f) for f in frames[t0:t1]]

        img_box, pts, pts_raw = FE.get_input_features_from_numpy(img_batch)
        au_disfa, au_fera, z = FE.mod.predict(img_box)

        for n,file_name in enumerate(frames[t0:t1]):
            img_out = ExpressionAPI.visualize.make_hub(img_batch[n], pts_raw[n], au_disfa[n], img_box[n])
            imsave(args.output+'/.tmp/hub_'+str(n+t0).zfill(6)+'.png', img_out)


        with open(args.output+'/AU_DISFA.csv','a') as f:
            for s in au_disfa:
                s = [str(ii) for ii in s]
                f.write(','.join(s)+'\n')

        with open(args.output+'/AU_FERA.csv','a') as f:
            for s in au_fera:
                s = [str(ii) for ii in s]
                f.write(','.join(s)+'\n')

        with open(args.output+'/Z.csv','a') as f:
            z = z.reshape(z.shape[0],-1)
            for s in z:
                s = [str(ii) for ii in s]
                f.write(','.join(s)+'\n')

        with open(args.output+'/pts_box.csv','a') as f:
            pts = pts.reshape(pts.shape[0],-1)
            for s in pts:
                s = [str(ii) for ii in s]
                f.write(','.join(s)+'\n')

        with open(args.output+'/pts_raw.csv','a') as f:
            pts_raw = pts_raw.reshape(pts.shape[0],-1)
            for s in pts:
                s = [str(ii) for ii in s]
                f.write(','.join(s)+'\n')


        t0 += batch_size
        t1 += batch_size
        print('batch:',(i+1),'/',nb_batches)



    # merge frames to video
    cmd = ['ffmpeg', '-i', args.output+'/.tmp/hub_%06d.png', '-vf', 'fps='+str(args.fps) , args.output+'/out.mp4']
    check_output(cmd)

    shutil.rmtree(args.output+'/.tmp', ignore_errors=True)





if __name__=='__main__':

    parser = argparse.ArgumentParser(description='ExpressionAPI for feature extraction and AU intensity estimation')

    # input output
    # parser.add_argument("-i","--input", type=str, default='tests/test_videos/arnold.mp4')
    parser.add_argument("-i","--input", type=str, default='/homes/rw2614/projects/ExpressionAPI_RESNET/tests/test_videos/girls_speaking.mp4')
    # parser.add_argument("-i","--input", type=str, default='/homes/rw2614/test_videos/SVL_C2_S049_P097_VC1_003160_004143.avi')
    parser.add_argument("-o","--output",type=str, default='girls/')
    parser.add_argument("-f","--fps",type=str, default='25')
    args = parser.parse_args()

    suffix = args.input.split('.')[-1]

    # apply pipeline on video clip
    if suffix in ['mp4','avi']:
        run_pipeline_on_video_clip(args)

    if suffix in ['jpg','png','bmp']:
        print('apply pipeline on image')

    if len(suffix)>5:
        print('apply pipeline on folder of images')

