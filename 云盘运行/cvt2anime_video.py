import numpy as np
import tensorflow as tf 
import network
import guided_filter
import cv2, argparse
import os
from tqdm import tqdm

def parse_args():
    desc = "Tensorflow implementation of AnimeGANv3"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--video', type=str, help='video file or number for webcam')
    return parser.parse_args()

def process_image(img, x32=True):
    h, w = img.shape[:2]
    if x32: # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x%32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))
    img = img.astype(np.float32)/ 127.5 - 1.0
    return img

def post_precess(img, wh):
    img = (img.squeeze()+1.) / 2 * 255
    img = cv2.resize(img, (wh[0], wh[1]))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
    

def cvt2anime_video(video, output, model_path='saved_models', output_format='MP4V'):
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
     # load video
    vid = cv2.VideoCapture(video)
    vid_name = os.path.basename(video)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*output_format)
    video_out = cv2.VideoWriter(os.path.join(output, vid_name.rsplit('.', 1)[0] + "_AnimeGANv3.mp4"), codec, fps, (width, height))
    pbar = tqdm(total=total, ncols=80)
    pbar.set_description(f"Making: {os.path.basename(video).rsplit('.', 1)[0] + '_AnimeGANv3.mp4'}")
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame = np.asarray(np.expand_dims(process_image(frame),0))
        fake_img = sess.run(final_out, feed_dict={input_photo: frame})
        fake_img = post_precess(fake_img, (width, height))
        video_out.write(fake_img)
        pbar.update(1)

    pbar.close()
    vid.release()
    video_out.release()
    sess.close()
    return os.path.join(output, vid_name.rsplit('.', 1)[0] + "_AnimeGANv3.mp4")

if __name__ == '__main__':
    arg = parse_args()
    info = cvt2anime_video(arg.video, os.path.dirname(arg.video))  # 开始执行视频转换，耐心等待其运行完毕
    print(f'output video: {info}')  # 运行完毕， 输出运行信息