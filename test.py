import tensorflow as tf
from video_processing import io

video_path = 'resources/person15_running_d1_uncomp.avi'
# print io.get_num_frames(video_path)

video = io.video_to_array(video_path=video_path, resize=[224, 224], start_frame=0, end_frame=20)
# video [channels, frames, height, width]
print video.shape
#tensor [frames, height, width, channels]
newtensor = io.video_to_tensor(video)
print newtensor
