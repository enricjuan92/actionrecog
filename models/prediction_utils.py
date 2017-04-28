import tensorflow as tf
from video_processing import io

def call_vgg16(video_path, sframe, eframe):
    video = io.video_to_array(video_path=video_path, start_frame=sframe, end_frame=eframe)
    # video [channels, frames, height, width]
    # tensor [frames, height, width, channels]
    t_images = io.video_to_tensor(video)

    url = 'http://cv-tricks.com/wp-content/uploads/2017/03/pexels-photo-361951.jpeg'
    image_string = urllib.urlopen(url).read()
    print image_string, 'image string'
    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images = tf.expand_dims(processed_image, 0)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, _ = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)
    probabilities = tf.nn.softmax(logits)