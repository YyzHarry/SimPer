"""
Augmentations for SimPer (and other SSL methods).
"""
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Optional, Tuple


def random_apply(func, p, x):
    return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)),
        lambda: func(x),
        lambda: x)


def resize_and_rescale(x, y, img_size):
    x = tf.cast(x, tf.float32)
    x = tf.image.resize(x, [img_size, img_size])
    return x, y


def _sample_or_pad_sequence_indices(sequence: tf.Tensor, num_steps: int,
                                    stride: int,
                                    offset: tf.Tensor) -> tf.Tensor:
    sequence_length = tf.shape(sequence)[0]
    sel_idx = tf.range(sequence_length)

    max_length = num_steps * stride + offset
    num_repeats = tf.math.floordiv(max_length + sequence_length - 1,
                                   sequence_length)
    sel_idx = tf.tile(sel_idx, [num_repeats])

    steps = tf.range(offset, offset + num_steps * stride, stride)
    return tf.gather(sel_idx, steps)


def sample_sequence(sequence: tf.Tensor,
                    num_steps: int,
                    random: bool = True,
                    stride: int = 1,
                    seed: Optional[int] = None) -> tf.Tensor:
    sequence_length = tf.shape(sequence)[0]

    if random:
        sequence_length = tf.cast(sequence_length, tf.float32)
        frame_stride = tf.cast(stride, tf.float32)
        max_offset = tf.cond(
            sequence_length > (num_steps - 1) * frame_stride,
            lambda: sequence_length - (num_steps - 1) * frame_stride,
            lambda: sequence_length)
        offset = tf.random.uniform((),
                                   maxval=tf.cast(max_offset, dtype=tf.int32),
                                   dtype=tf.int32,
                                   seed=seed)
    else:
        raise NotImplementedError(f"Only `random == True` is supported now.")
        offset = (sequence_length - num_steps * stride) // 2
        offset = tf.maximum(0, offset)

    indices = _sample_or_pad_sequence_indices(
        sequence=sequence, num_steps=num_steps, stride=stride, offset=offset)
    indices.set_shape((num_steps,))

    return tf.gather(sequence, indices)


def random_crop_resize(frames: tf.Tensor,
                       output_h: int,
                       output_w: int,
                       aspect_ratio: Tuple[float, float] = (0.75, 1.33),
                       area_range: Tuple[float, float] = (0.5, 1)) -> tf.Tensor:
    shape = tf.shape(frames)
    seq_len, _, _, channels = shape[0], shape[1], shape[2], shape[3]
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    factor = output_w / output_h
    aspect_ratio = (aspect_ratio[0] * factor, aspect_ratio[1] * factor)
    sample_distorted_bbox = tf.image.sample_distorted_bounding_box(
        shape[1:],
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=aspect_ratio,
        area_range=area_range,
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bbox
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    size = tf.convert_to_tensor((seq_len, target_height, target_width, channels))
    offset = tf.convert_to_tensor((0, offset_y, offset_x, 0))
    frames = tf.slice(frames, offset, size)
    frames = tf.cast(tf.image.resize(frames, (output_h, output_w)), frames.dtype)
    return frames


def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
    radius = tf.cast(kernel_size / 2, dtype=tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), dtype=tf.float32)
    blur_filter = tf.exp(
        -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(
        image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(
        blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred


def random_blur(image, height, p=0.2):
    def _transform(image):
        sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
        return gaussian_blur(
            image, kernel_size=height // 20, sigma=sigma, padding='SAME')

    return random_apply(_transform, p=p, x=image)


def random_flip_left_right(frames: tf.Tensor, seed: Optional[int] = None) -> tf.Tensor:
    is_flipped = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32, seed=seed)
    frames = tf.cond(
        tf.equal(is_flipped, 1),
        true_fn=lambda: tf.image.flip_left_right(frames),
        false_fn=lambda: frames)
    return frames


def random_flip_up_down(frames: tf.Tensor, seed: Optional[int] = None) -> tf.Tensor:
    is_flipped = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32, seed=seed)
    frames = tf.cond(
        tf.equal(is_flipped, 1),
        true_fn=lambda: tf.image.flip_up_down(frames),
        false_fn=lambda: frames)
    return frames


def random_rotation(frames: tf.Tensor, seed: Optional[int] = None) -> tf.Tensor:
    is_flipped = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32, seed=seed)
    frames = tf.cond(
        tf.equal(is_flipped, 1),
        true_fn=lambda: tf.image.rot90(frames),
        false_fn=lambda: frames)
    return frames


def to_grayscale(image, keep_channels=True):
    image = tf.image.rgb_to_grayscale(image)
    if keep_channels:
        image = tf.tile(image, [1, 1, 3])
    return image


def random_grayscale_3d(frames, p=0.2):
    num_frames, width, height, channels = frames.shape.as_list()
    big_image = tf.reshape(frames, [num_frames * width, height, channels])
    big_image = random_apply(to_grayscale, p=p, x=big_image)
    return tf.reshape(big_image, [num_frames, width, height, channels])


def random_brightness(image, max_delta=0.3):
    factor = tf.random.uniform(
        [], tf.maximum(1.0 - max_delta, 0), 1.0 + max_delta)
    image = image * factor
    return image


def random_reverse(frames: tf.Tensor, seed: Optional[int] = None) -> tf.Tensor:
    is_flipped = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32, seed=seed)
    frames = tf.cond(
        tf.equal(is_flipped, 1),
        true_fn=lambda: tf.experimental.numpy.flip(frames, axis=0),
        false_fn=lambda: frames)
    return frames


# Arbitrary speed / frequency augmentation for SimPer
def arbitrary_speed_subsample(frames_speed,
                              num_steps: int,
                              random: bool,
                              img_size: int,
                              channels: int = 3,
                              stride: int = 1,
                              seed: Optional[int] = None) -> tf.Tensor:
    frames, speed = frames_speed
    frame_len = tf.cast(tf.shape(frames)[0], tf.float32)
    max_frame_len = tf.math.floordiv(frame_len, speed) if speed > 1 else frame_len

    x_ref = tf.range(0, speed * (len(frames) - 0.5), speed, dtype=tf.float32)
    x_ref = tf.stack([x_ref] * (img_size * img_size * channels))
    new_frames = tfp.math.batch_interp_regular_1d_grid(
        x=x_ref,
        x_ref_min=[0] * (img_size * img_size * channels),
        x_ref_max=[len(frames)] * (img_size * img_size * channels),
        y_ref=tf.transpose(tf.reshape(frames, [len(frames), -1]))
    )
    sequence = tf.reshape(
        tf.transpose(new_frames), frames.shape.as_list()
    )[:tf.cast(max_frame_len, tf.int32)]

    sequence_length = tf.shape(sequence)[0]

    if random:
        sequence_length = tf.cast(sequence_length, tf.float32)
        frame_stride = tf.cast(stride, tf.float32)
        max_offset = tf.cond(
            sequence_length > (num_steps - 1) * frame_stride,
            lambda: sequence_length - (num_steps - 1) * frame_stride,
            lambda: sequence_length)
        offset = tf.random.uniform((),
                                   maxval=tf.cast(max_offset, dtype=tf.int32),
                                   dtype=tf.int32,
                                   seed=seed)
    else:
        raise NotImplementedError(f"Only `random == True` is supported now.")
        offset = (sequence_length - num_steps * stride) // 2
        offset = tf.maximum(0, offset)

    indices = _sample_or_pad_sequence_indices(
        sequence=sequence, num_steps=num_steps, stride=stride, offset=offset)
    indices.set_shape((num_steps,))

    return tf.gather(sequence, indices)


# (batched) Arbitrary speed / frequency augmentation for SimPer
def batched_arbitrary_speed(frames, num_diff_speeds, speed_range=(0.5, 2)):
    random_speeds = tf.random.uniform([num_diff_speeds],
                                      minval=speed_range[0],
                                      maxval=speed_range[1],
                                      dtype=tf.float32)
    random_speeds = tf.sort(random_speeds)
    random_speeds = tf.concat([random_speeds, random_speeds], 0)

    # construct (2 * M) sub-video batch for SimPer loss
    batched_frames = tf.stack([frames] * num_diff_speeds * 2)
    batched_frames = tf.map_fn(
        arbitrary_speed_subsample, (batched_frames, random_speeds),
        fn_output_signature=tf.float32)

    return batched_frames, random_speeds
