import numpy as np


def yuv_import(file_path, dims, num_frames=1, start_frame=0, frames=None, yuv444=False):
    """
    Import frame images from a YUV file.
    :param file_path: Path of the file.
    :param dims: (height, width) of the frames.
    :param num_frames: Number of the consecutive frames to be imported.
    :param start_frame: Index of the frame to be started. The first frame is indexed as 0.
    :param frames: Indexes of the frames to be imported. Inconsecutive frames are supported.
    :param yuv444: Whether the YUV file is in YUV444 mode.
    :return: Y, U, V, all as the numpy ndarray.
    """

    fp = open(file_path, 'rb')
    ratio = 3 if yuv444 else 1.5
    blk_size = int(np.prod(dims) * ratio)
    if frames is None:
        assert num_frames > 0
        fp.seek(blk_size * start_frame, 0)

    height, width = dims
    Y = []
    U = []
    V = []
    if yuv444:
        height_half = height
        width_half = width
    else:
        height_half = height // 2
        width_half = width // 2

    if frames is not None:
        previous_frame = -1
        for frame in frames:
            fp.seek(blk_size * (frame - previous_frame - 1), 1)
            Yt = np.fromfile(fp, dtype=np.uint8, count=width * height).reshape((height, width))
            Ut = np.fromfile(fp, dtype=np.uint8, count=width_half * height_half).reshape((height_half, width_half))
            Vt = np.fromfile(fp, dtype=np.uint8, count=width_half * height_half).reshape((height_half, width_half))
            previous_frame = frame
            Y = Y + [Yt]
            U = U + [Ut]
            V = V + [Vt]

    else:
        for i in range(num_frames):
            Yt = np.fromfile(fp, dtype=np.uint8, count=width * height).reshape((height, width))
            Ut = np.fromfile(fp, dtype=np.uint8, count=width_half * height_half).reshape((height_half, width_half))
            Vt = np.fromfile(fp, dtype=np.uint8, count=width_half * height_half).reshape((height_half, width_half))
            Y = Y + [Yt]
            U = U + [Ut]
            V = V + [Vt]

    fp.close()
    return np.array(Y), np.array(U), np.array(V)


def yuv2rgb(Y, U, V):
    """
    Convert YUV to RGB.
    """

    if not Y.shape == U.shape:
        U = U.repeat(2, axis=1).repeat(2, axis=2).astype(np.float64)
        V = V.repeat(2, axis=1).repeat(2, axis=2).astype(np.float64)

    Y = Y.astype(np.float64)
    U = U.astype(np.float64)
    V = V.astype(np.float64)
    U -= 128.0
    V -= 128.0

    rr = 1.001574765442552 * Y + 0.002770649292941 * U + 1.574765442551769 * V
    gg = 0.999531875325065 * Y - 0.188148872370914 * U - 0.468124674935631 * V
    bb = 1.000000105739993 * Y + 1.855609881994441 * U + 1.057399924810358e-04 * V

    rr = rr.clip(0, 255).round().astype(np.uint8)
    gg = gg.clip(0, 255).round().astype(np.uint8)
    bb = bb.clip(0, 255).round().astype(np.uint8)

    return np.stack((rr, gg, bb), axis=1)
