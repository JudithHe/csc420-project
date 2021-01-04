import cv2
import numpy as np
from keras.models import load_model
from layers import BilinearUpSampling2D
from skimage.transform import resize


def get_depth(model_str, img, savePath):
    """
    Generate the depth map of img using pre-trained model.
    :param model: pre-trained model want to use
    :param img: input image
    :param savePath: the file name you wish to save the output file
    """
    # load pre-trained model
    # custom layer BilinearUpSampling2D
    model = load_model(model_str, custom_objects={'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}, compile=False)

    # read input image
    im = cv2.imread(img)
    img_read = resize(im, (im.shape[0], im.shape[1]), anti_aliasing=False)

    # use model to predict depth of every pixel
    predict = model.predict(img_read.reshape((1, img_read.shape[0], img_read.shape[1], img_read.shape[2])), batch_size=2)
    # (1 * 288 * 384 * 1) -> (1 * 288 * 384)
    if model_str == 'models/nyu.h5':
        outputs = np.squeeze(500 / predict, axis=3)
    else:
        outputs = np.squeeze(1000 / predict, axis=3)

    oriImg = cv2.resize(cv2.imread(img), (im.shape[1] // 2, im.shape[0] // 2))

    # save outputs
    cv2.imwrite(savePath + ' InputImg.png', oriImg)
    cv2.imwrite(savePath + ' DepthImg.png', outputs[0])


if __name__ == "__main__":
    img1 = "../Images/teapot1/pointcloud.txt"
    img2 = "../Images/teapot2/pointcloud.txt"

    # plot kitti.h5
    get_depth("models/kitti.h5", img1, "kitti teapot1")
    get_depth("models/kitti.h5", img2, "kitti teapot2")

    # plot NYU
    get_depth('models/nyu.h5', img1, "nyu teapot1")
    get_depth('models/nyu.h5', img2, "nyu teapot2")