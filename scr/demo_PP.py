import imageio
from skimage import img_as_ubyte
from demo import load_checkpoints
from demo import make_animation
from skimage.transform import resize
import paddle.fluid.dygraph as dygraph
import paddle.fluid as fluid
import warnings
warnings.filterwarnings("ignore")

source_image = imageio.imread('/home/aistudio/test_moving-gif.jpg')
driving_video = imageio.mimread('/home/aistudio/work/dataset/moving-gif/train/00454.gif')

#Resize image and video to 256x256
source_image = resize(source_image, (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
with dygraph.guard(fluid.CUDAPlace(0)):
    generator, kp_detector = load_checkpoints(config_path='config/mgif-256.yaml')
    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

#save resulting video
imageio.mimsave('../generated_Fashion_2.mp4', [img_as_ubyte(frame) for frame in predictions])