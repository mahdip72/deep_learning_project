from regression_model_lib import compute_detectron_detection
from extract_features import compute_imagenet_backbone_features
from regression_model import regression_model

if __name__ == '__main__':
    """
    given directory of video files and it computing 3 steps
    1- extracting and saving detectron detection
    2- extracting and saving features with pre train imagenet model
    3- training regression model

    """

    compute_detectron_detection(dir='/mnt/external/test_video', gpu='0')
    compute_imagenet_backbone_features(dir='/mnt/external/test_video')
    model = regression_model(dir="/mnt/external/test_video_features", model_name="test_video_dataset")
