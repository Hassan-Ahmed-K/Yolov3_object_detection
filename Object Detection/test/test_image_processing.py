import sys

sys.path.append('/home/hassan-ahmed-khan/Ai/Yolo/Object_Detection/src')

import pytest
import numpy as np
from image_processing import load_and_preprocess_image,cropped_detected_object,load_and_preprocess_frame
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import os
from unittest.mock import patch, MagicMock
import pytest

# Variables for tests
VALID_IMAGE_PATH = "/home/hassan-ahmed-khan/Ai/Yolo/Object_Detection/images/kangaroo.png"
INVALID_IMAGE_PATH = "Object_Detection/images/kangaroo1.png"
SHAPE_SMALL = (32, 32)
SHAPE_LARGE = (256, 256)
NON_EXISTENT_PATH = "/images/kangaroo.png"

@pytest.mark.parametrize("test_id, image_path, shape, expected_shape", [
    ("happy_path_small", VALID_IMAGE_PATH, SHAPE_SMALL, SHAPE_SMALL),
    ("happy_path_large", VALID_IMAGE_PATH, SHAPE_LARGE, SHAPE_LARGE),
])

def test_load_and_preprocess_image_happy_path(test_id, image_path, shape, expected_shape):
    original_image = load_img(image_path)
    original_width, original_height = original_image.size

    processed_image, width, height = load_and_preprocess_image(image_path, shape)
    assert processed_image.shape == (shape[0], shape[1], 3), f"Test ID {test_id}: Image shape is not as expected."
    assert width == original_width, f"Test ID {test_id}: Image width is not as expected."
    assert height == original_height, f"Test ID {test_id}: Image height is not as expected."
    assert processed_image.numpy().max() <= 1 , f"Test ID {test_id}: Image max value should be normalized to less than or equal 1.0."
    assert processed_image.numpy().min() >= 0 , f"Test ID {test_id}: Image min value should be normalized to greater than or equal 0."

@pytest.mark.parametrize("test_id, image_path, shape", [
    ("error_case_invalid_shape", VALID_IMAGE_PATH, (0, 0)),
    ("edge_case_empty_path", "", SHAPE_SMALL),
    ("edge_case_none_path", None, SHAPE_SMALL),
    ("edge_case_invalid_path", INVALID_IMAGE_PATH, SHAPE_SMALL),
    ("edge_case_nonexistent_path", NON_EXISTENT_PATH, SHAPE_SMALL),
    ("error_case_invalid_shape_type", VALID_IMAGE_PATH, "invalid_shape"),
    ("error_case_invalid_channels", VALID_IMAGE_PATH, (32, 32, 4)),
])

def test_load_and_preprocess_image_edge_cases(test_id, image_path, shape):
    with pytest.raises(Exception):
        # Act
        load_and_preprocess_image(image_path, shape)



# Define Variables

SHAPE_INVALID_CHANNELS = (32, 32, 4)
SHAPE_INVALID_TYPE = "invalid_shape"

@pytest.mark.parametrize("frame, shape, expected_shape", [
    (np.random.rand(480, 640, 3), (224, 224), (224, 224, 3)),  # ID: Normal-Case
    (np.random.rand(100, 100, 3), (50, 50), (50, 50, 3)),  # ID: Small-Frame
    (np.random.rand(200, 300, 3), (150, 100), (150, 100, 3)),  # ID: Non-Square-Shape
])
def test_load_and_preprocess_frame(frame, shape, expected_shape):
    # Act
    result_image, result_width, result_height = load_and_preprocess_frame(frame, shape)

    # Assert
    assert result_image.shape == expected_shape
    assert result_width ==  frame.shape[1]
    assert result_height == frame.shape[0]
    assert np.max(result_image) <= 1.0  # Check if normalized (pixel values <= 1.0)
    assert np.min(result_image) >= 0.0  # Check if normalized (pixel values >= 0.0)


# ERROR CASES
@pytest.mark.parametrize("test_id, image_path, shape", [
    ("error_case_invalid_shape", VALID_IMAGE_PATH, (0, 0)),
    ("edge_case_empty_path", "", SHAPE_SMALL),
    ("edge_case_none_path", None, SHAPE_SMALL),
    ("edge_case_invalid_path", INVALID_IMAGE_PATH, SHAPE_SMALL),
    ("edge_case_nonexistent_path", NON_EXISTENT_PATH, SHAPE_SMALL),
    ("error_case_invalid_shape_type", VALID_IMAGE_PATH, SHAPE_INVALID_TYPE),
    ("error_case_invalid_channels", VALID_IMAGE_PATH, SHAPE_INVALID_CHANNELS),
])

def test_load_and_preprocess_frame_edge_cases(test_id, image_path, shape):
    with pytest.raises(Exception):
        # Act
        load_and_preprocess_frame(image_path, shape)






BASE_DIR = '/home/hassan-ahmed-khan/Ai/Yolo/Object_Detection/ouput'  # Replace with the actual base directory used in your code

@pytest.fixture
def mock_image_open(request):
    mock_open = patch("PIL.Image.open", autospec=True)
    request.addfinalizer(mock_open.stop)
    return mock_open.start()


@pytest.fixture
def mock_os_mkdir():
    with patch("os.mkdir") as mock_mkdir:
        yield mock_mkdir

@pytest.fixture
def mock_os_listdir():
    with patch("os.listdir", return_value=[]):
        yield

@pytest.fixture
def mock_cropped_img_save():
    with patch("PIL.Image.Image.save") as mock_save:
        yield mock_save

def test_cropped_detected_object_no_save(mock_image_open, mock_os_mkdir, mock_os_listdir, mock_cropped_img_save):
    mock_image_open.return_value = MagicMock()
    image_path = "images/kangaroo.png"
    boxes = [(0, 0, 100, 100), (50, 50, 150, 150)]
    labels = ['label1', 'label2']

    cropped_detected_object(image_path, boxes, labels, save=False)

    mock_image_open.assert_called_once_with(image_path)
    mock_os_mkdir.assert_not_called()
    mock_cropped_img_save.assert_not_called()

def test_cropped_detected_object_with_save():
    with patch("PIL.Image.open") as mock_image_open:
        # Mock the necessary dependencies for testing
        mock_image_open.return_value = MagicMock()

        image_path = "images/kangaroo.png"
        boxes = [(0, 0, 100, 100), (50, 50, 150, 150)]
        labels = ['label1', 'label2']

        cropped_detected_object(image_path, boxes, labels, save=True)

    print(os.path.join(BASE_DIR, 'ouput', labels[0], f'{labels[0]}(2).jpg'))
    mock_image_open.assert_called_once_with(image_path)
    mock_os_mkdir.assert_called_once_with(os.path.join(BASE_DIR, 'ouput', labels[0]))
    mock_cropped_img_save.assert_called_once_with(os.path.join(BASE_DIR, 'ouput', labels[0], f'{labels[0]}(2).jpg'))
