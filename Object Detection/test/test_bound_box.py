import sys

sys.path.append('/home/hassan-ahmed-khan/Ai/Yolo/Object_Detection/src')

import unittest
import numpy as np
from bound_box import *
from unittest.mock import patch,MagicMock
import cv2
import pytest

class TestBoundBox(unittest.TestCase):

    def test_bound_box_initialization(self):
        xmin, ymin, xmax, ymax = 10, 20, 30, 40
        objness, classes = 0.8, [0.2, 0.5, 0.3]

        box = BoundBox(xmin, ymin, xmax, ymax, objness, classes)

        # Check if the attributes are set correctly
        self.assertEqual(box.xmin, xmin)
        self.assertEqual(box.ymin, ymin)
        self.assertEqual(box.xmax, xmax)
        self.assertEqual(box.ymax, ymax)
        self.assertEqual(box.objness, objness)
        self.assertEqual(box.classes, classes)
        self.assertEqual(box.label, -1)
        self.assertEqual(box.score, -1)

    def test_get_label(self):
        classes = [0.2, 0.5, 0.8]
        box = BoundBox(0, 0, 0, 0, classes=classes)

        # Check if get_label returns the index with the maximum score
        self.assertEqual(box.get_label(), 2)

    def test_get_score(self):
        classes = [0.2, 0.5, 0.8]
        box = BoundBox(0, 0, 0, 0, classes=classes)

        # Check if get_score returns the score of the predicted class
        self.assertEqual(box.get_score(), 0.8)

class TestDecodeNetout(unittest.TestCase):

    def test_decode_netout(self):
        # Define mock input values for testing
        netout = np.random.rand(13, 13, 9, 15)  # Example shape (grid_h, grid_w, nb_box, nb_class + 5)
        anchors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # Example anchor values
        net_h, net_w = 416, 416  # Example network height and width

        # Call the decode_netout function
        boxes = decode_netout(netout, anchors, net_h, net_w)

        # Assertions
        self.assertIsInstance(boxes, list)
        self.assertTrue(all(isinstance(box, BoundBox) for box in boxes))

        # Add more assertions based on the expected behavior of the function

class TestBoxFilter(unittest.TestCase):

    def setUp(self):
        # Set up any required resources or configurations before each test
        pass

    def tearDown(self):
        # Clean up any resources or configurations after each test
        pass

    def test_box_filter(self):
        # Define mock input values for testing
        mock_box = BoundBox(0.1, 0.2, 0.4, 0.6, objness=0.8, classes=[0.1, 0.9, 0.7])
        boxes = [mock_box]
        labels = ["label1", "label2", "label3"]
        threshold_score = 0.5

        # Call the box_filter function
        valid_boxes, valid_labels, valid_scores = box_filter(boxes, labels, threshold_score)

        # Print relevant information for debugging
        print("Actual valid_boxes:", valid_boxes)
        print("Expected valid_boxes:", [mock_box])

        # Assertions
        self.assertIsInstance(valid_boxes, list)
        self.assertIsInstance(valid_labels, list)
        self.assertIsInstance(valid_scores, list)

        self.assertEqual(len(valid_boxes), len(valid_labels))
        self.assertEqual(len(valid_scores), len(valid_labels))

class TestDrawBoxes:
    @pytest.fixture
    def sample_data(self):
        # Define sample data for testing
        boxes = [BoundBox(10, 20, 50, 80), BoundBox(60, 30, 120, 90)]
        labels = ['cat', 'dog']
        scores = [0.8, 0.9]
        return boxes, labels, scores

    @patch("cv2.imshow", new_callable=MagicMock)
    def test_draw_boxes_with_sample_data(self, mock_imshow, sample_data, tmp_path):
        # Arrange
        boxes, labels, scores = sample_data
        valid_data = (boxes, labels, scores)
        image_path = "/home/hassan-ahmed-khan/Ai/Yolo/Object_Detection/images/kangaroo.png"

        # Act
        result_boxes, result_labels = draw_boxes(image_path, valid_data)

        # Assert
        assert isinstance(result_boxes, list)
        assert isinstance(result_labels, list)

        # Additional assertions based on your requirements
        assert len(result_boxes) == len(boxes)
        assert len(result_labels) == len(labels)

        # Check if cv2.imshow is called
        mock_imshow.assert_called_once()

class TestEncoderDic(unittest.TestCase):

    def test_encoder_dic(self):
        # Mock data
        mock_box1 = BoundBox(0.1, 0.2, 0.4, 0.6, objness=0.8, classes=[0.1, 0.9, 0.7])
        mock_box2 = BoundBox(0.2, 0.3, 0.5, 0.7, objness=0.7, classes=[0.2, 0.8, 0.6])
        valid_data = ([mock_box1, mock_box2], ["label1", "label2"], [0.8, 0.7])

        # Call the encoder_dic function
        result_dic = encoder_dic(valid_data)

        # Assertions
        self.assertIsInstance(result_dic, dict)
        self.assertIn("label1", result_dic)
        self.assertIn("label2", result_dic)
        self.assertEqual(len(result_dic["label1"]), 1)
        self.assertEqual(len(result_dic["label2"]), 1)

        # Add more assertions based on the expected behavior of the function

class TestDecodeBoxCoor(unittest.TestCase):

    def test_decode_box_coor(self):
        # Mock data
        mock_box = BoundBox(0.1, 0.2, 0.4, 0.6, objness=0.8, classes=[0.1, 0.9, 0.7])

        # Call the decode_box_coor function
        result = decode_box_coor(mock_box)

        # Assertions
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0], 0.1, places=6)
        self.assertAlmostEqual(result[1], 0.2, places=6)
        self.assertAlmostEqual(result[2], 0.4, places=6)
        self.assertAlmostEqual(result[3], 0.6, places=6)

class TestIoU(unittest.TestCase):

    def test_iou(self):
        # Mock data
        mock_box1 = BoundBox(0.1, 0.2, 0.4, 0.6, objness=0.8, classes=[0.1, 0.9, 0.7])
        mock_box2 = BoundBox(0.3, 0.4, 0.6, 0.8, objness=0.7, classes=[0.2, 0.8, 0.6])

        # Call the iou function
        result = iou(mock_box1, mock_box2)

        # Assertions
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

class TestDoNMS(unittest.TestCase):

    def test_do_nms(self):
        # Mock data
        mock_data_dic = {
            'label1': [[0.8, BoundBox(0.1, 0.2, 0.4, 0.6), 'kept'],
                       [0.7, BoundBox(0.3, 0.4, 0.6, 0.8), 'kept']],
            'label2': [[0.9, BoundBox(0.2, 0.3, 0.5, 0.7), 'kept'],
                       [0.6, BoundBox(0.4, 0.5, 0.7, 0.9), 'kept']]
        }
        nms_thresh = 0.5

        # Call the do_nms function
        result = do_nms(mock_data_dic, nms_thresh)

        # Assertions
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)



if __name__ == '__main__':
    unittest.main()
