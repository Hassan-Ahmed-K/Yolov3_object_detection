import sys

sys.path.append('/home/hassan-ahmed-khan/Ai/Yolo/Object_Detection/src')

import pytest
from unittest.mock import patch, mock_open,Mock
from util import download_image,download_weight



# Sample URL and destination path for testing
SAMPLE_URL = 'https://storagecdn.strathcona.ca/files/filer_public_thumbnails/images/tas-medium-trafficsignals-intersection-660x396.jpg__660.0x396.0_q85_subsampling-2.jpg'
SAMPLE_DESTINATION_PATH = "Object_Detection/images/abc.download"

# Parametrize test cases with different response status codes
@pytest.mark.parametrize("status_code", [200, 404, 500])
def test_download_image(status_code):
    with patch('requests.get') as mock_get, \
         patch('builtins.open', new_callable=mock_open) as mock_open_file, \
         patch('builtins.print') as mock_print:
        # Mocking the response and open file
        mock_response = mock_get.return_value
        mock_response.status_code = status_code
        mock_response.content = b'Mock image content'

        # Act
        download_image(SAMPLE_URL, SAMPLE_DESTINATION_PATH)

        # Assert based on response status code
        if status_code == 200:
            mock_open_file.assert_called_once_with(SAMPLE_DESTINATION_PATH, 'wb')
            mock_open_file().write.assert_called_once_with(b'Mock image content')
            mock_print.assert_called_once_with(f"Image downloaded successfully to {SAMPLE_DESTINATION_PATH}")
        else:
            assert not mock_open_file.called
            mock_print.assert_called_once_with(f"Failed to download image. Status code: {status_code}")   
            



@pytest.fixture
def mock_os_listdir():
    with patch("os.listdir", return_value=[]):
        yield

@pytest.fixture
def mock_wget_download():
    with patch("wget.download") as mock_download:
        yield mock_download

def test_download_weight_successful(mock_os_listdir, mock_wget_download, capsys):
    mock_os_listdir = patch("os.listdir", return_value=[])
    download_weight()
    captured = capsys.readouterr()
    assert "Downloading ..." in captured.out
    mock_wget_download.assert_called_once()

def test_download_weight_exception(mock_os_listdir, mock_wget_download, capsys):
    mock_wget_download.side_effect = Exception("Download failed")
    download_weight()
    captured = capsys.readouterr()
    assert "Error: Download failed" in captured.out