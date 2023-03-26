from kedro_image_classification.pipelines.data_download.nodes import dummy_download


def test_dummy_download():
    """Test checking dummy data download"""
    data = dummy_download()

    assert data == {}
