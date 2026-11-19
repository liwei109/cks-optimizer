from PIL import Image

from cks_optimizer.cnn import OptimizeConfig, data_process


def test_data_process(tmp_path):
    for split in ("train", "val"):
        image_dir = tmp_path / split / "cat"
        image_dir.mkdir(parents=True)
        Image.new("RGB", (16, 16)).save(image_dir / "sample.png")

    config = OptimizeConfig(
        model_name="resnet18",
        dataset_root=str(tmp_path),
        input_shape=(1, 3, 16, 16),
        device="cpu",
    )
    data, sample_input = data_process(config, include_evaluation=True)

    assert data.calib_data[0].shape == (1, 3, 16, 16)
    assert sample_input is data.calib_data[0]
    assert len(data.evaluation_dataset) == 1

    data, sample_input = data_process(
        config, include_calibration=False, include_evaluation=True
    )
    assert data.calib_data == []
    assert data.num_samples == 0
    assert sample_input.shape == (1, 3, 16, 16)
