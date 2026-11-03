from __future__ import annotations

import json

import pytest

from cks_optimizer.cli import _build_parser
from cks_optimizer.cnn.config import infer_dataset_kind_from_root
from cks_optimizer.cnn.data.base import (
    CocoDatasetAdapter,
    CustomDatasetAdapter,
    ImageFolderDatasetAdapter,
    VocDatasetAdapter,
)
from cks_optimizer.cnn.data.registry import get_dataset_adapter
from cks_optimizer.cnn.tasks import get_task_name


@pytest.fixture
def image_root(tmp_path):
    root = tmp_path / "images"
    class_a = root / "cat"
    class_a.mkdir(parents=True)
    class_b = root / "dog"
    class_b.mkdir(parents=True)
    (class_a / "cat1.png").write_bytes(b"fakepng")
    (class_b / "dog1.jpg").write_bytes(b"fakejpg")
    return root


@pytest.fixture
def coco_root(tmp_path):
    root = tmp_path / "coco"
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "img1.jpg").write_bytes(b"fakejpg")
    annotation_file = root / "instances.json"
    annotation_file.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "images/img1.jpg"}],
                "annotations": [{"image_id": 1, "bbox": [0, 0, 10, 10]}],
            }
        )
    )
    return root


@pytest.fixture
def voc_root(tmp_path):
    root = tmp_path / "voc"
    jpeg_dir = root / "JPEGImages"
    annot_dir = root / "Annotations"
    jpeg_dir.mkdir(parents=True)
    annot_dir.mkdir(parents=True)
    (jpeg_dir / "img1.jpg").write_bytes(b"fakejpg")
    (annot_dir / "img1.xml").write_text("<annotation></annotation>")
    return root


@pytest.fixture
def custom_root(tmp_path):
    root = tmp_path / "custom"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            [
                {"image_path": str(root / "a.png"), "label": "custom-a"},
                {"image_path": str(root / "b.png"), "label": "custom-b"},
            ]
        )
    )
    (root / "a.png").write_bytes(b"fakepng")
    (root / "b.png").write_bytes(b"fakepng")
    return root


def test_imagefolder_adapter_loads_directory_samples(image_root):
    adapter = ImageFolderDatasetAdapter(str(image_root))
    assert len(adapter) == 2
    assert {sample["label"] for sample in adapter.samples} == {"cat", "dog"}
    assert all("image_path" in sample for sample in adapter.samples)


def test_coco_adapter_loads_annotation_file(coco_root):
    adapter = CocoDatasetAdapter(str(coco_root))
    assert len(adapter) == 1
    assert adapter[0]["image_id"] == 1
    assert adapter[0]["annotation_path"].endswith("instances.json")


def test_voc_adapter_loads_jpeg_and_annotation_pairs(voc_root):
    adapter = VocDatasetAdapter(str(voc_root))
    assert len(adapter) == 1
    assert adapter[0]["annotation_path"].endswith("img1.xml")


def test_custom_adapter_loads_manifest_entries(custom_root):
    adapter = CustomDatasetAdapter(str(custom_root))
    assert len(adapter) == 2
    assert {sample["label"] for sample in adapter.samples} == {
        "custom-a",
        "custom-b",
    }


def test_custom_adapter_reads_object_manifest(tmp_path):
    root = tmp_path / "custom"
    root.mkdir(parents=True)
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "items": [
                    {"image_path": str(root / "x.png"), "label": "a"},
                    {"image_path": str(root / "y.png"), "label": "b"},
                ]
            }
        )
    )
    (root / "x.png").write_bytes(b"x")
    (root / "y.png").write_bytes(b"y")

    adapter = CustomDatasetAdapter(str(root))
    assert len(adapter) == 2
    assert {item["label"] for item in adapter.samples} == {"a", "b"}


def test_custom_adapter_reads_csv_manifest(tmp_path):
    root = tmp_path / "custom"
    root.mkdir(parents=True)
    (root / "manifest.csv").write_text(
        "image_path,label\n"
        f"{root / 'a.png'},alpha\n"
        f"{root / 'b.png'},beta\n"
    )
    (root / "a.png").write_bytes(b"a")
    (root / "b.png").write_bytes(b"b")

    adapter = CustomDatasetAdapter(str(root))
    assert len(adapter) == 2
    assert {item["label"] for item in adapter.samples} == {"alpha", "beta"}


def test_imagefolder_adapter_handles_flat_directory_without_class_folders(
    tmp_path,
):
    root = tmp_path / "flat"
    root.mkdir(parents=True)
    (root / "img1.png").write_bytes(b"png")

    adapter = ImageFolderDatasetAdapter(str(root))
    assert len(adapter) == 1
    assert adapter[0]["label"] == "unknown"


def test_voc_adapter_falls_back_to_image_files_when_annotations_missing(
    tmp_path,
):
    root = tmp_path / "voc"
    jpeg_dir = root / "JPEGImages"
    jpeg_dir.mkdir(parents=True)
    (jpeg_dir / "img1.jpg").write_bytes(b"jpg")

    adapter = VocDatasetAdapter(str(root))
    assert len(adapter) == 1
    assert adapter[0]["annotation_path"].endswith("img1.xml")


def test_dataset_registry_rejects_unsupported_kind(tmp_path):
    with pytest.raises(ValueError):
        get_dataset_adapter("unknown", str(tmp_path))


def test_dataset_registry_factory_returns_correct_adapter(image_root):
    adapter = get_dataset_adapter("imagefolder", str(image_root))
    assert isinstance(adapter, ImageFolderDatasetAdapter)

    with pytest.raises(ValueError):
        get_dataset_adapter("unsupported", str(image_root))


def test_task_registry_accepts_expected_names():
    assert get_task_name("classification") == "classification"
    assert get_task_name("Detection") == "detection"
    assert get_task_name("Segmentation") == "segmentation"

    with pytest.raises(ValueError):
        get_task_name("unknown")


def test_adapter_matches_detects_layout_signatures(tmp_path):
    voc_root = tmp_path / "voc"
    (voc_root / "JPEGImages").mkdir(parents=True)
    (voc_root / "Annotations").mkdir()
    assert VocDatasetAdapter.matches(voc_root)
    assert not CocoDatasetAdapter.matches(voc_root)

    coco_root = tmp_path / "coco"
    coco_root.mkdir()
    (coco_root / "instances_val2017.json").write_text("{}")
    assert CocoDatasetAdapter.matches(coco_root)
    assert not VocDatasetAdapter.matches(coco_root)

    custom_root = tmp_path / "custom"
    custom_root.mkdir()
    (custom_root / "notes.json").write_text("{}")
    assert CustomDatasetAdapter.matches(custom_root)
    assert not CocoDatasetAdapter.matches(custom_root)


def test_infer_dataset_kind_respects_registry_priority(tmp_path):
    # VOC layout wins over an instances.json also present in the tree.
    root = tmp_path / "mixed"
    (root / "JPEGImages").mkdir(parents=True)
    (root / "Annotations").mkdir()
    (root / "instances_val2017.json").write_text("{}")

    assert infer_dataset_kind_from_root(root) == "voc"


def test_infer_dataset_kind_falls_back_to_imagefolder(tmp_path):
    plain_root = tmp_path / "plain"
    plain_root.mkdir()
    (plain_root / "img.jpg").write_bytes(b"jpg")

    assert infer_dataset_kind_from_root(plain_root) == "imagefolder"
    assert infer_dataset_kind_from_root(tmp_path / "missing") == "imagefolder"


def test_cnn_parser_accepts_dataset_kind_override():
    parser = _build_parser()

    args = parser.parse_args(
        [
            "cnn",
            "--model",
            "resnet18",
            "--dataset",
            "./tmp",
            "--dataset-kind",
            "coco",
        ]
    )
    assert args.dataset_kind == "coco"
    assert parser.parse_args(
        ["cnn", "--model", "resnet18", "--dataset", "./tmp"]
    ).dataset_kind == "auto"

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "cnn",
                "--model",
                "resnet18",
                "--dataset",
                "./tmp",
                "--dataset-kind",
                "unknown",
            ]
        )
