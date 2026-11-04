import json

from totalsegmentator import config as config_module


def test_setup_totalseg_creates_parent_directories(tmp_path, monkeypatch):
    config_dir = tmp_path / "new" / "nested" / ".totalsegmentator"
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(config_dir))

    config = config_module.setup_totalseg(totalseg_id="totalseg_test")

    expected = {
        "totalseg_id": "totalseg_test",
        "send_usage_stats": True,
        "prediction_counter": 0,
    }
    assert config == expected
    assert json.loads((config_dir / "config.json").read_text()) == expected


def test_setup_totalseg_preserves_existing_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "nested" / ".totalsegmentator"
    config_dir.mkdir(parents=True)
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(config_dir))
    expected = {
        "totalseg_id": "totalseg_existing",
        "send_usage_stats": False,
        "prediction_counter": 7,
        "custom_setting": "keep",
    }
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps(expected))
    original = config_file.read_bytes()

    config = config_module.setup_totalseg(totalseg_id="totalseg_replacement")

    assert config == expected
    assert config_file.read_bytes() == original


def test_set_license_number_skips_validation_for_existing_license(tmp_path, monkeypatch):
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path))
    config_module.setup_totalseg()

    license_number = "aca_12345678901234"
    config_file = tmp_path / "config.json"
    config = json.loads(config_file.read_text())
    config["license_number"] = license_number
    config_file.write_text(json.dumps(config))

    def fail_validation(unused_license_number):
        raise AssertionError("license validation should not be called")

    monkeypatch.setattr(config_module, "is_valid_license", fail_validation)

    config_module.set_license_number(license_number)

    assert json.loads(config_file.read_text())["license_number"] == license_number


def test_set_license_number_validates_changed_license(tmp_path, monkeypatch):
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path))
    config_module.setup_totalseg()

    old_license_number = "aca_12345678901234"
    new_license_number = "aca_56789012345678"
    config_file = tmp_path / "config.json"
    config = json.loads(config_file.read_text())
    config["license_number"] = old_license_number
    config_file.write_text(json.dumps(config))

    validated_license_numbers = []

    def record_validation(license_number):
        validated_license_numbers.append(license_number)
        return True

    monkeypatch.setattr(config_module, "is_valid_license", record_validation)

    config_module.set_license_number(new_license_number)

    assert validated_license_numbers == [new_license_number]
    assert json.loads(config_file.read_text())["license_number"] == new_license_number
