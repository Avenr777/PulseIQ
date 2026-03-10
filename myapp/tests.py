from unittest.mock import Mock, patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from myapp.models import Asset, PowerReading


class ModelParsingTests(TestCase):
    def test_parse_model_filename_handles_multi_part_reading_type(self):
        from myapp.ml_models import parse_model_filename

        asset_name, reading_type = parse_model_filename(
            "transformer_block_a_power_w_model.pkl"
        )

        self.assertEqual(asset_name, "transformer_block_a")
        self.assertEqual(reading_type, "power_w")

    def test_parse_model_filename_rejects_unknown_suffix(self):
        from myapp.ml_models import parse_model_filename

        asset_name, reading_type = parse_model_filename("bad_name_model.pkl")

        self.assertIsNone(asset_name)
        self.assertIsNone(reading_type)


class ElectricityViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="tester",
            password="safe-pass-123",
        )
        self.client.force_login(self.user)
        self.asset = Asset.objects.create(name="transformer_block_a")

        for idx in range(6):
            PowerReading.objects.create(
                asset=self.asset,
                reading_type="power_w",
                value=100 + idx,
            )

    @patch("myapp.views.asset_models")
    def test_electricity_view_sets_assets_in_anomaly_context(self, mock_asset_models):
        mock_model = Mock()
        mock_model.predict.return_value = [-1]

        mock_scaler = Mock()
        mock_scaler.transform.return_value = [[0.1, 0.2, 0.3]]

        mock_asset_models.get.return_value = {
            "power_w": {
                "model": mock_model,
                "scaler": mock_scaler,
            }
        }

        response = self.client.get(
            reverse("electricity"),
            {"asset": self.asset.name, "reading_type": "power_w"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["selected_asset"], self.asset.name)
        self.assertEqual(response.context["assets_in_anomaly"], 1)
        self.assertContains(response, "Assets in anomaly")
