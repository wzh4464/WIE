import unittest


class TestDVEInfluenceSimple(unittest.TestCase):
    """Simplified test suite for DVE influence calculator registration."""

    def test_dve_registration(self):
        """Test that DVE calculator is properly registered."""
        from wie.infl import InfluenceCalculatorFactory

        # Import to trigger registration
        import wie.infl.dve  # noqa: F401

        # Check registration
        self.assertIn("dve", InfluenceCalculatorFactory._calculators)

        cls = InfluenceCalculatorFactory._calculators["dve"]
        from wie.infl.dve import DVEInfluenceCalculator

        self.assertIs(cls, DVEInfluenceCalculator)

    def test_dve_available_in_cli(self):
        """Test that DVE is available as an influence type."""
        from wie.infl import InfluenceCalculatorFactory

        # Import DVE to trigger registration
        import wie.infl.dve  # noqa: F401

        available_types = list(InfluenceCalculatorFactory._calculators.keys())
        self.assertIn("dve", available_types)

        # Verify at least these key types are available
        expected_types = ["lava", "dve", "sgd", "tracin"]
        for expected_type in expected_types:
            self.assertIn(
                expected_type,
                available_types,
                f"Expected influence type '{expected_type}' not found in {available_types}",
            )

    def test_dve_class_methods(self):
        """Test that DVE class has required methods."""
        from wie.infl.dve import DVEInfluenceCalculator

        # Check required methods exist
        required_methods = ["calculate", "_get_infl_type", "run"]
        for method in required_methods:
            self.assertTrue(
                hasattr(DVEInfluenceCalculator, method),
                f"DVEInfluenceCalculator missing method: {method}",
            )

        # Check abstract method implementation
        calc = DVEInfluenceCalculator.__new__(DVEInfluenceCalculator)
        self.assertEqual(calc._get_infl_type(), "dve")


if __name__ == "__main__":
    unittest.main()
