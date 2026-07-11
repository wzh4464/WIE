import unittest


class TestInflImportBridge(unittest.TestCase):
    def test_experiment_infl_cli_factory_exposed(self):
        # Import CLI shim and ensure InfluenceCalculatorFactory is exposed
        from wie.infl.cli import InfluenceCalculatorFactory, main  # noqa: F401

        self.assertTrue(hasattr(InfluenceCalculatorFactory, "create"))

    def test_td_influence_registered_in_src_factory(self):
        # Import src factory and ensure 'td_influence' is registered when package is imported
        from wie.infl.core import InfluenceCalculatorFactory

        # Import the package to trigger registration decorators
        import wie.infl  # noqa: F401

        self.assertIn("td_influence", InfluenceCalculatorFactory._calculators)

    def test_experiment_infl_td_imports_class(self):
        # Ensure experiment path exposes the TDInfluenceCalculator class
        from wie.infl.td_influence import TDInfluenceCalculator
        from wie.infl.core import InfluenceCalculator

        self.assertTrue(issubclass(TDInfluenceCalculator, InfluenceCalculator))


if __name__ == "__main__":
    unittest.main()
