from automated_support_ticket_classification.models.evaluate import main as evaluate_main
from automated_support_ticket_classification.models.train import main as train_main


def test_model_scripts_import_and_define_main():
    assert callable(train_main)
    assert callable(evaluate_main)
