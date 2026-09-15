import importlib


def test_hit_preprocessing_contract_is_defined_from_repo():
    try:
        mod = importlib.import_module('hit_preprocessing')
    except ImportError:
        raise AssertionError('hit_preprocessing.py is expected to exist')

    spec = mod.get_training_contract()
    assert spec['channels'] == 2
    assert spec['raw_sequence_length'] == 175
    assert spec['cropped_sequence_length'] == 50
    assert spec['window_ms'] == (25, 75)
    assert 'mean/std' in spec['normalization']
