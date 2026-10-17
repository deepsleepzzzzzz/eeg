"""Run the five core tests without requiring pytest."""
from _bootstrap import PROJECT_ROOT  # noqa: F401

import test_split_invariants
import test_no_leakage
import test_representation
import test_metrics
import test_golden_regression
import test_protocol_config
import test_bin_edges_scale
import test_protocol_structure
import test_audits

TESTS = [
    ("split_invariants", test_split_invariants.test_main_70_15_15_counts),
    ("split_invariants_old_70_20_10", test_split_invariants.test_old_70_20_10_counts),
    ("no_leakage", test_no_leakage.test_main_protocol_no_window_overlap),
    ("no_duplicate_window_starts", test_no_leakage.test_no_duplicate_window_starts),
    ("representation_shape", test_representation.test_transition_shape_and_normalization),
    ("representation_train_only_edges", test_representation.test_bin_edges_fit_train_only),
    ("metrics_manual", test_metrics.test_metrics_manual),
    ("metrics_perfect", test_metrics.test_metrics_perfect),
    ("golden_regression", test_golden_regression.test_golden_saved_predictions_match_saved_metrics),
    ("protocol_values_from_yaml", test_protocol_config.test_protocol_values_are_loaded_from_yaml),
    ("protocol_loader_follows_changes", test_protocol_config.test_loader_follows_protocol_changes),
    ("bin_edges_scale_full_states", test_bin_edges_scale.test_global_edges_use_full_state_space),
    ("bin_edges_scale_invariance", test_bin_edges_scale.test_pipeline_is_scale_invariant),
    ("bin_edges_more_bins_add_info", test_bin_edges_scale.test_more_bins_add_information),
    ("one_main_protocol", test_protocol_structure.test_there_is_exactly_one_main_protocol),
    ("gcn_steps_eq_lags", test_protocol_structure.test_gcn_steps_equals_number_of_lags),
    ("input_channels_consistent", test_protocol_structure.test_declared_input_channels_match_channels_times_lags),
    ("channel_indices_in_range", test_protocol_structure.test_channel_indices_within_csv_width),
    ("declared_window_counts", test_protocol_structure.test_declared_window_counts_match_reality),
    ("audit_main", test_audits.test_audit_main_passes),
    ("audit_clean", test_audits.test_audit_clean_passes),
    ("audit_detects_injected_faults", test_audits.test_audit_detects_injected_faults),
]


def main():
    passed = 0
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception as exc:
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"RESULT passed={passed} failed={failed}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
